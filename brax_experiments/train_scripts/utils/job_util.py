import os
import signal
import time
import sys
import flax

import wandb

def slurm_time_to_seconds(time_str:str)->int:
  if '-' in time_str:
    days, time = time_str.split('-')
  else:
    days = 0
    time = time_str
  hours, minutes, seconds = time.split(':')
  time_str = days * (24 * 3600) + int(hours) * 3600 + int(minutes) * 60 + int(seconds)
  return time_str

def gather_slurm_metadata(get_gpu_model=True):
  if "SLURM_JOB_ID" in os.environ:
    slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
    slurm_data = {}
    for k in slurm_env_keys:
      d_key = k.replace("SLURMD_", "SLURM_")
      slurm_data[d_key] = os.environ[k]
    if get_gpu_model:
      slurm_data["GPU_MODEL"] = get_gres_on_node(slurm_data) #TODO: not tested
    return slurm_data
  return None

def get_gres_on_node(slurm_data):
  if "SLURM_JOB_ID" in slurm_data:
    import subprocess
    slurm_node_name = slurm_data.get('SLURM_JOB_NODELIST', None)
    if slurm_node_name is not None:
      cmd = f"scontrol show node {slurm_node_name} | grep Gres"
      gres_string = subprocess.check_output(cmd, shell=True).decode("utf-8")
      return gres_string.split("=")[1].strip()
  return 'N/A'

def get_job_state():
  if "SLURM_JOB_ID" in os.environ:
    import subprocess
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = f"scontrol show jobid {job_id} | grep JobState"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    job_state = string.split("=")[1].split()[0].strip()
    return job_state
  return None

def job_preempted():
  if "SLURM_JOB_ID" in os.environ:
    import subprocess
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = f"scontrol show jobid {job_id} | grep PreemptTime"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    prempt_time = string.split("=")[-1].strip()
    if prempt_time.lower() != "none":
      preempted = True
    else:
      preempted = False
    return preempted
  return None

def get_job_runtime():
  if "SLURM_JOB_ID" in os.environ:
    import subprocess
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = f"scontrol show jobid {job_id} | grep RunTime"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    runtime = string.split("=")[1].split()[0].strip()
    runtime = slurm_time_to_seconds(runtime)# convert to seconds
    return runtime
  return None

def get_job_timelimit():
  if "SLURM_JOB_ID" in os.environ:
    import subprocess
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = f"scontrol show jobid {job_id} | grep TimeLimit"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    timelimit = string.split("=")[2].split()[0].strip()
    timelimit = slurm_time_to_seconds(timelimit)# convert to seconds
    return timelimit
  return None

def save_agent_state(checkpoint_path, agent_state, args, runstate_meta):
  agent_state = flax.jax_utils.unreplicate(agent_state)
  with open(checkpoint_path, "wb") as f:
    f.write(
      flax.serialization.to_bytes(
        [
          agent_state,
          # args,
          # runstate_meta,
        ],
      )
    )
  print(f"model saved to {checkpoint_path}")

def restore_agent_state(checkpoint_path, agent_state, runstate_meta):
  print(f'Restoring from: {checkpoint_path}')
  with open(checkpoint_path, "rb") as f:
    # agent_state, _, runstate_meta = \
    #   flax.serialization.from_bytes(
    #     (
    #       agent_state,
    #       # None,
    #       # runstate_meta,
    #     ),
    #     f.read(),
    #   )
    agent_state = flax.serialization.from_bytes(agent_state, f)
  return agent_state, {}

class RunState:
  # exit codes, not standard but used to communicate with sbatch script
  exit_code_no_requeue = 2
  exit_code_requeue = 3
  timeout = 3600 # 1 hour - not used in the current logic

  # This is how preemption works in SLURM:
  # if partition.GraceTime > 0:
  #   sends SIGCONT, job.PremptTime is set
  #   sleep(GraceTime)
  # sends SIGTERM, job.State is set to COMPLETING
  # sleep(config.KillWait) # default 30s
  # sends SIGKILL, job.State is set to PREEMPTED/FAILED/COMPLETED

  # state variables "controlled" by RunState, based on signal received
  # RunState.apply_signals() will act on these variables
  _check_preempted_soon = False # set to True when we want to check if job is preempted and save checkpoint at next opportunity. Takes precedence over save_soon, kill_soon and sleep_soon
  _save_soon = False # set to True when we want to save at next opportunity. Takes precedence over kill_soon and sleep_soon
  _kill_soon = False # set to True when we want to kill the job at next opportunity. Takes precedence over sleep_soon
  _sleep_soon = False # set to True when we want to sleep at next opportunity
  _requeue = False # set to True when we want to send exit_code_requeue when killing the job

  # "observed" state variables, their value is only affected by elements outside of RunState
  _preempted = False
  _training_completed = False
  _eval_completed = False
  _learner_policy_version = 0

  _sigint_received = False
  _sigcont_received = False
  _sigterm_received = False

  def __init__(self, model_path, save_fn, to_close=None, wandb_sweep=False):
    # desired behavior
    # - receive SIGINT (on timeout, or using scancel --signal=INT) : save checkpoint + terminate with exit code exit_code_requeue -> Wandb state: FAILED
    # - receive SIGCONT or SIGTERM + job_preempted() (on job pre-emption) : save checkpoint + terminate with exit code exit_code_requeue -> Wandb state: FAILED
    # - receive SIGCONT or SIGTERM + not job_preempted() (on scancel): terminate with exit code exit_code_no_requeue -> Wandb state: FAILED
    signal.signal(signal.SIGINT, self._on_sigint)
    signal.signal(signal.SIGCONT, self._on_sigcont)
    signal.signal(signal.SIGTERM, self._on_sigterm)
    self.model_path = model_path
    self.save_fn = save_fn
    self.to_close = to_close if to_close is not None else []
    self.wandb_sweep = wandb_sweep

  @property
  def metadata(self):
    return {
      'learner_policy_version': self._learner_policy_version,
      'training_completed': self._training_completed,
      'eval_completed': self._eval_completed,
    }

  @metadata.setter
  def metadata(self, meta_dict):
    self._learner_policy_version = meta_dict['learner_policy_version']
    self._training_completed = meta_dict['training_completed']
    self._eval_completed = meta_dict['eval_completed']

  def save_state(self, agent_state, args):
    if not isinstance(args, dict):
      args = vars(args)
    self.save_fn(self.model_path, agent_state, args, self.metadata)
    self._save_soon = False

  def apply_signals(self, learner_policy_version, agent_state, args):
    args = vars(args)
    if args.get('local_rank', 0) != 0:
      return
    self._learner_policy_version = learner_policy_version
    if self._check_preempted_soon:
      if self._check_job_preempted():
        self._preempted = True
        self._save_soon = True
    if args.get('preemptible', False) and self._save_soon:
      self.save_state(agent_state, args)
      self._requeue = True
    if self._kill_soon:
      self.kill()

  def kill(self):
    exit_code = self.exit_code_requeue if self._requeue else self.exit_code_no_requeue
    print(f"Killed by RunState. Received SIGCONT: {self._sigcont_received} | Received SIGINT: {self._sigint_received} "
          f"| Received SIGTERM: {self._sigterm_received} \n"
          f"Preempted: {self._preempted}. Requeue: {self._requeue}. Exit code: {exit_code}")
    time.sleep(60) #leave some time for logging etc to finish
    self.close()
    sys.exit(exit_code)

  def close(self):
    for entity in self.to_close:
      if hasattr(entity, 'close'):
        entity.close()

  # this function is not used in the current logic
  def sleep_until_timeout(self):
    self.close()
    while True:
      time.sleep(1)
      self.timeout -= 1
      if self.timeout <= 0:
        break
    sys.exit(f"Exceeded max sleep timeout ({self.timeout} s). This should not happen!")

  def _on_sigcont(self, signum, frame):
    self._sigcont_received = True
    if self.wandb_sweep:
      wandb.mark_preempting()
    self._check_preempted_soon = True
    self._kill_soon = True

  def _on_sigint(self, signum, frame):
    self._sigint_received = True
    if self.wandb_sweep:
      wandb.mark_preempting()
    self._save_soon = True
    self._kill_soon = True

  def _on_sigterm(self, signum, frame):
    self._sigterm_received = True
    #only act if no other signal has been received, then act as if SIGCONT was received
    if not self._sigint_received or not self._sigcont_received:
      if self.wandb_sweep:
        wandb.mark_preempting()
      self._check_preempted_soon = True
      self._kill_soon = True

  def _check_job_preempted(self):
    time.sleep(1) # make sure slurm has time to broadcast info
    return job_preempted()

  def after_training(self, agent_state, args):
    args = vars(args)
    self._training_completed = True
    self._learner_policy_version = -1
    if args.get('save_model', False) and args.get('local_rank', 0) == 0:
      self.save_state(agent_state, args)
      self._requeue = True

  def after_eval(self, agent_state, args):
    args = vars(args)
    self._eval_completed = True
    if args.get('save_model', False) and args.get('local_rank', 0) == 0:
      self.save_state(agent_state, args)
    self._requeue = False

  @property
  def training_completed(self):
    return self._training_completed

  @property
  def eval_completed(self):
    return self._eval_completed

  @property
  def completed(self):
    return self._training_completed and self._eval_completed


# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose file interface to support different backing file APIs."""


import glob
import os
from typing import List


def Glob(pattern: str) -> List[str]:
  return glob.glob(pattern)


def Exists(pathname: str) -> bool:
  """Check if a path exists."""
  exist = None
  if exist is None:
    exist = os.path.exists(pathname)
  return exist


def MakeDirs(dirname: str):
  """Make directory if it doesn't exist."""
  exist_dir = False
  if not exist_dir:
    os.makedirs(dirname, exist_ok=True)


class File:
  """General purpose file resource."""

  def __init__(self, fileName: str, mode='r'):
    self.f = None
    if not self.f:
      self.f = open(fileName, mode)

  def __enter__(self):
    return self.f

  def __exit__(self, exc_type, exc_value, traceback):
    self.f.close()



# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading/saving of inference functions."""

import pickle
from typing import Any

from brax.v1.io.file import File


def load_params(path: str) -> Any:
  with File(path, 'rb') as fin:
    buf = fin.read()
  return pickle.loads(buf)


def save_params(path: str, params: Any):
  """Saves parameters in Flax format."""
  with File(path, 'wb') as fout:
    fout.write(pickle.dumps(params))
