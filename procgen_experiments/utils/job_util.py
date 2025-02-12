import os
import signal
import time
import sys
import flax
import uuid
import wandb
import copy
import subprocess

def generate_run_name(args, run_base_dir="runs", resume_only=False):
  run_name_no_uuid = f"{args.env_id}__{args.exp_name}__{args.seed}"
  run_name = None
  resumed = False
  if resume_only or args.start_from_checkpoint:
    runlist = [d for d in os.listdir(run_base_dir) if os.path.isdir(os.path.join(run_base_dir, d))]
    # check for most recent matching run folder in runs (may have different uuid)
    runlist = [run for run in runlist if run.startswith(run_name_no_uuid)]
    if runlist:
      prev_rundir = max(runlist, key=lambda d: os.path.getmtime(os.path.join(run_base_dir, d)))
      print(f"Found previous run: {prev_rundir}")
      if os.path.exists(f"{run_base_dir}/{prev_rundir}/{args.exp_name}.ckpt"):
        resumed = True
        run_name = str(prev_rundir)
        print(f"Found existing checkpoint in {run_name}. Starting from checkpoint.")
      else:
        print(f"No checkpoint in matching run folder {str(prev_rundir)}.")

  if not resumed and not resume_only:
    run_name = f"{run_name_no_uuid}__{uuid.uuid4()}"
    print(f"Starting new run: {run_name}")

  if run_name is None:
    raise ValueError(f"No run name generated. Probably no match for {run_name_no_uuid} found in {run_base_dir}.")

  return run_name, resumed

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
      slurm_data["GPU_MODEL"] = get_gres_on_node(slurm_data)
    return slurm_data
  return None

def get_gres_on_node(slurm_data):
  if "SLURM_JOB_ID" in slurm_data:
    slurm_node_name = slurm_data.get('SLURM_JOB_NODELIST', None)
    if slurm_node_name is not None:
      cmd = f"scontrol show node {slurm_node_name} | grep Gres"
      gres_string = subprocess.check_output(cmd, shell=True).decode("utf-8")
      return gres_string.split("=")[1].strip()
  return 'N/A'

def get_job_state():
  if "SLURM_JOB_ID" in os.environ:
    cmd = f"scontrol show jobid $SLURM_JOB_ID | grep JobState"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    job_state = string.split("=")[1].split()[0].strip()
    return job_state

def job_preempted():
  return "SLURM_JOB_ID"     in     os.environ and \
         "preempttime=none" not in \
         subprocess.check_output(f"scontrol show jobid $SLURM_JOB_ID", shell=True).decode("utf-8").lower()

def get_job_runtime():
  if "SLURM_JOB_ID" in os.environ:
    cmd = f"scontrol show jobid $SLURM_JOB_ID | grep RunTime"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    runtime = string.split("=")[1].split()[0].strip()
    runtime = slurm_time_to_seconds(runtime)# convert to seconds
    return runtime
  return None

def get_job_timelimit():
  if "SLURM_JOB_ID" in os.environ:
    cmd = f"scontrol show jobid $SLURM_JOB_ID | grep TimeLimit"
    string = subprocess.check_output(cmd, shell=True).decode("utf-8")
    timelimit = string.split("=")[2].split()[0].strip()
    timelimit = slurm_time_to_seconds(timelimit)# convert to seconds
    return timelimit
  return None

def save_agent_state(checkpoint_path, agent_state, args, runstate_meta, unreplicate=True):
  if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))
  if unreplicate:
    agent_state = flax.jax_utils.unreplicate(agent_state)
  with open(checkpoint_path, "wb") as f:
    f.write(
      flax.serialization.to_bytes(
        [
          agent_state,
          args,
          runstate_meta,
        ],
      )
    )
  print(f"model saved to {checkpoint_path}")

def restore_from_bytes_lax(target, encoded_bytes, lax_entries={}):
  state_dict = flax.serialization.msgpack_restore(encoded_bytes)
  target = list(target)
  for i in range(len(target)):
    key = str(i)
    if target[i] is None:
      continue
    if key in lax_entries:
      if lax_entries[key] == 'replace':
        target[i] = state_dict[key]
      elif lax_entries[key] == 'update':
        target[i].update(state_dict[key])
    else:
      target[i] = flax.serialization.from_state_dict(target[i], state_dict[key])
  return tuple(target)

def restore_agent_state(checkpoint_path, target, lax_entries=None, args_entry_idx=1):
  if any([isinstance(t, dict) and t == {} for t in target]):
    raise ValueError("There are empty dict(s) in the parsed target. This causes unexpected behavior. "
                     "If you don't have a template and are using lax_entries you can parse an empty string "
                     "instead")
  args_obj = copy.deepcopy(target[args_entry_idx]) if args_entry_idx is not None else None
  try:
    with open(checkpoint_path, "rb") as f:
        target = \
          flax.serialization.from_bytes(
            target,
            f.read(),
          )
  except ValueError as e:
    if lax_entries is not None:
      print(f"Warning: error loading checkpoint. Error given: {e}"
            f"\n"
            f"Will retry with lax_entries config: {lax_entries}")
      with open(checkpoint_path, "rb") as f:
        target = \
          restore_from_bytes_lax(
            target,
            f.read(),
            lax_entries=lax_entries,
          )
    else:
      raise e
  if args_obj is not None and isinstance(target[args_entry_idx], dict):
    args_dict = target[args_entry_idx]
    for arg, argval in args_dict.items():
      if isinstance(argval, dict) and list(argval.keys())[0] == '0':
        args_dict[arg] = list(argval.values())
    args_obj.__dict__ = args_dict
    target = target[:args_entry_idx] + (args_obj,) + target[args_entry_idx + 1:]
  return target

def load_run_args(checkpoint_path, args):
  args_obj = copy.deepcopy(args)
  with open(checkpoint_path, "rb") as f:
    _, args, _ = \
      flax.serialization.from_bytes(
        (
          None,
          args,
          None,
        ),
        f.read(),
      )
  for arg, argval in args.items():
    if isinstance(argval, dict) and list(argval.keys())[0] == '0':
      args[arg] = list(argval.values())
  args_obj.__dict__ = args
  return args_obj

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
  _post_eval_completed = False
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
      'post_eval_completed': self._post_eval_completed,
    }

  @metadata.setter
  def metadata(self, meta_dict):
    self._learner_policy_version = meta_dict['learner_policy_version']
    self._training_completed = meta_dict['training_completed']
    self._eval_completed = meta_dict['eval_completed']
    self._post_eval_completed = meta_dict.get('post_eval_completed', False)

  def save_state(self, agent_state, args):
    if not isinstance(args, dict):
      args = vars(args)
    self.save_fn(self.model_path, agent_state, args, self.metadata)
    self._save_soon = False

  def apply_signals(self, learner_policy_version, agent_state, args, save_model=True):
    args = vars(args)
    if args.get('local_rank', 0) != 0:
      return
    self._learner_policy_version = learner_policy_version
    if self._check_preempted_soon:
      if self._check_job_preempted():
        self._preempted = True
        self._save_soon = True
    if args.get('preemptible', False) and self._preempted:
      self._requeue = True
    if save_model and self._save_soon:
      self.save_state(agent_state, args)
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

  def after_post_eval(self, agent_state, args):
    args = vars(args)
    self._post_eval_completed = True
    self.save_state(agent_state, args)
    self._requeue = False

  @property
  def training_completed(self):
    return self._training_completed

  @property
  def eval_completed(self):
    return self._eval_completed

  @property
  def post_eval_completed(self):
    return self._post_eval_completed

  @property
  def completed(self):
    return self._training_completed and self._eval_completed
