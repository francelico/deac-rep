import importlib
import traceback
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

API = wandb.Api(timeout=1800) #30 min timeout

def download_tfevents(run, output_dir, replace=True, join_and_pickle=True):

  if not replace and os.path.exists(os.path.join(output_dir, f'{run.name}.tfevents.pkl')):
    print(f"Skipping {run.name} as it already exists")
    return

  event_files = []
  for file in run.files():
    if file.name.startswith('events.out.tfevents'):
      event_files.append(file)
      file.download(output_dir, replace=True)
  if not event_files:
    print(f"Found {len(event_files)} event files for {run.name} expected at least 1")
    return

  if join_and_pickle:
    tfdf = pd.concat([tfevents2pandas(os.path.join(output_dir, event_file.name)) for event_file in event_files])
    save_to = os.path.join(output_dir, f'{run.name}.tfevents.pkl')
    with open(save_to, 'wb') as f:
      pickle.dump(tfdf, f)
    # delete individual event files
    for event_file in event_files:
      os.remove(os.path.join(output_dir, event_file.name))

# Extraction function
def tfevents2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def get_metric_labels(tfdf):
    return tfdf['metric'].unique()

def get_timeseries(tfdf, metric_label):
    return tfdf[tfdf['metric'] == metric_label][['step','value']]

def get_final_metric_value(tfdf, metric_label):
    max_step = tfdf[tfdf['metric'] == metric_label]['step'].max()
    tfdf = get_timeseries(tfdf, metric_label)
    return tfdf[tfdf['step'] == max_step]['value']

def add_metrics_to_run_df(run_df, metric_labels, exp_names, download_to, warning=False):
  # tfdf_dict: dict of {run_name: tfdf}
  # metric_labels: list of metric labels

  def check_step_mismatch(max_step, tfdf, metric_label):
    if max_step is None:
      max_step = tfdf[tfdf['metric'] == metric_label]['step'].max()
      return max_step, True
    else:
      return max_step, max_step == tfdf[tfdf['metric'] == metric_label]['step'].max()

  #add columns for each metric
  for metric_label in metric_labels:
    run_df[metric_label] = np.nan

  max_step = 24985600
  for event_file in os.listdir(download_to):
    if not event_file.endswith('.tfevents.pkl'):
      continue
    run_name = event_file.split('.tfevents.pkl')[0]
    if not any(exp_name in run_name for exp_name in exp_names):
      continue
    with open(os.path.join(download_to, event_file), 'rb') as f:
      tfdf = pickle.load(f)
    for metric_label in metric_labels:
      metric_values = get_final_metric_value(tfdf, metric_label)
      if metric_values.empty:
        if warning:
          print(f"Warning: metric {metric_label} missing in run {run_name}")
        continue
      max_step, step_match = check_step_mismatch(max_step, tfdf, metric_label)
      if not step_match:
        assert tfdf[tfdf['metric'] == metric_label]['step'].max() >= 25000000, f"Step mismatch for {metric_label} in {run_name}"
      run_df.loc[run_df.name == run_name, metric_label] = metric_values.values[0]
  return run_df

def get_runs(plot_config):
  runs = API.runs(plot_config.entity + "/" + plot_config.project)
  keep_runs = []
  for run in runs:
    if run.state not in plot_config.run_state:
      continue
    exp_name = run.config.get('exp_name', None)
    if exp_name not in plot_config.exp_names:
      continue
    keep_runs.append(run)
  return keep_runs

def make_run_df(runs, plot_config, save_to=None):
  summary_list, config_list, name_list, label_list, exp_name_list, env_id_list = [], [], [], [], [], []
  for run in runs:
    # .summary contains output keys/values for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
    # .name is the human-readable name of the run.
    name_list.append(run.name)
    label_list.append(plot_config.exp_names[run.config['exp_name']])
    exp_name_list.append(run.config['exp_name'])
    env_id_list.append(run.config['env_id'])

  run_df = pd.DataFrame(
    {"label": label_list,
     "summary": summary_list,
     "config": config_list,
     "name": name_list,
     "exp_name": exp_name_list,
     "env_id": env_id_list,}
  )

  # avoids querying API multiple times
  if save_to:
    with open(save_to, 'wb') as f:
      pickle.dump(run_df, f)

  return run_df

def find_script_in_wandb_local_dir(wandb_dir, run_name):
  runlist = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d)) and run_name in d]
  if not runlist:
    print(f"No runs found matching {run_name} in {wandb_dir}")
    return None
  assert len(runlist) == 1, f"Found {len(runlist)} runs with the name {run_name} in the wandb directory"
  run_wandb_dir = os.path.join(wandb_dir, runlist[0])
  # list all python files in the run directory, recursively
  result = list(Path(run_wandb_dir).rglob("*.py"))
  if not result:
    print(f"No python files found in the run directory {run_wandb_dir}")
    return None
  assert len(result) == 1, f"Found {len(result)} python files in the run directory {run_wandb_dir}"
  script_path = str(result[0])
  return script_path

def import_script_from_wandb(run_name, args, check_local_dir_for_file=False):

  API = wandb.Api(timeout=1800)  # 30 min timeout
  run = API.run(args.wandb_entity + "/" + args.wandb_project_name + "/" + run_name)

  download_dir = f"{args.run_dir}/{run_name}"
  script_path = None
  for file in run.files():
    if file.name.endswith('.py'):
      file.download(download_dir, replace=True)
      script_path = os.path.join(download_dir, file.name)
      break
  if script_path is None and check_local_dir_for_file:
    print(f"No file found in wandb online. Checking local directory for file. Consider running wandb sync.")
    script_path = find_script_in_wandb_local_dir(args.wandb_dir, run_name)
  if script_path is None:
    raise ValueError(f"No python file found for the wandb run {run_name}")
  spec = importlib.util.spec_from_file_location(
    name="script_module",  # note that ".test" is not a valid module name
    location=script_path,
  )
  script_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(script_module)
  return script_module

# For testing only
if __name__ == "__main__":
    DATA_DIR = './data'
    event_file_name = 'DodgeballEasy-v0__runstate_deac_mico_procgen__8__86e96cac-9bd7-44aa-9bac-d83e282bef1e.tfevents'
    events_file_path = os.path.join(DATA_DIR, event_file_name)
    tfdf = tfevents2pandas(events_file_path)
    print(get_metric_labels(tfdf))
    print(get_timeseries(tfdf, 'losses/mico_loss'))
    print(get_final_metric_value(tfdf, 'losses/mico_loss'))
    print("Done")
