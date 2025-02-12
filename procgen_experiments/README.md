# Procgen Experiments

## Installation

```bash
conda create -n deac_procgen python=3.8
conda activate deac_procgen
conda install nvidia/label/cuda-12.4.1::cuda
conda install conda-forge::pipx
pipx ensurepath
source ~/.bashrc
conda activate deac_procgen
pipx install poetry
```

Install dependencies
```bash
cd deac-rep/procgen_experiments
poetry install
```

Install CUDA compiled JAX to enable running experiments on the GPU. Our experiments were run on a machine with CUDA 12.0. Adjust accordingly depending on your CUDA version and GPU driver.
```bash
poetry run pip install --upgrade "jax[cuda12_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run pip install nvidia-cudnn-cu12==8.9.7.29
```

You can valide your installation by running a short experiment (this should complete in a few minutes):
```bash
python train_scripts/runstate_ppo_shared_mico_c_procgen.py --local_num_envs 2 --num_minibatches 2 --num_eval_episodes 10 --total_timesteps 1024 --num_steps 64 --save_model --mi_eval_total_timesteps 256 --mi_eval_downsample_to_n 128 
```