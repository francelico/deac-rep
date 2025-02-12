# Decoupled Actor-Critic Representations
Official code repository for [Studying the Interplay Between the Actor and Critic Representations in Reinforcement Learning (ICLR 2025)](https://openreview.net/forum?id=tErHYBGlWc).

This repository provides code to reproduce the experiments conducted in the paper in the Procgen and PixelBrax benchmarks. We split the codebase into the `procgen_experiments` and `pixelbrax_experiments` directories, as each environment requires a different set of dependencies. You can refer to the README files in each directory for instructions on how to set up and run individual experiments.

In each benchmark we provide JAX implementations for the following algorithms:
- [Proximal Policy Optimisation (PPO)](https://arxiv.org/abs/1707.06347)
- [Phasic Policy Gradient (PPG)](https://arxiv.org/abs/2009.04416)
- [Delayed Critic Policy Gradient (DCPG)](https://arxiv.org/abs/2210.09960)
We follow the [CleanRL](https://github.com/vwxyzjn/cleanrl) structure and implement each algorithm/environment pair in a separate file. Additionally, we provide `_shared` and `_sep` versions of each algorithm, referring to shared and decoupled actor-critic architectures studied in the paper.

Finally, each algorithm supports the following representation learning methods, which may be applied to the actor, critic, or both:
- Value Distillation, the standard representation learning approach used in PPO, PPG and DCPG.
- Dynamics Prediction, as proposed in the [DCPG work](https://arxiv.org/abs/2210.09960)
- [Data Regularized Actor-Critic (DrAC)](https://arxiv.org/abs/2006.12862), a data augmentation approach.
- Advantage Targets Distillation, from the [Decoupled Advantage Actor-Critic (DAAC) work](https://arxiv.org/abs/2102.10330)
- [Matching under Independent Couplings (MICo)](https://arxiv.org/abs/2106.08229), an objective explicitly shaping the latent feature space to embed differences in state values.

We provide utilities to measure the information agent trajectories and learned latents carry about the environment (refer to the paper and [mutual_info_procgen_eval.py](procgen_experiments/eval_utils/mutual_info_procgen_eval.py) for a description of each metric and reference implementation). We also re-implement in JAX representation learning metrics first proposed in [Investigating the Properties of Neural Network Representations in Reinforcement Learning](https://arxiv.org/abs/2203.15955), see [repmetric_util.py](procgen_experiments/utils/repmetric_util.py).

Our experiment logs and data are publicly available in our [wandb project](TODO). The trained model checkpoints are accessible from our [model repository](TODO). When downloading experiment data, we strongly recommend relying on the `.tfevents` files stored in the wandb repository for each run. We found `.tfevents` files less likely to drop data than the data streamed through the wandb API and displayed in the web interface. The `.tfevents` files can be downloaded, merged and converted to a pandas dataframe using utilities provided in [log_util.py](procgen_experiments/utils/log_util.py).

## Installation instructions
Please refer to the README files in the `procgen_experiments` and `pixelbrax_experiments` directories for instructions on how to set up the dependencies and run experiments.

## Citation
If you find this repository useful in your research, please consider citing our paper:
```
@inproceedings{
garcin2025studying,
title={Studying the Interplay Between the Actor and Critic Representations in Reinforcement Learning},
author={Samuel Garcin and Trevor McInroe and Pablo Samuel Castro and Christopher G. Lucas and David Abel and Prakash Panangaden and Stefano V Albrecht},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=tErHYBGlWc}
}
```

## TODO
- [ ] Add wandb project link
- [ ] Add model repository link
- [ ] Add pixelbrax experiments