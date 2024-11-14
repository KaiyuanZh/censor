# CENSOR: Defense Against Gradient Inversion via Orthogonal Subspace Bayesian Sampling
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 2.2](https://img.shields.io/badge/pytorch-2.2-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-MIT-DodgerBlue.svg?style=plastic)

Table of Contents
=================
- [Overview](#Overview)
- [Paper](https://kaiyuanzhang.com/publications/NDSS25_Censor.pdf)
- [Install required packages](#Install-required-packages)
- [Download models](#Download-models)
- [How to Run the Code](#How-to-Run-the-Code)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)
- [Special thanks to...](#Special-thanks-to)

## Overview
- This is the PyTorch implementation for NDSS 2025 paper "[CENSOR: Defense Against Gradient Inversion via Orthogonal Subspace Bayesian Sampling](https://kaiyuanzhang.com/publications/NDSS25_Censor.pdf)".  
- **Take-Away**: CENSOR perturbs gradients within a *subspace* **orthogonal** to the original gradient.

![overview](./figures/overview.png)

## Results
![results](./figures/qualitative.png)

## Install required packages
```bash
# Create python environment (optional)
conda env create -f env.yml
conda activate censor 
```  

## Download models
  - download the `shape_predictor_68_face_landmarks.dat` from [here](https://drive.google.com/drive/folders/1B2I-1cXbvoYlMK-dSUsXwSuNo9LhcV1V?usp=sharing) to the root `censor` directory.
  - download the `stylegan2-ffhq-config-f.pt` from [here](https://drive.google.com/drive/folders/1B2I-1cXbvoYlMK-dSUsXwSuNo9LhcV1V?usp=sharing) to the `/inversefed/genmodels/stylegan2_io/` directory.


## How to Run the Code
```bash
python run_rec.py --config $CONFIG_PATH
```

Example command for evaluating CENSOR with BigGAN:
```bash
python run_rec.py --config configs_biggan.yml
```

## Citation
Please cite our work as follows for any purpose of usage.
```bibtex
@inproceedings{zhang2025censor,
  title={CENSOR: Defense Against Gradient Inversion via Orthogonal Subspace Bayesian Sampling},
  author={Zhang, Kaiyuan and Cheng, Siyuan and Shen, Guangyu and Ribeiro, Bruno and An, Shengwei and Chen, Pin-Yu and Zhang, Xiangyu and Li, Ninghui},
  booktitle={32nd Annual Network and Distributed System Security Symposium, {NDSS} 2025},
  year = {2025},
}
```

## Acknowledgement
Part of the code is adapted from the following repos. We express great gratitude for their contribution to our community!
- [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients)  
- [ILO](https://github.com/giannisdaras/ilo)  
- [GGL](https://github.com/zhuohangli/GGL)  
- [GIFD_Gradient_Inversion_Attack](https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack)

The BigGAN implementation, we use PyTorch official [implementation and weights](https://github.com/rosinality/stylegan2-pytorch). For StyleGAN2, we adapt this [Pytorch implementation](https://github.com/rosinality/stylegan2-pytorch), which is based on the [official Tensorflow code](https://github.com/NVlabs/stylegan2).

## Special thanks to...
[![Stargazers repo roster for @KaiyuanZh/censor](https://reporoster.com/stars/KaiyuanZh/censor)](https://github.com/KaiyuanZh/censor/stargazers)
[![Forkers repo roster for @KaiyuanZh/censor](https://reporoster.com/forks/KaiyuanZh/censor)](https://github.com/KaiyuanZh/censor/network/members)