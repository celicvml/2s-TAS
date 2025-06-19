# 2s-TAS

[**论文链接 / 2s-TAS: Two-Stream Transformer for Multi-modal Human Action Segmentation**](https://your_paper_link_here)

## Introduction

This project is an open-source implementation for **Temporal Action Segmentation**, including full training code, inference scripts, and dataset. It aims to provide an efficient and reproducible research framework for temporal action segmentation.

## Dataset Download

The dataset is available at the links above.

Raw video files are needed to train our framework. Please download the datasets with RGB videos from the official websites ([Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) / [GTEA](https://cbs.ic.gatech.edu/fpv/) /[50Salads](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/)) and save them under the folder ./data/(name_dataset). 


## Environment Setup
- Python == 3.9
- PyTorch == 2.0.1
- Cuda == 11.8
  
## Extract Features
Extract features of 50salads, GTEA and Breakfast provided by [Br-Prompt](https://github.com/ttlmh/Bridge-Prompt) and [I3D](https://github.com/piergiaj/pytorch-i3d).

## Train your own model
you can retrain the model by yourself with following command.

```bash
python main.py --action=train --dataset=50salads/gtea/breakfast --split=1/2/3/4
python main.py --action=predict --dataset=50salads/gtea/breakfast --split=1/2/3/4
```
Our model adapted form [ASFormer](https://github.com/ChinaYi/ASFormer).

If you find our repo useful, please give us a star and cite:

```bash
@inproceedings{guo_2s-TAS,  
	author={Xuli Guo and Ce Li and Zhongbo Jiang and Fang Wan}, 
	booktitle={ International Conference on Multimedia and Expo Workshops(ICMEW)},   
	title={2s-TAS: Two-Stream Transformer for Multi-modal Human Action Segmentation},
	year={2025},  
}

