# Introduction
This repository contains the source code for the publication in ICML 2023 titled **Leveraging Label Non-Uniformity for Node Classification in Graph Neural Networks** in PyTorch.

#### Scheme of the proposed model.
There are two separate modules (in the dashed blue boxes corresponding to the two algorithms. They can be applied either separately or jointly to the base model.

<img width="400" alt="scheme" src="https://user-images.githubusercontent.com/19768905/234184054-3cb10642-ba0b-43f8-a428-2c62f33b0be2.png">

All models are trained for node classification task. 
The following datasets are included:
* Cora
* Citeseer
* Pubmed
* Texas
* Chameleon
* Wisconsin
* CS
* Photo

The following base models are includes:
* MLP
* GCN
* GAT

# Setup

First, unzip the data.zip file for the datasets required during training.

We used Python 3.8.13. 

The environment requirements are in the requirements.txt file and can be installed as follows:

```conda create --name <envname> --file requirements.txt```

Alternatively,

```pip install -r pip-requirements.txt```

# Usage

We provide examples of training commands used to train WGNN for node classification.

* Cora dataset (Test acc: 83.19 +/- 0.53) 

```python wgnn.py --dataset=cora --model=gcn --n-hidden=16 --self-loop --early-stop --eta1=30 --eta2=40```

To search for the best eta1 and eta2 hyperparameters for ${\color{blue} \text{Algo. 2}}$, we run all combinations using the ```--all-combination``` flag.

```python wgnn.py --dataset=cora --model=gcn --n-hidden=16 --self-loop --early-stop --save --all-combination```

# Citation

If you find this code useful, please cite the following paper:

F. Ji, S. H. Lee, H. Meng, K. Zhao, J. Yang, and W. P. Tay, “Leveraging label non-uniformity for node classification in graph neural networks,” in Proc. International Conference on Machine Learning, Hawaii, USA, Jul. 2023.

