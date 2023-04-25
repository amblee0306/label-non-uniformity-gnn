# Introduction
This repository contains the source code for the publication in ICML 2023 titled **Leveraging Label Non-Uniformity for Node Classification in Graph Neural Networks** in PyTorch.

#### Scheme of the proposed model.
There are two separate modules (in the dashed blue boxes corresponding to the two algorithms. They can be applied either separately or jointly to the base model.

<img width="400" alt="scheme" src="https://user-images.githubusercontent.com/19768905/234184054-3cb10642-ba0b-43f8-a428-2c62f33b0be2.png">

All models are trained for node classification task.

# Setup

We used Python 3.8.13.

The environment requirements are in the requirements.txt file and can be installed as follows:

```conda create --name <envname> --file requirements.txt```

# Usage
Step 1. Generate the graph G' using the ```gen_gprime.py``` file (This serves as ${\color{blue} Algo.2}$ in the figure above). We provided the G' that we obtained for cora and citeseer in the G_prime directory as samples.

* ```python gen_gprime.py --dataset=cora --model=gcn --n-hidden=16 --self-loop --save --eta1=30 --eta2=40```

Step 2. Use the generated pickle file (representing G') and logits from the plain model (trained on G) as inputs to train a model on new training sets and previously predicted labels (This serves as ${\color{blue} Algo.1}$ in the figure above).

* ```python main_algo1.py --model=gcn --eta3=60 --self-loop --gprime-file=<name_of_file>```

# Citation

If you find this code useful, please cite the following paper:

*pending citation*

