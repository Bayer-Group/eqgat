## Representation Learning on Biomolecular Structures using Equivariant Graph Attention
Pytorch implementation for the manuscript *Representation Learning on Biomolecular Structures using Equivariant Graph Attention* 
presented at the [**Machine Learning For Structural Biology Workshop at NeurIPS 2022**](https://www.mlsb.io/) (short paper)
as well as in the [**Learning On Graphs Conference 2022**](https://logconference.org/) as full-length conference paper.

<img src="https://github.com/Bayer-Group/eqgat/blob/main/figures/diagram.png" width="600" height="400" class="center">


## Overview
Here we provide benchmark scripts for our experiments on the EQGAT architecture.
Make sure to install the `eqgat` library.

```
git clone https://github.com/Bayer-Group/eqgat.git
cd eqgat
```

This repository is organised as follows:

* `eqgat/` contains the implementation of the Equivariant Graph Attention Model with all required submodules. Additionally, we provide implementations of other recent 3D Graph Neural Networks.
* `experiments/` contains the 5 python training-scripts from the [ATOM3D](https://www.atom3d.ai/) and 1 synthetic datasets. To execute each training script, please refer to the corresponding README.md in the sub-directories. 

### Installation with GPU support
```
# install the conda environment
conda env create -f environment.yml 
conda activate eqgat
pip install -e .
```

### Experiments
All experiments presented in the paper can be found in the `experiments/` directory.  
Make sure to download all requested public datasets from [ATOM3D](https://www.atom3d.ai/) as described in the corresponding READMEs.


### Example 
A minimal example using the proposed SO(3) equivariant graph attention network can be found in `eqgat/README.md`

### License
Code is available under BSD 3-Clause License.

### Reference
If you make use of our model architecture, please cite our full-length manuscript:
>T. Le et al., Representation Learning on Biomolecular Structures using Equivariant Graph Attention. *Proceedings
of the First Learning on Graphs Conference (LoG 2022)*, PMLR 198, Virtual Event, December 9–12, 2022.

```
@inproceedings{
le2022representation,
title={Representation Learning on Biomolecular Structures using Equivariant Graph Attention},
author={Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
booktitle={Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=kv4xUo5Pu6}
}
```