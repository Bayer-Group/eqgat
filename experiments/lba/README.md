## Experiments

Find the data description of the LBA dataset here: https://www.atom3d.ai/lba.html

Download the Data from the Zenodo Repository: https://zenodo.org/record/4914718#.YfMTCGAxlqs
The data split `split-by-sequence-identity-30` is required.
Place downloaded `LBA-split-by-sequence-identity-30.tar.gz` (~0.56 GB) into the `data/` directory and extract the files.

It is recommended to preprocess the data files and save the graphs as `torch_geometric.data.Data` objects.  
To do the preprocessing, please execute:  
`python data.py`

To run the base EQGAT model, execute the following:

`python train.py --nruns 3 --sdim 100 --vdim 16 --depth 3 --nruns 3 --save_dir base_eqgat`

The training can be tracked using Tensorboard which is all handled by Pytorch-Lightning.  
To track the PSR experiment, run:
```
bash tensorboard --logdir models/base_eqgat --port 9999
```
to see the learning curves for each run-experiment.