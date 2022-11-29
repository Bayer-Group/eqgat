## Experiment
Find the data description of the PPI dataset here: https://www.atom3d.ai/pip.html

Download the Data from the Zenodo Repository: https://zenodo.org/record/4911102#.YzWcHexBxqs
The data split `PPI-DIPS-split.tar.gz` is required.
Place downloaded `PPI-DIPS-split.tar.gz` (~13.6 GB) into the `data/` directory and extract the files.

The folder structure should look like this:
`/experiments/ppi/data/DIPS-split`


To run the base EQGAT model, execute the following:

`python train.py --nruns 3 --sdim 100 --vdim 16 --depth 5 --nruns 1 --model_type eqgat --save_dir base_eqgat`


### Tracking
The training can be tracked using Tensorboard which is all handled by Pytorch-Lightning.  
To track the PPI experiment, run:
```
bash tensorboard --logdir models/base_eqgat --port 9999
```
to see the learning curves for each run-experiment.


Notice that training the model per epoch can take quite some time. With the default setting of 10 batch-size, there are `149,914` iterations *per epoch*.
One epoch takes approximately 10 hours on an NVIDIA Tesla V100-SXM3-32GB.