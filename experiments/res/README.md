## Experiment
Find the data description of the RES dataset here: https://www.atom3d.ai/res.html

Download the Data from the Zenodo Repository: https://zenodo.org/record/5026743#.YzWatexBxqs
The data split `RES-raw.tar.gz` and `RES-split-by-cath-topology-indices.tar.gz` is required.
Place downloaded `RES-raw.tar.gz` (~31.6 GB) into the `data/` directory and extract the files.

The folder structure should look like this:
`experiments/res/data/RES/raw/data` and `experiments/res/data/RES/raw/indices`

To run the base EQGAT model, execute the following:

`python train.py --nruns 3 --sdim 100 --vdim 16 --depth 5 --nruns 1 --model_type eqgat --save_dir base_eqgat`

### Tracking
The training can be tracked using Tensorboard which is all handled by Pytorch-Lightning.  
To track the RES experiment, run:
```
bash tensorboard --logdir models/base_eqgat --port 9999
```
to see the learning curves for each run-experiment.

Notice that training the model per epoch can take quite some time. With the default setting of 32 batch-size, there are `114,273` iterations *per epoch*.
One epoch takes approximately 5 hours on an NVIDIA Tesla V100-SXM3-32GB.