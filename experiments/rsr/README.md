## Experiment
Find the data description of the RSR dataset here: https://www.atom3d.ai/rsr.html

Download the Data from the Zenodo Repository: https://zenodo.org/record/4961085#.YfMWkmAxlqs
The data split `https://zenodo.org/record/4961085#.YfMWkmAxlqs` is required.
Place downloaded `RSR-candidates-split-by-time-indices.tar.gz` (~1.3 GB) into the `data/` directory and extract the files.

To run the base EQGAT model, execute the following:

`python train.py --nruns 3 --sdim 100 --vdim 16 --depth 5 --nruns 3 --save_dir base_eqgat`

### Tracking
The training can be tracked using Tensorboard which is all handled by Pytorch-Lightning.  
To track the RSR experiment, run:
```
bash tensorboard --logdir models/base_eqgat --port 9999
```
to see the learning curves for each run-experiment.