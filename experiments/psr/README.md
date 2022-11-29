## Experiment
Find the data description of the PSR dataset here: https://www.atom3d.ai/psr.html

Download the Data from the Zenodo Repository: https://zenodo.org/record/4915648#.YfMVuWAxlqs
The data split `PSR-split-by-year` is required.
Place downloaded `PSR-split-by-year.tar.gz` (~2.0 GB) into the `data/` directory and extract the files.

To run the base EQGAT model, execute the following:

`python train.py --nruns 3 --sdim 100 --vdim 16 --depth 5 --nruns 3 --model_type eqgat --save_dir base_eqgat`

### Tracking
The training can be tracked using Tensorboard which is all handled by Pytorch-Lightning.  
To track the PSR experiment, run:
```
bash tensorboard --logdir models/base_eqgat --port 9999
```
to see the learning curves for each run-experiment.