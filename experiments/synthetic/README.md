## Synthetic Dataset

Here we provide the code to generate the synthetic `triangle` dataset.

To generate the dataset, please execute:

```bash
python data.py --num_samples 50_000 --num_points 100 --max_r 10.0 --min_dist 2.0
```

will generate a dataset of 50,000 "structures", where each structure consist of 100 random points in the ball of radius 10.0.


To train a network, execute the following:

```bash
python train.py --save_dir base_eqgat --model_type eqgat --depth 3 --nruns 3
```