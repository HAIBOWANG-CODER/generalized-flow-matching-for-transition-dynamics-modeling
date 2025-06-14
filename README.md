# generalized-flow-matching-for-transition-dynamics-modeling

![alanie dipeptide trajectory](assets/energy.png)

## Installation

Use the following code to set up the runtime environment, with the required dependencies specified in the `requirements.txt` file.

```bash
conda create --name myenv python=3.11.9
conda activate myenv
pip install -r requirements.txt
```

## Datasets

The Muller-Brown potential data is generated using the `GFM/train_data/toy_data/main_2D.py` script.
The alanine dipeptide data is generated using the `GFM_test/src/resample/md_unbiased.py` script.

## Running Experiments

All the hyperparameters corresponding to each training script mentioned in the paper are stored in YAML files located in the same directory. The script uses the `--config_path` flag to specify the YAML configuration file. 
There are four executable files, and their corresponding `--config_path` are as follows:

1. Run the metric-based flow matching code, training spline and velocity networks simultaneously.
```bash
python -m GFM.src.run.main_metic_simul.py  --config_path ./configs/simultaneous/config_metric.yaml
```
2. Run the latent space-based flow matching code, training spline and velocity networks simultaneously.
```
python -m GFM.src.run.main_latent_simul.py  --config_path ./configs/simultaneous/config_latent.yaml
```
3. Run the metric-based flow matching code, training spline first, then velocity network.
```
python -m GFM.src.run.main_metic_separ.py  --config_path ./configs/separate/config_metric.yaml
```
4. Run the latent space-based flow matching code, training spline first, then velocity network.
```
python -m GFM.src.run.main_latent_separ.py  --config_path ./configs/separate/config_latent.yaml
```

The working paths of all four modules above are `.\GFM`. And set `data_on_cluster` to the project directory, i.e., `.\GFM`.

## Evaluation

The data for Muller-Brown potential and alanine dipeptide will be evaluated during the test phase after training is completed. Set `save_address` to the folder where the data is saved. The results will be saved in the `save_address` folder.

## Citation
If you find this repository useful, we would greatly appreciate it if you could cite our paper:
```
@inproceedings{
wang2024generalized,
title={Generalized Flow Matching for Transition Dynamics Modeling},
author={Haibo Wang and Yuxuan Qiu and Yanze Wang and Rob Brekelmans and Yuanqi Du},
booktitle={NeurIPS 2024 Workshop on AI for New Drug Modalities},
year={2024},
url={https://openreview.net/forum?id=zxyP6YXknv}
}
```
