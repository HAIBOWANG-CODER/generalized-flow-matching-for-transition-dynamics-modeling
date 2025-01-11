# generalized-flow-matching-for-transition-dynamics-modeling

## Installation

Use the following code to set up the runtime environment, with the required dependencies specified in the `requirements.txt` file.

```bash
conda create --name myenv python=3.11.9
conda activate myenv
pip install -r requirements.txt
```

## Datasets

The Muller-Brown potential data is generated using the `GFM/toy_data/main_2D.py` script.
The alanine dipeptide data is generated using the `GFM_test\src\resample\md_unbiased.py` script.

## Running Experiments

All the hyperparameters corresponding to each training script mentioned in the paper are stored in YAML files located in the same directory. The script uses the `--config_path` flag to specify the YAML configuration file. For example, to run the file `.\GFM\src\train\simultaneous\main_metic_simul.py`, use the following command:

```bash
--config_path .\GFM\src\train\simultaneous\config_metric.yaml
```

Set `data_on_cluster` to the project directory, i.e., `.\GFM`.

## Evaluation

The data for Muller-Brown potential and alanine dipeptide will be evaluated during the test phase after training is completed. The results will be saved in the `save_address` folder.
