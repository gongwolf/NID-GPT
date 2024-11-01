import subprocess
import sys
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path
import numpy as np  
import json 
import torch 
import pandas as pd 

import lib
from eval.models import MLP_Mult
from scripts.evaluation import load_data, train_mlp, encode_labels

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)
parser.add_argument('--test', type=str, required=True, help='Path to the testing CSV file')


args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)

pipeline = f'scripts/pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'    
parent_path = Path(f'NID-GPT/exp/{ds_name}/')
exps_path = Path(f'NID-GPT/exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdir
eval_seeds = f'NID-GPT/scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)



def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    test_df = pd.read_csv(args.test)
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0    
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    steps = trial.suggest_categorical('steps', [5000, 10000, 20000, 30000, 50000, 80000])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    os.makedirs(str(exps_path / f"{trial.number}"), exist_ok=True)
 
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)
    lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--sample'], check=True)
    
    x_feature_train = np.load(exps_path / f"{trial.number}/X_num_train.npy")
    y_label_train = np.load(exps_path / f"{trial.number}/y_train.npy")
    train_data = np.hstack((x_feature_train, y_label_train.reshape(-1, 1)))

    num_features = x_feature_train.shape[1]  # Number of features in x_feature_train
    feature_columns = [f"feature_{i}" for i in range(num_features)]  # Naming the feature columns
    label_column = ["label"]  # Naming the label column

    # Combine columns into a single list for DataFrame creation
    columns = feature_columns + label_column

    # Create the DataFrame
    train_df = pd.DataFrame(train_data, columns=columns)

    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]    
    if train_df.iloc[:, -1].dtype=="object":
        y_train, y_test, _= encode_labels(y_train, y_test)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(f"tuned_models/mlp_{args.ds_name}.json", "r") as f:
        params = json.load(f)

    model = MLP_Mult(
        input_shape=params["input_shape"],
        d_layers=params["d_layers"],     # Layer configuration
        num_classes=params["num_classes"],                  # Adjust if necessary for your dataset
        dropout_rate=params["dropout"]
    ).to(device)

    trial.set_user_attr("params", params)
    
    f1, _ = train_mlp(
        model=model,
        X_train=X_train,           # Replace with actual training data
        y_train=y_train,           # Replace with actual training labels
        X_test=X_test,             # Replace with actual test data
        y_test=y_test,             # Replace with actual test labels
        device=device,
        epochs=100,                # Define epochs or suggest it if you want to optimize
        batch_size=params["batch_size"],
        lr=params["lr"]
    )
    
    shutil.rmtree(exps_path / f"{trial.number}")
    return f1

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True)
