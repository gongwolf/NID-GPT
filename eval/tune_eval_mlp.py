from pathlib import Path
import optuna
import argparse
import json 

from eval.models import MLP_Mult
from scripts.evaluation import load_data, train_mlp

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, help='dataset name used to create a json file to store configuration of the MLP model under ~/tuned_models')
parser.add_argument('--device', type=str, help="device used to train the model, cuda, cuda:1, or cpu")
parser.add_argument('--train', type=str, required=True, help='Path to the training CSV file')
parser.add_argument('--test', type=str, required=True, help='Path to the testing CSV file')
args = parser.parse_args()

X_train, X_test, y_train, y_test, label_mapping = load_data(args.train, args.test)

print("training X shape:", X_train.shape)
print("training Y shape:", y_train.shape)
print("test X shape:", X_test.shape)    
print("test Y shape:", y_test.shape)

def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f'suggest_{distribution}')(label, *args)

def _suggest_optional(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    if trial.suggest_categorical(f"optional_{label}", [True, False]):
        return _suggest(trial, distribution, label, *args)
    else:
        return 0.0
    
def _suggest_mlp_layers(trial: optuna.trial.Trial, mlp_d_layers: list[int]):

    min_n_layers, max_n_layers = mlp_d_layers[0], mlp_d_layers[1]
    d_min, d_max = mlp_d_layers[2], mlp_d_layers[3]

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t


    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last

    return d_layers

def suggest_mlp_params(trial):
    params = {}
    params["lr"] = trial.suggest_loguniform("lr", 5e-5, 0.005)
    params["dropout"] = _suggest_optional(trial, "uniform", "dropout", 0.0, 0.5)
    params["weight_decay"] = _suggest_optional(trial, "loguniform", "weight_decay", 1e-6, 1e-2)
    params["d_layers"] = _suggest_mlp_layers(trial, [1, 8, 6, 10]) #[min_n_layers, max_n_layers, d_min, d_max],
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Choose from specified list
    return params

def objective(trial):
    params = suggest_mlp_params(trial)
    # print(params)
    params["input_shape"] = X_train.shape[1]
    params["num_classes"] = len(y_train.iloc[:, -1].unique())

    model = MLP_Mult(
        input_shape=X_train.shape[1],    # Replace with the actual input shape of your data
        d_layers=params["d_layers"],     # Layer configuration from Optuna
        num_classes=len(y_train.iloc[:, -1].unique()),                  # Adjust as per your dataset
        dropout_rate=params["dropout"]
    ).to(args.device)

    trial.set_user_attr("params", params)
    
    f1, _ = train_mlp(
        model=model,
        X_train=X_train,           # Replace with actual training data
        y_train=y_train,           # Replace with actual training labels
        X_test=X_test,             # Replace with actual test data
        y_test=y_test,             # Replace with actual test labels
        device=args.device,
        epochs=50,                # Define epochs or suggest it if you want to optimize
        batch_size=params["batch_size"],
        lr=params["lr"]
    )
    
    return f1


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

bets_params = study.best_trial.user_attrs['params']

best_params_path = f"tuned_models/mlp_{args.ds_name}.json"
Path(best_params_path).write_text(json.dumps(bets_params, indent=4) + '\n')
