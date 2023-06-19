import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import random
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import json
from utils import *
from itertools import product

random.seed(42)


def prepair_attributes(X: pd.DataFrame, Y: pd.DataFrame, size: int):
    n_register = len(X)
    X = X.to_dict(orient="list")
    Y = Y.to_dict(orient="list")
    X_attr, Y_attr = [], []
    for city, ts in X.items():
        for interval_init in range(0, n_register - size):
            x = ts[interval_init : interval_init + size]
            y = Y[city][interval_init + size]
            X_attr.append(x)
            Y_attr.append(y)
    return np.array(X_attr), np.array(Y_attr)


if __name__ == "__main__":
    MDL_CONFIG_PATH = Path("./model_params")
    LNR_CONFIG_PATH = MDL_CONFIG_PATH / "model_config.json"
    window_size = 6
    N_TRIALS = 100

    if not MDL_CONFIG_PATH.exists():
        MDL_CONFIG_PATH.mkdir(exist_ok=True, parents=True)
    # -- data prepair --

    data = read_data_file(data_path=DATA_PATH, columns=COLUMNS)
    cities = data.columns.values
    # --     end      --

    models_gen = model_generator(file_path=LNR_CONFIG_PATH)
    for model, model_params, objective_func, model_name in models_gen:
        random.shuffle(cities)
        data = data[cities]
        nrols, ncols = data.shape

        scaler = TimeSeriesScalerMeanVariance()
        scaler.fit(data)
        scaled_data = scaler.transform(data).reshape(nrols, ncols)
        scaled_data = pd.DataFrame(
            data=scaled_data, index=data.index, columns=data.columns
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            scaled_data, data, train_size=0.9, shuffle=False
        )
        # window prep
        X_train, Y_train = prepair_attributes(X=X_train, Y=Y_train, size=window_size)
        X_val, Y_val = prepair_attributes(X=X_val, Y=Y_val, size=window_size)

        objective = create_objective(
            x_data=X_train,
            x_val=X_val,
            y_data=Y_train,
            y_val=Y_val,
            model=model,
            objective_function=objective_func,
            params=model_params,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)

        output_name = f"{model_name}_{study.study_name}WINDOW-{window_size}.json"
        model_folder = MDL_CONFIG_PATH / model_name
        output_name = model_folder / output_name
        if not model_folder.exists():
            model_folder.mkdir(exist_ok=True, parents=True)
        with open(output_name, "w") as file:
            config = {"window_size": window_size}
            config["params"] = study.best_params
            json.dump(fp=file, obj=config, indent=4)
