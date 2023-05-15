import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import random
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from datetime import timedelta
from datetime import datetime

import json
from itertools import product

random.seed(42)


def prepair_attributes(X: pd.DataFrame, Y: pd.DataFrame, window_size: int):
    n_register, _ = X.shape
    X = scaled_data.to_dict(orient="list")
    Y = data.to_dict(orient="list")
    X_attr, Y_attr = [], []
    for city, ts in X.items():
        for interval_init in range(0, n_register - window_size):
            x = ts[interval_init : interval_init + window_size]
            y = Y[city][interval_init + window_size]
            X_attr.append(x)
            Y_attr.append(y)
    return np.array(X_attr), np.array(Y_attr)


def create_stuty(x_data, y_data, x_val, y_val, model, objective_function):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 500),
            "max_depth": trial.suggest_int("max_depth", -1, 100),
            "n_estimators": trial.suggest_int("n_estimators", 2, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 10000),
            "subsample_freq": trial.suggest_int("subsample_freq", 2, 15000),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart"]
            ),
        }
        regressor = model(**params)
        regressor.fit(
            x_data,
            y_data,
        )
        preds = regressor.predict(x_val)
        return objective_function(y_val, preds)

    return objective


if __name__ == "__main__":
    COLUMNS = ["dt_notific", "slug_name", "dengue_cases"]
    PERIOD_INTERVAL = {"BEGIN": "2011-05-01", "END": "2020-03-29"}
    DATA_PATH = "dados/dengue_popGT100K.gzip"
    DATA_PATH = Path(DATA_PATH)
    MDL_CONFIG_PATH = Path("./model_params")
    WINDOW_SIZE = np.linspace(2, 14, 7, dtype=int)
    INTERVAL_SIZE = np.linspace(2, 14, 7, dtype=int)
    N_TRIALS = 100

    if not MDL_CONFIG_PATH.exists():
        MDL_CONFIG_PATH.mkdir(exist_ok=True, parents=True)

    print("#################### data prepair ####################")
    data = pd.read_parquet(DATA_PATH, columns=COLUMNS)
    data = pd.pivot_table(
        data=data,
        values="dengue_cases",
        index="dt_notific",
        columns="slug_name",
        aggfunc=np.mean,
        fill_value=0,
    )
    # freq p/ 10k habitantes
    data /= 10_000

    for window_size, interval_days in product(WINDOW_SIZE, INTERVAL_SIZE):
        interval_size = timedelta(days=interval_days)
        cities = data.columns.values
        data = data.rolling(interval_size, axis="index").mean()
        random.shuffle(cities)
        data = data[cities]

        # prepair data train

        ncols, nrols = data.shape
        scaler = TimeSeriesScalerMeanVariance()
        scaler.fit(data)
        scaled_data = scaler.transform(data).reshape(ncols, nrols)
        scaled_data = pd.DataFrame(
            data=scaled_data, index=data.index, columns=data.columns
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            scaled_data, data, train_size=0.9, shuffle=False
        )

        X_train, Y_train = prepair_attributes(
            X=X_train, Y=Y_train, window_size=window_size
        )
        X_val, Y_val = prepair_attributes(X=X_val, Y=Y_val, window_size=window_size)
        objective = create_stuty(
            x_data=X_train,
            x_val=X_val,
            y_data=Y_train,
            y_val=Y_val,
            model=lgb.LGBMRegressor,
            objective_function=mean_squared_error,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)

        output_name = f"LGBM_{study.study_name}WINDOW-{window_size}_deltatime-{interval_days}.json"
        output_name = MDL_CONFIG_PATH / output_name
        with open(output_name, "w") as file:
            config = {"window_size": window_size, "time_delta": interval_days}
            config["params"] = study.best_params
            print(config)
            json.dump(fp=file, obj=config, indent=4)
