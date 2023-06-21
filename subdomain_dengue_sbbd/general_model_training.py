import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import random
from utils import *
from pathlib import Path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from datetime import datetime
from threading import Thread
import general_model_tuning as featuring

random.seed(42)


def prepair_attributes(X: pd.DataFrame, Y: pd.DataFrame, size: int):
    n_register = len(X)
    X = X.values
    Y = Y.values
    X_attr, Y_attr = [], []
    for interval_init in range(0, n_register - size):
        x = X[interval_init : interval_init + size]
        y = Y[interval_init + size]
        X_attr.append(x)
        Y_attr.append(y)
    return np.array(X_attr), np.array(Y_attr)


def load_models_params(model_config):
    model_config_dir = Path(model_config)
    model_config_dir = [
        folder for folder in model_config_dir.iterdir() if folder.is_dir()
    ]
    for folder in model_config_dir:
        for file in folder.glob("*.json"):
            yield folder.name, file


if __name__ == "__main__":
    # -- data load and transform --
    data = read_data_file(data_path=DATA_PATH, columns=COLUMNS)
    nrows, ncols = data.shape
    cities = data.columns.values
    random.shuffle(cities)
    general_data = data[cities]
    # -- data scaling --
    series_scaler = TimeSeriesScalerMeanVariance()
    series_scaler.fit(general_data)
    scaled = series_scaler.transform(general_data)
    scaled = scaled.reshape(nrows, ncols)
    scaled = pd.DataFrame(
        data=scaled, columns=general_data.columns, index=general_data.index
    )

    # -- train test split --
    xtrain, xtest, ytrain, ytest = train_test_split(
        scaled, general_data, shuffle=False, train_size=0.8
    )

    for name, model_file in load_models_params("model_params"):
        result = {
        "cidades": [],
        "n_cluster": [],
        "cluster_idx": [],
        "model_name": [],
        "window_size": [],
        "rmse": [],
        "time": [],
    }

        with open(file=model_file, mode="r") as conf_file:
            model_config = json.load(fp=conf_file)
        window_size = model_config["window_size"]
        print(f"[{name}-{window_size}] prepairing attributes")
        x, y = featuring.prepair_attributes(X=xtrain, Y=ytrain, size=window_size)

        print(f"[{name}-{window_size}] training model")
        match name:
            case "RandomForestRegressor":
                regressor = RandomForestRegressor(**model_config["params"])
            case "LinearSVR":
                regressor = LinearSVR(**model_config["params"])
            case "LGBMRegressor":
                regressor = lgb.LGBMRegressor(**model_config["params"])
        print(f"[TRAINING] - {name} - {window_size}")
        t_init = datetime.now()
        regressor.fit(X=x, y=y)
        t_end = datetime.now()
        train_time = (t_end - t_init).seconds
        for city in cities:
            print(f"[TRAINING] error estimating - {name} - {city}, {window_size}")
            xcity, ycity = xtest[city], ytest[city]
            xcity, ycity = prepair_attributes(X=xcity, Y=ycity, size=window_size)
            y_hat = regressor.predict(X=xcity)
            rmse = mean_squared_error(y_true=ycity, y_pred=y_hat)
            result["window_size"].append(window_size)
            result["time"].append(train_time)
            result["model_name"].append(regressor.__class__.__name__)
            result["cidades"].append(city)
            result["n_cluster"].append(1)
            result["cluster_idx"].append(0)
            result["rmse"].append(rmse)
        result_folder = Path(f"results/general_model/{name}")
        result_file = (
            f"regressor_{regressor.__class__.__name__}_window_size{window_size}.csv"
        )
        result_file = result_folder / result_file
        if not result_folder.exists():
            result_folder.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame(result)
        result.to_csv(result_file, index=False)
