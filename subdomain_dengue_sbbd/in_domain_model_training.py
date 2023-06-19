import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from pathlib import Path
import random
from utils import *
import general_model_tuning as train_prepair
import general_model_training as test_prepair
import json
from general_model_training import load_models_params
from datetime import datetime


def shuffle_city(cities):
    names = cities.copy()
    random.shuffle(names)
    return names


def read_clusters(cluster_path):
    clusters_root = Path(cluster_path)
    for cluster_file in clusters_root.glob("*.json"):
        with open(cluster_file, "r") as cls_file:
            cluster = json.load(fp=cls_file)
            yield cluster


def split_cluster(cluster_path, X, Y):
    # load clusters
    X = data.copy()
    for cls_conf in read_clusters(cluster_path=cluster_path):
        k = len(cls_conf)
        # in_domain train
        for cluster, names in cls_conf.items():
            shuffled = names
            shuffled = shuffle_city(cities=shuffled)
            yield k, cluster, X[shuffled], Y[shuffled]


def model_train(param_folder, X, Y):
    param_folder = Path(param_folder)
    for name, config_json in load_models_params(param_folder):
        with open(config_json, "r") as param_set:
            params = json.load(param_set)
        w_size = params["window_size"]
        x, y = train_prepair.prepair_attributes(X=X, Y=Y, size=w_size)
        params = params["params"]
        print(f"[TRAINING] model {name} - {w_size}")

        match name:
            case "RandomForestRegressor":
                regressor = RandomForestRegressor(**params)
            case "LinearSVR":
                regressor = LinearSVR(**params)
            case "LGBMRegressor":
                regressor = lgb.LGBMRegressor(**params)
        t_init = datetime.now()
        regressor.fit(X=x, y=y)
        t_end = datetime.now()
        train_time = (t_end - t_init).seconds
        yield name, w_size, train_time, regressor


def in_domain_eval(xtest, ytest, train_process, cidades, k, cluster_idx):
    result = {
        "cidades": [],
        "n_cluster": [],
        "cluster_idx": [],
        "model_name": [],
        "window_size": [],
        "rmse": [],
        "time": [],
    }
    model_name, window_size, train_time, regressor = train_process
    for cidade in cidades:
        x, y = test_prepair.prepair_attributes(
            xtest[cidade], ytest[cidade], size=window_size
        )
        y_hat = regressor.predict(x)
        error = mean_squared_error(y_true=y, y_pred=y_hat, squared=False)
        print(f"[EVALUATING] model {regressor.__class__.__name__} - {cidade} - {error}")
        result["cidades"].append(cidade)
        result["n_cluster"].append(k)
        result["cluster_idx"].append(cluster_idx)
        result["model_name"].append(model_name)
        result["window_size"].append(window_size)
        result["rmse"].append(error)
        result["time"].append(train_time)
        out_name = f"in-domain_{model_name}_K_{k}_cluster_{cluster_idx}.csv"
        yield out_name, result


if __name__ == "__main__":
    # -- data transform --
    data = read_data_file(data_path=DATA_PATH, columns=COLUMNS)
    nrows, ncols = data.shape
    scaler = TimeSeriesScalerMeanVariance()
    scaler.fit(data)
    scaled = scaler.transform(data)
    scaled = scaled.reshape(nrows, ncols)
    scaled = pd.DataFrame(data=scaled, columns=data.columns, index=data.index)

    for k, cluster, cluster_x, cluster_y in split_cluster(
        cluster_path="clustering", X=scaled, Y=data
    ):
        xtrain, xtest, ytrain, ytest = train_test_split(
            cluster_x, cluster_y, train_size=0.8, shuffle=False, random_state=42
        )
        cities = cluster_x.columns.values
        train_process = model_train(param_folder="model_params", X=xtrain, Y=ytrain)
        for model_name, window_size, train_time, regressor in train_process:
            for out_name, result in in_domain_eval(
                xtest=xtest,
                ytest=ytest,
                train_process=[model_name, window_size, train_time, regressor],
                cidades=cities,
                k=k,
                cluster_idx=cluster
            ):
                # -- write result file
                result_rf = pd.DataFrame(result)
                result_rf.to_csv(
                    f"results/in-domain/{model_name}/{out_name}", index=False
                )
