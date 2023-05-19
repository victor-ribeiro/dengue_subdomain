import importlib
import pandas as pd
from pathlib import Path
import numpy as np
import json

COLUMNS = ["dt_notific", "slug_name", "dengue_cases"]
DATA_PATH = "dados/dengue_popGT100K.gzip"
DATA_PATH = Path(DATA_PATH)


def create_objective(x_data, y_data, x_val, y_val, model, objective_function, params):
    def objective(trial):
        params_ = {item: eval(params[item], {"trial": trial}) for item in params}
        regressor = model(**params_)
        regressor.fit(
            x_data,
            y_data,
        )
        preds = regressor.predict(x_val)
        return objective_function(y_val, preds)

    return objective


def read_data_file(data_path, columns, freq=None):
    data = pd.read_parquet(data_path, columns=columns)
    data = pd.pivot_table(
        data=data,
        values="dengue_cases",
        index="dt_notific",
        columns="slug_name",
        aggfunc=np.sum,
        fill_value=0,
    )
    # freq p/ 10k habitantes
    if freq:
        data /= 10_000
    return data


def load_learner(pckg, model, params, func_pkg, func):
    # -- prepair learner --
    pckg = importlib.import_module(**pckg)
    learner = eval(f"pckg.{model}")
    # -- objective function --
    func_pkg = importlib.import_module(**func_pkg)
    objective_func = eval(f"func_pkg.{func}")
    return learner, params, objective_func, model


def model_generator(file_path):
    with open(file_path, "r") as file:
        config_models = json.load(fp=file)
    for config in config_models:
        yield load_learner(**config)
