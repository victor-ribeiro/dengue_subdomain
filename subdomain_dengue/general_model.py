# %%
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import random
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from datetime import timedelta

random.seed(42)
# %%


def prepair_attributes(X: pd.DataFrame, Y: pd.DataFrame, window_size: int):
    n_register, _ = X.shape
    X = scaled_data.to_dict(orient="list")
    Y = data.to_dict(orient="list")
    X_attr, Y_attr = [], []
    for city, ts in X.items():
        for interval_init in range(0, n_register - window_size):
            x = ts[0 : interval_init + window_size]
            y = Y[city][interval_init + window_size]
            X_attr.append(x)
            Y_attr.append(y)
    return np.array(X_attr), np.array(Y_attr)


# %%
COLUMNS = ["dt_notific", "slug_name", "dengue_cases"]
PERIOD_INTERVAL = {"BEGIN": "2011-05-01", "END": "2020-03-29"}

DATA_PATH = "dados/dengue_popGT100K.gzip"
DATA_PATH = Path(DATA_PATH)
WINDOW_SIZE = 4
# data prepair
data = pd.read_parquet(DATA_PATH, columns=COLUMNS)
data = pd.pivot_table(
    data=data,
    values="dengue_cases",
    index="dt_notific",
    columns="slug_name",
    aggfunc=np.sum,
    fill_value=0,
)

interval_size = timedelta(days=4)
cities = data.columns.values
data = data.rolling(interval_size, axis="index").max()
random.shuffle(cities)
data = data[cities]
# %%
# prepair data train

ncols, nrols = data.shape
scaler = TimeSeriesScalerMeanVariance()
scaler.fit(data)
scaled_data = scaler.transform(data).reshape(ncols, nrols)
scaled_data = pd.DataFrame(data=scaled_data, index=data.index, columns=data.columns)

X_train, X_test, Y_train, Y_test = train_test_split(
    scaled_data, data, train_size=0.8, shuffle=False
)
X_train, Y_train, X_val, Y_val = train_test_split(
    X_train, Y_train, shuffle=False, random_state=42, train_size=0.6
)


X_train, Y_train = prepair_attributes(X=X_train, Y=Y_train, window_size=WINDOW_SIZE)

print(len(X_train))

X_val, Y_val = prepair_attributes(X=X_val, Y=Y_val, window_size=WINDOW_SIZE)

# %%

from lightgbm import early_stopping

dtrain = lgb.Dataset(X_train, label=Y_train)
dval = lgb.Dataset(X_val, label=Y_val)


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
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
    }
    regressor = lgb.LGBMRegressor(**params)
    regressor.fit(
        X_train,
        Y_train,
    )
    preds = regressor.predict(X_val)
    mse = mean_squared_error(Y_val, preds)
    return mse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
study.best_params, study.best_value

# %%

from yellowbrick.model_selection import LearningCurve

model = lgb.LGBMRegressor(**study.best_params)

sizes = np.linspace(0.1, 1, 5)

vizualizer = LearningCurve(
    estimator=model,
    train_sizes=sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
vizualizer.fit(X_train, Y_train)
vizualizer.show()
# %%
