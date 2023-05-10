import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from keras import regularizers

import random


def data_generator(df, window_cols, batch_size=32):
    n_batches = df.shape[0] // batch_size
    cidades = df.slug_name.unique()
    random.shuffle(cidades)
    while True:
        for cidade in cidades:
            aux = df.loc[df.slug_name == cidade]
            aux.reset_index(inplace=True, drop=True)
            for samples in range(0, n_batches + 1, batch_size):
                samples_df = aux.iloc[samples : samples + batch_size]
                yield samples_df[window_cols].values, samples_df.y.values


BATCH_SIZE = 256

if __name__ == "__main__":
    data = pd.read_parquet("data/data_model.parquet")
    DENSE_UNT = [16, 8]
    WINDOW_COLS = [i for i in data.columns if "t" in i]
    WINDOW_COLS

    general_model = Sequential()

    general_model.add(Input((len(WINDOW_COLS), 1)))
    regL2 = regularizers.L1L2(0.1)
    [
        general_model.add(
            LSTM(
                52,
                activation="tanh",
                return_sequences=True,
                recurrent_dropout=0.01,
                dropout=0.05,
                recurrent_regularizer=regL2,
            )
        )
        for _ in range(4)
    ]

    general_model.add(
        LSTM(
            52,
            activation="tanh",
            # recurrent_dropout=0.05,
            recurrent_regularizer=regL2,
        )
    )
    [
        (
            # general_model.add(Dense(unds, activation="relu")),
            general_model.add(
                Dense(unds, activation="linear", kernel_regularizer=regL2)
            ),
            general_model.add(Dropout(0.05)),
        )
        for unds in DENSE_UNT
    ]

    general_model.add(Dense(1, "relu"))
    general_model.summary()

    train_init = 2011
    train_end = 2018
    data_train = data[data.year.isin(range(train_init, train_end))]
    train_steps = data_train.shape[0] // BATCH_SIZE
    data_train = data_generator(
        df=data_train, window_cols=WINDOW_COLS, batch_size=BATCH_SIZE
    )

    val_init = 2018
    val_end = 2020
    data_val = data[data.year.isin(range(val_init, val_end))]
    val_steps = data_val.shape[0] // BATCH_SIZE
    data_val = data_generator(
        df=data_val, window_cols=WINDOW_COLS, batch_size=BATCH_SIZE
    )

    cp_rio = ModelCheckpoint(
        "general_model/", save_best_only=True, mode="min", monitor="val_loss"
    )
    gradient = Adam(
        10e-6,
        # momentum=0.09
    )
    general_model.compile(
        loss=MeanSquaredError(), optimizer=gradient, metrics=[RootMeanSquaredError()]
    )
    history = general_model.fit(
        data_train,
        steps_per_epoch=train_steps,
        validation_data=data_val,
        validation_steps=val_steps,
        callbacks=[
            cp_rio,
            EarlyStopping(patience=30, monitor="val_loss"),
        ],
        epochs=300,
    )
    general_model = load_model("general_model/")
    history = pd.DataFrame(history.history)
    history

    fig, ax = plt.subplots(2, 1, figsize=(17, 10))

    LOSS_COLS = [
        "loss",
        "val_loss",
    ]
    ERROR = ["root_mean_squared_error", "val_root_mean_squared_error"]

    sns.lineplot(history[LOSS_COLS], ax=ax[0])
    ax[0].set_title("Loss function")
    ax[0].grid(axis="y")

    sns.lineplot(history[ERROR], ax=ax[1])
    ax[1].set_title("Error function")
    ax[1].grid(axis="y")
    plt.savefig("figs/history_selu_3.png")
    plt.show()

    original_data = pd.read_parquet("data/data_year.parquet")
    original_data = original_data[original_data.year == 2020]
    original_data

    cidades = [
        "rio-de-janeiro",
        "belo-horizonte",
        "vitoria",
        "sao-paulo",
    ]

    for cidade in cidades:
        test_init = 2019
        test_end = 2021
        data_test = data[
            data.year.isin(range(test_init, test_end)) & (data.slug_name == cidade)
        ]

        data_test = data_test.drop(["slug_name", "year", "y"], axis="columns")
        data_test = data_test[data_test.shape[0] - 13 :].values

        rio_data = original_data[(original_data.slug_name == cidade)]
        rio_data = rio_data.dropna(axis="columns")
        rio_data.reset_index(inplace=True)
        rio_data.drop(["index", "year", "slug_name"], inplace=True, axis="columns")
        rio_data = rio_data[: data_test.shape[0]].T
        rio_data[1] = general_model.predict(data_test)
        rio_data.columns = ["real", "pred"]
        rio_data.plot()
        plt.box(False)
        plt.grid(axis="y")
        # plt.yscale("log")
        plt.title(cidade)
        plt.savefig(f"figs/{cidade}_pred.png")
        plt.show()
