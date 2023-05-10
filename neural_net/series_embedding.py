# %%
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

# from tensorflow.keras.callbaclks import EarlyStopping
# %%

#
# preparo dos dados de treinamento
#


def data_genetator(df):
    data = df[df.year.isin(range(2012, 2019))].copy()
    data = data.fillna(0)
    data = data.drop(["slug_name", "year", "53"], axis="columns")
    data = data.apply(lambda x: (x - x.mean()) / x.std())
    cidades = df.slug_name.unique()
    np.random.shuffle(cidades)
    while True:
        X = pd.DataFrame()
        Y = pd.DataFrame()
        for cidade in cidades:
            x = data.loc[df.slug_name == cidade]
            y = df.loc[(df.slug_name == cidade) & (df.year.isin(range(2012, 2019)))]
            y = y.drop(["slug_name", "year", "53"], axis="columns")
            y = y.fillna(0)
            Y = pd.concat([Y, y], axis="index", ignore_index=False)
            x = x.fillna(0)
            X = pd.concat([X, x], axis="index", ignore_index=False)
        print(X.shape, Y.shape)
        yield X.values, Y.values


data = pd.read_parquet("data/data_year.parquet")
print(data)
x, y = next(data_genetator(df=data))
x.shape, y.shape
# %%

data_train = data_genetator(df=data)
input_size = data.shape[1] - 3
n_dims = 8

decoder_input = Input(shape=(input_size, 1), dtype=np.float32)
# lstm
lstm_layer = LSTM(
    input_size, return_sequences=True, recurrent_dropout=0.05, dropout=0.1
)(decoder_input)
lstm_layer = LSTM(input_size, recurrent_dropout=0.01, dropout=0.1)(lstm_layer)
lstm_layer = Reshape((1, 52))(lstm_layer)
# embedding - codding
embedding_layer = Dense(32, activation="selu")(lstm_layer)
embedding_layer = Dense(16, activation="selu")(embedding_layer)
embedding_layer = Dense(8, activation="selu")(embedding_layer)

# embedding - decoding
embedding_layer = Dense(16, activation="selu")(lstm_layer)
embedding_layer = Dense(32, activation="selu")(embedding_layer)
# embedding_layer = LSTM(32)(embedding_layer)
# output
output_layer = Dense(input_size, activation="relu")(embedding_layer)

print(output_layer.shape)

model = Model(inputs=decoder_input, outputs=output_layer)
model.compile(loss=["binary_crossentropy"], optimizer=Adam(10e-4), metrics=["accuracy"])
model.summary()

# %%
steps = data.shape[0] // 5
epochs = 60
history = model.fit(data_train, epochs=epochs, steps_per_epoch=steps)

import matplotlib.pyplot as plt

# history_df = pd.DataFrame(history.history)
# history_df["accuracy"].plot()
# plt.box(False)
# plt.grid(axis="y")
# plt.show()

# %%
##### embeddin_operator = Model(inputs=encoder_input, outputs=embedding_layer)

for i in range(2012, 2020):
    cidade = "rio-de-janeiro"
    test = data.loc[(data.slug_name == cidade) & (data.year == i)].drop(
        ["slug_name", "year", "53"], axis=1
    )
    x = test.columns.values
    y = test.values.flatten()
    y_hat = model.predict(test)
    plt.plot(x, y.reshape(-1, 1), label="original")
    plt.title(f"{cidade} - {i} ({y_hat.shape})")
    plt.plot(x, y_hat.reshape(-1, 1), label="gerado")
    plt.yscale("log")
    plt.legend()
    plt.show()


# # %%
# test = data.loc[(data.slug_name == "petropolis") & (data.year == 2012)].drop(
#     ["slug_name", "year", "53"], axis=1
# )
# x = test.columns.values
# y = test.values.reshape(-1, 1)
# y_hat = code_decode_model.predict(test.values).reshape(-1, 1)

# plt.plot(x, y, label="original")
# plt.plot(x, y_hat, label="gerado")
# plt.legend()
# plt.show()
# # %%
# data[(data.year == 2016) & (data.slug_name == "rio-de-janeiro")]
# # %%
# %%
