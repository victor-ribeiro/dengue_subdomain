import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


def prepair_dataset(
    df: pd.DataFrame,
    window_size: int = 5,
) -> List[List[float]]:
    df_out = pd.DataFrame()
    # pegando somente as series
    aux = df.iloc[:, 2:-1].apply(lambda x: (x - x.mean()) / x.std())
    window = []
    # for week in range(1, 52 - window_size):  # df.columns[2:-1]:
    for week in range(1, 52):  # df.columns[2:-1]:
        # janelamento
        # window = [(i % 52) + 1 for i in range(week - 1, (week - 1) + window_size)]
        # window = [i for i in range(week, week + window_size)]
        window.append(week)
        tmp_df = aux.loc[:, window].dropna(axis="index")
        # tmp_df = tmp_df.apply(lambda x: (x - x.mean()) / x.std())
        columns = [f"t_{i}" for i in window]
        tmp_df.columns = columns
        # enriquecimento
        tmp_df["y"] = df.loc[:, window[-1] + 1]
        tmp_df["slug_name"] = df.slug_name
        tmp_df["year"] = df.year
        df_out = pd.concat([df_out, tmp_df], axis="index", ignore_index=False)
    df_out.sort_values(by=["year", "slug_name"], inplace=True)
    df_out.reset_index(inplace=True)
    df_out.drop("index", inplace=True, axis="columns")
    df_out.fillna(0, inplace=True)
    return df_out


WINDOW_SIZE = 4
INTERVAL = {
    "begin": pd.Timestamp("2011-05-01"),
    "end": np.datetime64("2020-03-29"),
}
COLUMNS = ["dt_notific", "dengue_cases", "slug_name"]
DATA_ROOT = Path("./data")

if __name__ == "__main__":
    if not DATA_ROOT.exists():
        DATA_ROOT.mkdir(parents=True)
    data_year_path = DATA_ROOT / "data_year.parquet"
    data_model_path = DATA_ROOT / "data_model.parquet"
    data: pd.DataFrame = pd.read_parquet("dengue_popGT100K2.gz", columns=COLUMNS)

    #################################################################################
    ################# Divisão do dataset em anos ####################################
    #################################################################################

    # tratamento das datas para filtro
    print("[LOADING] reading parquet file", end="\t")
    data["dt_notific"] = pd.to_datetime(data["dt_notific"].values, format="%y.%m.%D")
    print("OK")
    print("[TRANSFORM] building yearly dataframe", end="\t")
    data.dt_notific = data.dt_notific[
        (data.dt_notific >= INTERVAL["begin"]) & (data.dt_notific <= INTERVAL["end"])
    ]
    data["year"] = data.dt_notific.map(lambda x: x.year)
    data["week"] = data.dt_notific.map(lambda x: x.week)
    data.dropna(axis="rows", inplace=True)
    data.year = data.year.astype(int)
    data.week = data.week.astype(int)

    data.drop("dt_notific", axis="columns", inplace=True)
    print("OK")
    print("[TRANSFORM] pivot dataframe", end="\t")
    data = pd.pivot_table(
        data=data,
        index=["slug_name", "year"],
        values=["dengue_cases"],
        columns="week",
        aggfunc=np.sum,
    )
    data.columns = data.columns.droplevel(0)
    data.reset_index(inplace=True)
    print("OK")

    #################################################################################
    ################# prepraro dados treinamento ####################################
    #################################################################################

    print("[TRANSFORM] prepair model data", end="\t")
    data_window = prepair_dataset(df=data, window_size=WINDOW_SIZE)
    print(data_window)
    data_window.fillna(0, inplace=True)
    print("OK")

    #################################################################################
    ################# prepraro dados treinamento ####################################
    #################################################################################

    print(f"[WRITING] writing parquet file {data_window.shape}", end="\t")
    data.columns = data.columns.astype(str)
    data.to_parquet(data_year_path, index=False)
    data_window.to_parquet(data_model_path, index=False)
    print("OK")
