import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
import json
from utils import *

print("-- load and data transform -")
data = read_data_file(data_path=DATA_PATH, columns=COLUMNS)
data = data.T
n_rows, n_cols = data.shape

scaler = TimeSeriesScalerMeanVariance()
scaler.fit(data)
scaled = scaler.transform(data)
metric = "dtw"
cities = data.index.values
for i in range(2, 13):
    clusters = {}
    model = TimeSeriesKMeans(
        metric=metric,
        n_clusters=i,
        n_jobs=-1,
        random_state=42,
        metric_params={"global_constraint": "itakura"},
    )
    print(f"[CLUSTERING] model {model.__class__.__name__} - [{i}]")
    model.fit(scaled)
    cls_idx = model.predict(scaled)
    for j in np.unique(cls_idx).tolist():
        clusters[j] = cities[(cls_idx == j).tolist()].tolist()
    print("-- writing cluster json -- ")
    file_name = f"{model.__class__.__name__}_{metric}_k_{i}.json"
    with open(f"clustering/{file_name}", "w") as file:
        json.dump(fp=file, obj=clusters, indent=4)
    print("DONE")
