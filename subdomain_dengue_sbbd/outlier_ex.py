# %%
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as regressor
from general_model_tuning import prepair_attributes
from in_domain_model_training import read_clusters
from sklearn.model_selection import train_test_split

# %%
OUTLIER = "sobral"
data = read_data_file(data_path=DATA_PATH, columns=COLUMNS)
K = []
CLUSTER_LABEL = []

for cluster in read_clusters("clustering"):
    for k, v in cluster.items():
        # treinamento do modelo pra cada cluster
        if OUTLIER in v:
            K.append(len(cluster))
            CLUSTER_LABEL.append(k)
# %%
