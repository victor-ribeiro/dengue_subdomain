
import numpy as np
import pandas as pd
import time
from datetime import datetime

# models
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

# errors
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

import csv
from collections import namedtuple


LAGS = 12  # how many points in the past to include
# HORIZONS = [4, 8, 12]
HORIZONS = [12]  # 4 weeks
ROLLINGS = [2, 3, 4, 6, 8, 12]  # for moving average lags
# we are considering that the past #cases, past precipitation, past tempMax influences present #cases

LAGGED_FEATURES = ['cases', 'pr_sum',	'Tmax_max']

MODEL_FEATURES = ['cases', 'pr_sum', 'Tmax_max']

DENGUE_INTERVAL_BEGIN = '2011-05-01'
DENGUE_INTERVAL_END = '2020-03-29'
# Expected timeserie points considering the interval ('2011-05-01', '2020-03-29')
EXPECTED_DATA_POINTS = 466

CASES_NORMALIZATION_FACTOR = 100000  # CASES/100k

# filters out cities that do not have all points between bein and end

#MODELS = ['arimaSimple', 'local_LGBM', 'all_LGBM', 'meso_LGBM', 'mesoExc_LGBM', 'micro_LGBM', 'microExc_LGBM', 'kshape6_LGBM', 'kshape6Exc_LGBM', 'kshape17_LGBM', 'kshape17Exc_LGBM']

MODELS = ['arimaSimple', 'local_LGBM', 'all_LGBM', 'meso_LGBM',
          'micro_LGBM',  'kshape6_LGBM',  'kshape17_LGBM']


def prepare_city_filter_range(maindf, city_name, begin=DENGUE_INTERVAL_BEGIN, end=DENGUE_INTERVAL_END):
    data = maindf[maindf['slug_name'] == city_name]
    data = data.copy()
    data['dt_notific'] = pd.to_datetime(data['dt_notific'], format='%Y-%m-%d')
    data['cases'] = data['dengue_cases'] / \
        (data['pop_21']/CASES_NORMALIZATION_FACTOR)
    data = data.set_index('dt_notific')
    data = data.asfreq('W')
    data = data.sort_index()
    data = data.loc[begin:end]
    if data.shape[0] < EXPECTED_DATA_POINTS:
        return None
    else:
        return data


def create_features(df, adjusted_horizon, lagged_features=LAGGED_FEATURES):
    df = df.reset_index()
    df['week_number'] = df['dt_notific'].dt.isocalendar().week.astype('int')
    df['quarter'] = df['dt_notific'].dt.quarter
    df['month'] = df['dt_notific'].dt.month

    for i in range(adjusted_horizon):
        fname = f'cases{i+2}'
        df[fname] = df['cases'].shift(-i-1)

    for f in lagged_features:
        for i in range(LAGS):
            fname = f'{f}_lag{i+1}'
            df[fname] = df[f].shift(i+1)

    for f in lagged_features:
        for i in ROLLINGS:
            fname = f'{f}_rolling_mean{i}W'
            df[fname] = df[f].rolling(window=i).mean()

    df = df.set_index('dt_notific')
    df = df.dropna()  # droping points without moving averages

    return df


def load_subdomain_dicts(maindf, cities2remove):

    # Meso REgion dict
    groups = maindf.groupby('mesoregion')['slug_name'].apply(set)
    global meso_dict
    meso_dict = {}
    for cities in groups.values:
        for city in cities:
            if city in cities2remove:
                continue
            meso_dict[city] = list(set(cities) - cities2remove)

    # Micro Region dict
    groups = maindf.groupby('microregion')['slug_name'].apply(set)
    global micro_dict
    micro_dict = {}
    for cities in groups.values:
        for city in cities:
            if city in cities2remove:
                continue
            micro_dict[city] = list(set(cities) - cities2remove)

     # KSHAPE with clusters=6 dict
    cdf6 = pd.read_csv('clustering/kshape_clusters6.csv', sep=',')
    groups = cdf6.groupby('ClusterId')['slug_name'].apply(set)
    global kshape6_dict
    kshape6_dict = {}
    for cities in groups.values:
        for city in cities:
            kshape6_dict[city] = list(cities)

    # KSHAPE with clusters=17 dict
    cdf17 = pd.read_csv('clustering/kshape_clusters17.csv', sep=',')
    groups = cdf17.groupby('ClusterId')['slug_name'].apply(set)
    global kshape17_dict
    kshape17_dict = {}
    for cities in groups.values:
        for city in cities:
            kshape17_dict[city] = list(cities)


def generate_city_dfs(mainfile, adjusted_horizon, model_features=MODEL_FEATURES):

    start_time = time.time()

    maindf = pd.read_parquet(mainfile)

    print('Loading cities data and creating features... ')
    cities_data_dict = {}

    cities2remove = set()

    for city_name in maindf['slug_name'].unique():

        data = prepare_city_filter_range(maindf, city_name)
        if data is not None:
            data = data[model_features]  # filter non-used columns
            # create lagged features
            data = create_features(
                data, adjusted_horizon, lagged_features=model_features)

            cities_data_dict[city_name] = data
        else:
            cities2remove.add(city_name)

    print(f'Loaded {len(cities_data_dict)} cities')
    print(
        f'Discarding {len(cities2remove)} cities, because they are incomplete')
    print("--- Time to Load the data and create the features: %s secs ---" %
          (round(time.time() - start_time, 2)))

    load_subdomain_dicts(maindf, cities2remove)

    return cities_data_dict

# gien the city names in cluster, returns


def get_subdomain_train_data(cities_data_dict, citiesInCluster, targets, end):

    X_train_list = []
    y_train_list = []

    # print(len(citiesInCluster))
    for city in citiesInCluster:
        df = cities_data_dict[city]
        df.index = range(len(df.index))  # important for slicing

        y = df[targets].copy()
        X = df.drop(targets, axis=1)

        X_train, y_train = X.loc[0: end-1], y.loc[0: end-1]
        X_train_list.append(X_train.values)
        y_train_list.append(y_train.values)

    X_trains = np.vstack(X_train_list)
    y_trains = np.vstack(y_train_list)

    return X_trains, y_trains


def run_lgb_prediction(X_train, y_train, X_test):

    model = lgb.LGBMRegressor(
        learning_rate=0.09, max_depth=-5, random_state=42)
    # Define MultiOutputRegressor model with LightGBM as the estimator
    multi_model = MultiOutputRegressor(model)
    multi_model.fit(X_train, y_train)  # Fit the model to the training data
    y_pred = multi_model.predict([X_test])

    return y_pred


def run_for_model(X_test, model_name, subdomain_city_names, cities_data, targets, end, y_model_predictions, model_times):

    # Model execution: all_LGBM
    start_time = time.time()

    X_trainsub, y_trainsub = get_subdomain_train_data(
        cities_data, subdomain_city_names, targets, end)  # it is fast for the data we have

    # most time consuming part (particularly for the "all cities" model)
    y_pred = run_lgb_prediction(X_trainsub, y_trainsub, X_test)

    y_model_predictions[model_name].extend(y_pred[0])
    model_times[model_name] += (round(time.time() - start_time, 2))


def runModels(cities_data, city,  horizon):
    # cases, cases2,  ... casesHorizon
    targets = ['cases'] + [('cases'+str(h)) for h in range(2, horizon+1)]

    # format the dataframe for the target city
    city_df = cities_data[city]
    city_df.index = range(len(city_df.index))
    y = city_df[targets].copy()
    X = city_df.drop(targets, axis=1)

    # Define walk forward validation parameters

    time_series_size = X.shape[0]  # the same as EXPECTED_DATA_POINTS

    # using ifs to avoid arbitrary horizons other than 4,8,12
    # it could be  first_end = time_series_size - (96/horizon)
    if horizon == 4:
        # we want to predict 24 months (24 * 4 weeks)
        first_end = time_series_size - (horizon * 24)
    elif horizon == 8:
        # we want to predict 24 months (12 * 8 weeks)
        first_end = time_series_size - (horizon * 12)
    elif horizon == 12:
        # we want to predict 24 months (8 * 12 weeks)
        first_end = time_series_size - (horizon * 8)

    y_real_values = []  # real values, for error calculation

    # for each model, a dictionary entry that stores all predictions
    y_model_predictions = {}
    for model in MODELS:
        y_model_predictions[model] = []

    # for each model, a dictionary entry that stores all training/prediction times
    model_times = {}
    for model in MODELS:
        model_times[model] = 0

    for end in range(first_end, time_series_size-1, horizon):

        X_train, X_test = X.loc[0: end-1], X.loc[end]
        y_train, y_test = y.loc[0: end-1], y.loc[end]

        # form a simple list for later plotting
        y_real_values.extend(y_test)

        ### Model execution: arimaSimple -- does not need the X_Train, as we use the data from the dataframe ###
        start_time = time.time()

        model_name = 'arimaSimple'

        series = city_df['cases'].loc[0: end-1].values
        model = ARIMA(series, order=(1, 0, 1))
        model_fit = model.fit()
        y_pred = model_fit.forecast(horizon)
        y_model_predictions[model_name].extend(y_pred)

        model_times[model_name] += (round(time.time() - start_time, 2))

        ### Model execution: local_LGBM ###
        start_time = time.time()

        model_name = 'local_LGBM'

        y_pred = run_lgb_prediction(X_train, y_train, X_test)
        y_model_predictions[model_name].extend(y_pred[0])

        model_times[model_name] += (round(time.time() - start_time, 2))

        ### PLEASE, provide the model building and prediction FOR EACH model IN MODELS. See examples below ####

        run_for_model(X_test, 'all_LGBM', cities_data.keys(
        ), cities_data, targets, end, y_model_predictions, model_times)

        run_for_model(X_test, 'meso_LGBM',
                      meso_dict[city], cities_data, targets, end, y_model_predictions, model_times)

        run_for_model(X_test, 'micro_LGBM',
                      micro_dict[city], cities_data, targets, end, y_model_predictions, model_times)

        run_for_model(X_test, 'kshape6_LGBM',
                      kshape6_dict[city], cities_data, targets, end, y_model_predictions, model_times)

        run_for_model(X_test, 'kshape17_LGBM',
                      kshape17_dict[city], cities_data, targets, end, y_model_predictions, model_times)
        



        #########################

    for model, predictions in y_model_predictions.items():

        # nmse
        nmse = mean_squared_error(y_real_values, predictions, squared=False)
        # print(f'Error for {city} with {model} model: {error}')
        result = resultTup(horizon, city, model, 'nmse',
                           nmse, model_times[model])
        resultOutput.writerow(list(result))

        # mape
        mape = mean_absolute_percentage_error(y_real_values, predictions)
        # print(f'Error for {city} with {model} model: {error}')
        result = resultTup(horizon, city, model, 'mape',
                           mape, model_times[model])
        resultOutput.writerow(list(result))


def load_run():

    # main parquet file
    mainfile = 'dados/dengue_popGT100K.gzip'

    # set the forecast horizons
    for horizon in HORIZONS:

        resultFields = ['horizon', 'city', 'model',
                        'errorMetric', 'errorValue', 'modelTime']
        global resultTup
        resultTup = namedtuple('resultTup', resultFields)

        # open a (new) file to write
        global resultFile
        dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        resultFileName = f'results/result_horizon{horizon}_{dt_string}.csv'
        resultFile = open(resultFileName, "w")

        global resultOutput
        resultOutput = csv.writer(resultFile)
        resultOutput.writerow(resultFields)

        # r1 = resultTup(horizon,'sao-paulo','arima','mape',223.4,233232)
        # resultOutput.writerow(list(r1))

        adjusted_horizon = horizon - 1  # only for the feature enginnering
        # load the data with features
        cities_data = generate_city_dfs(mainfile, adjusted_horizon)

        # could be any other city, as we just need the index data
        global global_date_index
        # for later use in plotting
        global_date_index = cities_data['rio-de-janeiro'].index.copy()

        # target_cities = cities_data.keys() ## all cities
        #target_cities = ['sao-paulo']
        target_cities = ['palmas', 'rio-de-janeiro', 'brasilia']

        count = 0
        total = len(target_cities)

        # past = pd.read_csv('results/result_horizon4_14042023_185348.csv',sep=',') ## past experiment
        # seen = set(past['city'].unique())

        for city in target_cities:

            # if city in seen:
            #     print('Skipping', city)
            #     continue

            start_time_city = time.time()

            print(f'-Running models for {city}')

            runModels(cities_data, city, horizon)

            print(f'Time: ', (round(time.time() - start_time_city, 2)))

            count += 1
            print(f'Processed {count} ou of {total} cities.')

        resultFile.close()


def main():
    load_run()


if __name__ == "__main__":
    main()
