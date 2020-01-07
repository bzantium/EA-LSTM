import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def load_data(data_name):
    path = os.path.join('data', data_name)
    dataset = pd.read_csv(path).dropna()
    data = dataset.values
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(np.expand_dims(data[:, -4], axis=1)).toarray()
    data = np.delete(data, -4, axis=1)
    data = np.concatenate((data, one_hot), axis=1)
    data = data.astype('float32')
    scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(data[:, 1:])
    scaled_y = y_scaler.fit_transform(np.expand_dims(data[:, 0], axis=1))
    scaled_data = np.concatenate((scaled_y, scaled_X), axis=1)
    return scaled_data, y_scaler


def data_to_series_features(data, time_steps):
    data_size = len(data) - time_steps
    series_X = []
    series_y = []
    for i in range(data_size):
        series_X.append(data[i:i + time_steps])
        series_y.append(data[i + time_steps, 0])
    series_X = np.array(series_X)
    series_y = np.array(series_y)
    return series_X, series_y


def is_minimum(value, indiv_to_rmse):
    if len(indiv_to_rmse) == 0:
        return True
    temp = list(indiv_to_rmse.values())
    return True if value < min(temp) else False


def apply_weight(series_X, weight):
    weight = np.array(weight)
    weighted_series_X = series_X * np.expand_dims(weight, axis=1)
    return weighted_series_X
