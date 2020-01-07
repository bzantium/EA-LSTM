import pandas as pd
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import numpy as np


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_feature, target_value):
        self.input_feature = input_feature
        self.target_value = target_value


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


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
    scaled_X_data = scaler.fit_transform(data[:, 1:])
    scaled_y_data = y_scaler.fit_transform(np.expand_dims(data[:, 0], axis=1))
    scaled_data = np.concatenate((scaled_y_data, scaled_X_data), axis=1)
    return scaled_data, y_scaler


def data_to_series_features(data, time_steps):
    data_size = len(data) - time_steps
    target_values = data[:, 0]
    input_features = data
    series_features = []
    for i in range(data_size):
        series_features.append(InputFeatures(input_feature=input_features[i:i + time_steps],
                                             target_value=target_values[i + time_steps]))
    return series_features


def get_data_loader(features, batch_size):
    all_input_features = torch.tensor([f.input_feature for f in features], dtype=torch.float32)
    all_target_values = torch.tensor([f.target_value for f in features], dtype=torch.float32)
    dataset = TensorDataset(all_input_features, all_target_values)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def is_minimum(value, indiv_to_rmse):
    if len(indiv_to_rmse) == 0:
        return True
    temp = list(indiv_to_rmse.values())
    return True if value < min(temp) else False
