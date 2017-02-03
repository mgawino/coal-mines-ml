# -*- coding: utf-8 -*-
import pandas as pd

import numpy as np

from sklearn.base import TransformerMixin
from utils import measure_time


class SensorTransformer(TransformerMixin):

    def __init__(self, function, **function_kwargs):
        self.function = function
        self.function_kwargs = function_kwargs
        self.sensor_names = None

    def set_sensor_names(self, sensor_names):
        self.sensor_names = sensor_names

    def get_feature_names(self):
        fun_name = self.function.__name__
        fun_kwargs_sufix = '_'.join('{}_{}'.format(k,v) for k, v in self.function_kwargs.items())
        feature_names = ['{}_{}'.format(sensor_name, fun_name)
                         for sensor_name in self.sensor_names]
        if fun_kwargs_sufix:
            feature_names = ['{}_{}'.format(name, fun_kwargs_sufix) for name in feature_names]
        return feature_names

    @measure_time()
    def transform(self, X):
        result = []
        for row in X:
            features = []
            for sensor_data in row:
                features.append(self.function(sensor_data, **self.function_kwargs))
            result.append(np.asarray(features, dtype=np.float32))
        return np.asarray(result, dtype=np.float32)


class SensorMultiTransformer(SensorTransformer):

    def __init__(self, function, **function_kwargs):
        super(SensorMultiTransformer, self).__init__(function, **function_kwargs)
        self.feature_names = None

    def get_feature_names(self):
        return self.feature_names

    @measure_time()
    def transform(self, X):
        result = []
        for row in X:
            features = pd.Series()
            for sensor_name, sensor_data in zip(self.sensor_names, row):
                res = self.function(sensor_data, c=sensor_name, **self.function_kwargs)
                features.append(res)
            if self.feature_names is None:
                self.feature_names = list(features.index)
            result.append(features.values)
        return np.asarray(result, dtype=np.float32)
