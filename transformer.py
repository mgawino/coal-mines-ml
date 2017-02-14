# -*- coding: utf-8 -*-
import pandas as pd

import numpy as np
from reader import DataReader
from sklearn.base import TransformerMixin
from utils import measure_time


class SensorGroupingTransformer(TransformerMixin):

    def __init__(self, sensor_data_count, sensor_group_count):
        assert sensor_data_count % sensor_group_count == 0
        self.sensor_data_count = sensor_data_count
        self.sensor_group_count = sensor_group_count

    def _grouped(self, iterable, count):
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == count:
                yield chunk
                chunk = []
        if len(chunk) > 0:
            yield chunk

    @measure_time
    def transform(self, X):
        assert len(X.shape) == 2
        result = []
        for sensors_data in X:
            groups = []
            for sensor_data in self._grouped(sensors_data, self.sensor_data_count):
                sensor_data_groups = np.asarray(self._grouped(sensor_data, self.sensor_group_count))
                groups.append(sensor_data_groups)
            result.append(np.asarray(groups, dtype=np.float32))
        result = np.asarray(result, dtype=np.float32)
        print('Grouping transformer shape: {}'.format(result.shape))
        return result


class SensorTransformer(TransformerMixin):

    def __init__(self, function, **function_kwargs):
        self.function = function
        self.function_kwargs = function_kwargs
        self.sensor_names = None
        self.sensor_split_interval = None

    def get_feature_names(self):
        assert self.sensor_names is not None
        assert self.sensor_split_interval is not None
        fun_name = self.function.__name__
        fun_kwargs_sufix = '_'.join('{}_{}'.format(k, v) for k, v in self.function_kwargs.items())
        if fun_kwargs_sufix:
            fun_name = '{}_{}'.format(fun_name, fun_kwargs_sufix)
        feature_names = ['{}_{}m-{}m_{}'.format(sensor_name, m, m + self.sensor_split_interval, fun_name)
                         for sensor_name in self.sensor_names
                         for m in range(0, 10, self.sensor_split_interval)]
        return feature_names

    @measure_time
    def transform(self, X):
        assert len(X.shape) == 4
        assert X.shape[1] == DataReader.SENSOR_NUM
        result = []
        for sensors_data in X:
            features = []
            for sensor_data in sensors_data:
                for sensor_group in sensor_data:
                    features.append(self.function(sensor_group, **self.function_kwargs))
            result.append(np.asarray(features, dtype=np.float32))
        result = np.asarray(result, dtype=np.float32)
        return result


class SensorMultiTransformer(SensorTransformer):

    def get_feature_names(self):
        assert self.sensor_names is not None
        assert self.sensor_split_interval is not None
        feature_names = []
        for sensor_name in self.sensor_names:
            for m in range(0, 10, self.sensor_split_interval):
                name = '{}_{}m-{}m'.format(sensor_name, m, m + self.sensor_split_interval)
                res = self.function([0], c=name, **self.function_kwargs)
                feature_names.extend(list(res.index))
        return feature_names

    @measure_time
    def transform(self, X):
        assert len(X.shape) == 4
        assert X.shape[1] == DataReader.SENSOR_NUM
        result = []
        for sensors_data in X:
            features = pd.Series()
            for sensor_name, sensor_data in zip(self.sensor_names, sensors_data):
                for sensor_group in sensor_data:
                    res = self.function(sensor_group, c=sensor_name, **self.function_kwargs)
                    features = features.append(res)
            result.append(features.values)
        result = np.asarray(result, dtype=np.float32)
        return result
