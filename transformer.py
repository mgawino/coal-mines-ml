# -*- coding: utf-8 -*-
import numpy as np

from sklearn.base import TransformerMixin


def grouped(iterable, count):
    """ Group @iterable into lists of length @count """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == count:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class SensorFunctionTransformer(TransformerMixin):
    SENSOR_VALUES_IN_ROW = 600

    def __init__(self, func, sensor_names):
        self.func = func
        self.sensor_names = sensor_names

    def get_feature_names(self):
        return self.sensor_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for row in X:
            for sensor_data in grouped(row, self.SENSOR_VALUES_IN_ROW):
                features.append(self.func(sensor_data))
        return np.array(features)
