# -*- coding: utf-8 -*-
import numpy as np

from sklearn.base import TransformerMixin
from utils import measure_time


class SensorsDataTransformer(TransformerMixin):

    def __init__(self, function, **function_kwargs):
        self.function = function
        self.function_kwargs = function_kwargs

    @measure_time()
    def transform(self, X):
        result = []
        for row in X:
            features = []
            for sensor_data in row:
                features.append(self.function(sensor_data, **self.function_kwargs))
            result.append(np.asarray(features, dtype=np.float32))
        return np.asarray(result, dtype=np.float32)
