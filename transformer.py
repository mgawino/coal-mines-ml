# -*- coding: utf-8 -*-
import time

import numpy as np

import click

from sklearn.base import TransformerMixin


def measure_time(fun):
    def inner(self, *args, **kwargs):
        context = self.function.__name__ if hasattr(self, 'function') else self.__class__.__name__
        click.secho('Started {}... '.format(context), fg='yellow')
        start_time = time.process_time()
        result = fun(self, *args, **kwargs)
        elapsed_sec = round(time.process_time() - start_time, 2)
        click.secho('Finished {} in {} sec'.format(context, elapsed_sec), fg='yellow')
        return result
    return inner


class SensorsDataTransformer(TransformerMixin):

    def __init__(self, function, **function_kwargs):
        self.function = function
        self.function_kwargs = function_kwargs

    @measure_time
    def transform(self, X):
        result = []
        for row in X:
            features = []
            for sensor_data in row:
                features.append(self.function(sensor_data, **self.function_kwargs))
            result.append(np.asarray(features, dtype=np.float32))
        return np.asarray(result, dtype=np.float32)
