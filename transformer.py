# -*- coding: utf-8 -*-
import time

import click
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


class MatrixTransformer(TransformerMixin):

    def __init__(self, sensor_names, sensor_data_len):
        self.data_type = np.dtype([
            (sensor_name, np.float32, (sensor_data_len,)) for sensor_name in sensor_names
        ])
        self.sensor_data_len = sensor_data_len
        self.sensor_names = sensor_names

    @measure_time
    def transform(self, X):
        matrix = np.empty(X.shape[0], dtype=self.data_type)
        for row_ix, data_row in enumerate(X):
            sensors_data = list(grouped(data_row, self.sensor_data_len))
            for sensor_name, sensor_data in zip(self.sensor_names, sensors_data):
                matrix[row_ix][sensor_name] = sensor_data
        del X
        return matrix


class SensorsDataTransformer(TransformerMixin):

    def __init__(self, sensor_names, function, **function_kwargs):
        self.sensor_names = sensor_names
        self.function = function
        self.function_kwargs = function_kwargs

    def get_feature_names(self):
        return self.sensor_names

    @measure_time
    def transform(self, X):
        result = np.empty(shape=(X.shape[0], len(self.sensor_names)), dtype=np.float32)
        for row_ix, row in enumerate(X):
            for col_ix, sensor_name in enumerate(self.sensor_names):
                result[row_ix][col_ix] = self.function(row[sensor_name], **self.function_kwargs)
        return result
