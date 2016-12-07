# -*- coding: utf-8 -*-
import pandas as pd
from reader import DataReader
from sklearn.base import TransformerMixin
from tsfresh import extract_features
from tsfresh.feature_extraction import FeatureExtractionSettings


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


class DataFrameTransformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, sensor_data_iterable):
        sensor_names = DataReader.get_sensor_names()
        result = pd.DataFrame()
        for row_ix, data_row in enumerate(sensor_data_iterable):
            sensors_data = grouped(data_row, DataReader.SENSOR_DATA_COUNT_IN_ROW)
            df_columns = [pd.Series(sensor_data) for sensor_data in sensors_data]
            df = pd.DataFrame.from_items(zip(sensor_names, df_columns))
            assert df.shape == (DataReader.SENSOR_DATA_COUNT_IN_ROW, DataReader.SENSOR_NUM), df.shape
            df['id'] = row_ix
            result = pd.concat([result, df], ignore_index=True)
        return result


class FeatureExtractorTransformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = extract_features(X, column_id="id", feature_extraction_settings=FeatureExtractionSettings())
        return X