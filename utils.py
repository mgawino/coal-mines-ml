# -*- coding: utf-8 -*-

import csv
import os

import click
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from transformer import SensorsDataTransformer
from tsfresh.feature_extraction import feature_calculators


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


class DataReader(object):
    SENSOR_DATA_COUNT_IN_ROW = 600
    SENSOR_NUM = 28
    SENSOR_NAMES_FILE_PATH = "data/column_names.txt"
    TRAINING_FILE_PATHS = ("data/trainingData{}.csv".format(i) for i in range(1, 6))
    TRAINING_LABEL_PATHS = ("data/trainingLabels{}.csv".format(i) for i in range(1, 6))
    TEST_FILE_PATH = "data/testData.csv"
    TEST_LABELS_PATH = "data/testLabels.csv"

    @classmethod
    def _iter_file_data(cls, filepaths):
        for filepath in filepaths:
            file_data = []
            with open(filepath) as file:
                print('Reading file: {} ...'.format(filepath))
                reader = csv.reader(file)
                for row in reader:
                    sensors_data = list(grouped(row, cls.SENSOR_DATA_COUNT_IN_ROW))
                    assert len(sensors_data) == cls.SENSOR_NUM
                    file_data.append(np.asarray(sensors_data, dtype=np.float32))
            yield np.asarray(file_data, dtype=np.float32)

    @classmethod
    def read_test_data(cls):
        return next(cls._iter_file_data((cls.TEST_FILE_PATH,)))

    @classmethod
    def iter_train_files_data(cls):
        yield from cls._iter_file_data(cls.TRAINING_FILE_PATHS)

    @staticmethod
    def _iter_labels(filepaths):
        for filepath in filepaths:
            with open(filepath) as file:
                reader = csv.reader(file)
                for sensor_labels in reader:
                    assert len(sensor_labels) == 3
                    yield np.asarray(sensor_labels)

    @classmethod
    def read_test_labels(cls):
        return np.array(list(cls._iter_labels((cls.TEST_LABELS_PATH,))))

    @classmethod
    def read_training_labels(cls):
        return np.array(list(cls._iter_labels(cls.TRAINING_LABEL_PATHS)))

    @classmethod
    def get_sensor_names(cls):
        with open(cls.SENSOR_NAMES_FILE_PATH) as file:
            seen_names = set()
            sensor_names = []
            for column_name in file:
                sensor_name = column_name.split('_')[0]
                if sensor_name not in seen_names:
                    sensor_names.append(sensor_name)
                seen_names.add(sensor_name)
            assert len(sensor_names) == cls.SENSOR_NUM
            return sensor_names


class FeatureExtractor:
    TRAIN_FEATURES_CACHE_PATH = '/tmp/train_features_cache.npy'
    TEST_FEATURES_CACHE_PATH = '/tmp/test_features_cache.npy'

    def __init__(self):
        self.transformer = Pipeline([
            ('features', FeatureUnion([
                ('max', SensorsDataTransformer(max)),
                ('min', SensorsDataTransformer(min)),
                ('mean', SensorsDataTransformer(feature_calculators.mean)),
                ('median', SensorsDataTransformer(feature_calculators.median)),
                ('variance', SensorsDataTransformer(feature_calculators.variance)),
                ('skewness', SensorsDataTransformer(feature_calculators.skewness)),
                ('kurtosis', SensorsDataTransformer(feature_calculators.kurtosis)),
                ('std', SensorsDataTransformer(feature_calculators.standard_deviation)),
                ('sum_values', SensorsDataTransformer(feature_calculators.sum_values)),
                ('mean_abs_change', SensorsDataTransformer(feature_calculators.mean_abs_change)),
                ('mean_autocorrelation', SensorsDataTransformer(feature_calculators.mean_autocorrelation)),
                ('abs_energy', SensorsDataTransformer(feature_calculators.abs_energy))
            ], n_jobs=3))
        ])
        sensor_names = DataReader.get_sensor_names()
        self.transformer_names, _ = zip(*self.transformer.named_steps['features'].transformer_list)
        self.feature_names = ['{}_{}'.format(transformer_name, sensor_name)
                              for transformer_name in self.transformer_names
                              for sensor_name in sensor_names]

    @classmethod
    def clear_cache(cls):
        for path in (cls.TRAIN_FEATURES_CACHE_PATH, cls.TEST_FEATURES_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

    def _load_features_from_cache(self):
        train_features = np.load(self.TRAIN_FEATURES_CACHE_PATH)
        test_features = np.load(self.TEST_FEATURES_CACHE_PATH)
        return train_features, test_features

    @staticmethod
    def _cache_features(features, cache_path):
        np.save(cache_path, features)

    def _transform_data_to_features(self):
        train_features_partials = []
        for X_train_partial in DataReader.iter_train_files_data():
            train_features_partial = self.transformer.transform(X_train_partial)
            assert train_features_partial.shape[0] == X_train_partial.shape[0]
            assert train_features_partial.shape[1] == len(self.transformer_names) * DataReader.SENSOR_NUM
            train_features_partials.append(train_features_partial)
        train_features = np.concatenate(train_features_partials, axis=0)
        X_test = DataReader.read_test_data()
        test_features = self.transformer.transform(X_test)
        return train_features, test_features

    def load_features(self):
        if os.path.exists(self.TRAIN_FEATURES_CACHE_PATH) and\
           os.path.exists(self.TEST_FEATURES_CACHE_PATH):
            click.secho('Loading features from cache', fg='blue')
            train_features, test_features = self._load_features_from_cache()
        else:
            click.secho('Transforming data to features', fg='blue')
            train_features, test_features = self._transform_data_to_features()
            self._cache_features(train_features, self.TRAIN_FEATURES_CACHE_PATH)
            self._cache_features(test_features, self.TEST_FEATURES_CACHE_PATH)

        assert train_features.shape[1] == test_features.shape[1] == len(self.feature_names)
        click.secho('Train features shape: {}'.format(train_features.shape))
        click.secho('Test features shape: {}'.format(test_features.shape))
        return train_features, test_features, self.feature_names
