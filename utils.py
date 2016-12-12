# -*- coding: utf-8 -*-

import csv
import os

import click
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from transformer import MatrixTransformer, SensorsDataTransformer
from tsfresh.feature_extraction import feature_calculators


class DataReader(object):
    SENSOR_DATA_COUNT_IN_ROW = 600
    SENSOR_NUM = 28
    SENSOR_NAMES_FILE_PATH = "data/column_names.txt"
    TRAINING_FILE_PATHS = ("data/trainingData{}.csv".format(i) for i in range(1, 2))
    TRAINING_LABEL_PATHS = ("data/trainingLabels{}.csv".format(i) for i in range(1, 2))
    TEST_FILE_PATH = "data/testData.csv"
    TEST_LABELS_PATH = "data/testLabels.csv"

    @classmethod
    def _iter_data(cls, filepaths):
        for filepath in filepaths:
            with open(filepath) as file:
                print('Reading file: {} ...'.format(filepath))
                reader = csv.reader(file)
                for sensor_data in reader:
                    assert len(sensor_data) == cls.SENSOR_DATA_COUNT_IN_ROW * cls.SENSOR_NUM
                    yield np.asarray(sensor_data, dtype=np.float)

    @classmethod
    def read_test_data(cls):
        return np.array(list(cls._iter_data((cls.TEST_FILE_PATH,))))

    @classmethod
    def read_training_data(cls):
        return np.array(list(cls._iter_data(cls.TRAINING_FILE_PATHS)))

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
        sensor_names = DataReader.get_sensor_names()
        self.transformer = Pipeline([
            ('dataframe', MatrixTransformer(sensor_names, DataReader.SENSOR_DATA_COUNT_IN_ROW)),
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
        transformer_names, _ = zip(*self.transformer.named_steps['features'].transformer_list)
        self.feature_names = ['{}_{}'.format(transformer_name, sensor_name)
                              for transformer_name in transformer_names
                              for sensor_name in sensor_names]

    @staticmethod
    def _load_raw_data():
        X_train = DataReader.read_training_data()
        click.secho('Loaded training data: {}'.format(X_train.shape), fg='green')

        X_test = DataReader.read_test_data()
        click.secho('Loaded test data: {}'.format(X_test.shape), fg='green')
        return X_train, X_test

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
        X_train = DataReader.read_training_data()
        train_features = self.transformer.transform(X_train)
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
