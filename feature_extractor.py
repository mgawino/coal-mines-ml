# -*- coding: utf-8 -*-
import os

import numpy as np

import click
from sklearn.pipeline import FeatureUnion, Pipeline
from transformer import SensorTransformer, SensorMultiTransformer, SensorGroupingTransformer
from tsfresh.feature_extraction.feature_calculators import (
    mean_change,
    first_location_of_maximum,
    last_location_of_maximum,
    binned_entropy,
    mean_abs_change,
    absolute_sum_of_changes,
    cwt_coefficients,
    fft_coefficient
)
from reader import DataReader


class SensorFeatureUnion(FeatureUnion):

    def get_feature_names(self):
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not "
                                     "provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            feature_names.extend(trans.get_feature_names())
        return feature_names


class SensorPipeline(Pipeline):

    def get_feature_names(self):
        return self.steps[-1][1].get_feature_names()


def abs_energy(x):
    return np.sum(x * x)


class FeatureExtractor:
    TRAIN_FEATURES_CACHE_PATH = os.path.expanduser('~/train_features_cache.npy')
    TEST_FEATURES_CACHE_PATH = os.path.expanduser('~/test_features_cache.npy')
    FEATURE_NAMES_CACHE_PATH = os.path.expanduser('~/feature_names_cache.npy')

    @staticmethod
    def make_feature_transformer_pipeline(sensor_split_interval, n_jobs):
        feature_transformers = [
            ('max', SensorTransformer(np.max)),
            ('min', SensorTransformer(np.min)),
            ('first_location_of_maximum', SensorTransformer(first_location_of_maximum)),
            ('last_location_of_maximum', SensorTransformer(last_location_of_maximum)),
            ('binned_entropy_5', SensorTransformer(binned_entropy, max_bins=5)),
            ('mean', SensorTransformer(np.mean)),
            ('median', SensorTransformer(np.median)),
            ('variance', SensorTransformer(np.var)),
            ('std', SensorTransformer(np.std)),
            ('sum_values', SensorTransformer(np.sum)),
            ('mean_change', SensorTransformer(mean_change)),
            ('mean_abs_change', SensorTransformer(mean_abs_change)),
            ('absolute_sum_of_changes', SensorTransformer(absolute_sum_of_changes)),
            ('abs_energy', SensorTransformer(abs_energy)),
            ('percentile_10', SensorTransformer(np.percentile, q=10)),
            ('percentile_20', SensorTransformer(np.percentile, q=20)),
            ('percentile_80', SensorTransformer(np.percentile, q=80)),
            ('percentile_90', SensorTransformer(np.percentile, q=90)),
            ('fft_coefficent', SensorMultiTransformer(
                fft_coefficient,
                param=[{'coeff': coeff} for coeff in range(5)]
            )),
            # ('cwt_coeff', SensorMultiTransformer(
            #      cwt_coefficients,
            #      param=[{'coeff': coeff, 'widths': (2, 5, 10, 20), 'w': w}
            #             for coeff in range(15) for w in (2, 5, 10, 20)]
            #  ))
        ]
        sensor_names = DataReader.get_sensor_names()
        for _, feature_transformer in feature_transformers:
            feature_transformer.sensor_names = sensor_names
            feature_transformer.sensor_split_interval = sensor_split_interval
        sensor_group_count = DataReader.SENSOR_DATA_COUNT_IN_ROW / sensor_split_interval
        return SensorPipeline([
            ('groups', SensorGroupingTransformer(
                sensor_data_count=DataReader.SENSOR_DATA_COUNT_IN_ROW,
                sensor_group_count=sensor_group_count
            )),
            ('features', SensorFeatureUnion(feature_transformers, n_jobs=n_jobs)),
        ])

    def __init__(self, n_jobs):
        self.transformer = SensorFeatureUnion([
            ('first', self.make_feature_transformer_pipeline(sensor_split_interval=10, n_jobs=n_jobs)),
            ('second', self.make_feature_transformer_pipeline(sensor_split_interval=1, n_jobs=n_jobs))
        ])

    @classmethod
    def clear_cache(cls):
        for path in (cls.TRAIN_FEATURES_CACHE_PATH,
                     cls.TEST_FEATURES_CACHE_PATH,
                     cls.FEATURE_NAMES_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def _cache_exists(cls):
        return os.path.exists(cls.TRAIN_FEATURES_CACHE_PATH) and\
               os.path.exists(cls.TEST_FEATURES_CACHE_PATH) and\
               os.path.exists(cls.FEATURE_NAMES_CACHE_PATH)

    @classmethod
    def _load_from_cache(cls):
        train_features = np.load(cls.TRAIN_FEATURES_CACHE_PATH)
        test_features = np.load(cls.TEST_FEATURES_CACHE_PATH)
        feature_names = np.load(cls.FEATURE_NAMES_CACHE_PATH)
        return train_features, test_features, feature_names

    @classmethod
    def _cache_arrays(cls, train_features, test_features, feature_names):
        np.save(cls.TRAIN_FEATURES_CACHE_PATH, train_features)
        np.save(cls.TEST_FEATURES_CACHE_PATH, test_features)
        np.save(cls.FEATURE_NAMES_CACHE_PATH, feature_names)

    def _transform_data_to_features(self):
        X_train_partials = []
        for X_train_partial in DataReader.iter_train_files_data():
            X_train_partials.append(X_train_partial)
        X_train = np.concatenate(X_train_partials, axis=0)
        rows = sum(partial.shape[0] for partial in X_train_partials)
        assert X_train.shape == (rows, DataReader.SENSOR_NUM * DataReader.SENSOR_DATA_COUNT_IN_ROW)
        train_features = self.transformer.transform(X_train)
        feature_names = np.asarray(self.transformer.get_feature_names())
        assert train_features.shape == (rows, len(feature_names))
        X_test = DataReader.read_test_data()
        assert X_test.shape == (X_test.shape[0], DataReader.SENSOR_NUM * DataReader.SENSOR_DATA_COUNT_IN_ROW)
        test_features = self.transformer.transform(X_test)
        assert test_features.shape == (test_features.shape[0], len(feature_names))
        return train_features, test_features, feature_names

    def load_features(self):
        if self._cache_exists():
            click.secho('Loading features from cache', fg='blue')
            train_features, test_features, feature_names = self._load_from_cache()
        else:
            click.secho('Transforming data to features', fg='blue')
            train_features, test_features, feature_names = self._transform_data_to_features()
            self._cache_arrays(train_features, test_features, feature_names)

        click.secho('Train features shape: {}'.format(train_features.shape))
        click.secho('Test features shape: {}'.format(test_features.shape))
        return train_features, test_features, feature_names
