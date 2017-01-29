# -*- coding: utf-8 -*-
import os

import click
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, FeatureUnion
from transformer import SensorsDataTransformer
from tsfresh.feature_extraction import feature_calculators
from reader import DataReader
from utils import measure_time


class FeatureExtractor:
    TRAIN_FEATURES_CACHE_PATH = os.path.expanduser('~/train_features_cache.npy')
    TEST_FEATURES_CACHE_PATH = os.path.expanduser('~/test_features_cache.npy')
    FEATURE_NAMES_CACHE_PATH = os.path.expanduser('~/feature_names_cache.npy')

    def __init__(self, n_jobs):
        self.transformer = Pipeline([
            ('features', FeatureUnion([
                ('max', SensorsDataTransformer(max)),
                #('last_location_of_maximum', SensorsDataTransformer(feature_calculators.last_location_of_maximum)),
                #('first_location_of_maximum', SensorsDataTransformer(feature_calculators.first_location_of_maximum)),
                ('min', SensorsDataTransformer(min)),
                #('last_location_of_minimum', SensorsDataTransformer(feature_calculators.last_location_of_minimum)),
                #('first_location_of_minimum', SensorsDataTransformer(feature_calculators.first_location_of_minimum)),
                ('mean', SensorsDataTransformer(feature_calculators.mean)),
                #('longest_strike_below_mean', SensorsDataTransformer(feature_calculators.longest_strike_below_mean)),
                #('longest_strike_above_mean', SensorsDataTransformer(feature_calculators.longest_strike_above_mean)),
                #('count_above_mean', SensorsDataTransformer(feature_calculators.count_above_mean)),
                #('count_below_mean', SensorsDataTransformer(feature_calculators.count_below_mean)),
                #('median', SensorsDataTransformer(feature_calculators.median)),
                ('variance', SensorsDataTransformer(feature_calculators.variance)),
                #('std', SensorsDataTransformer(feature_calculators.standard_deviation)),
                # ('large_std', SensorsDataTransformer(feature_calculators.large_standard_deviation)),
                #('var_larger_than_std', SensorsDataTransformer(feature_calculators.variance_larger_than_standard_deviation)),
                #('skewness', SensorsDataTransformer(feature_calculators.skewness)),
                #('kurtosis', SensorsDataTransformer(feature_calculators.kurtosis)),
                #('sum_values', SensorsDataTransformer(feature_calculators.sum_values)),
                #('mean_change', SensorsDataTransformer(feature_calculators.mean_change)),
                #('mean_abs_change', SensorsDataTransformer(feature_calculators.mean_abs_change)),
                #('absolute_sum_of_changes', SensorsDataTransformer(feature_calculators.absolute_sum_of_changes)),
                # ('mean_autocorrelation', SensorsDataTransformer(feature_calculators.mean_autocorrelation)),
                #('abs_energy', SensorsDataTransformer(feature_calculators.abs_energy)),
                # ('augmented_dickey_fuller', SensorsDataTransformer(feature_calculators.augmented_dickey_fuller)),
                #('mean_second_derivate_central', SensorsDataTransformer(feature_calculators.mean_second_derivate_central)),
                #('number_peaks', SensorsDataTransformer(feature_calculators.number_peaks, n=10)),
                # ('sample_entropy', SensorsDataTransformer(feature_calculators.sample_entropy)),
                # ('quantile', SensorsDataTransformer(feature_calculators.quantile)),
                # ('fft_coefficient', SensorsDataTransformer(feature_calculators.fft_coefficient)),
                # ('index_mass_quantile', SensorsDataTransformer(feature_calculators.index_mass_quantile)),
                # ('number_cwt_peaks', SensorsDataTransformer(feature_calculators.number_cwt_peaks)),
                # ('cwt_coefficients', SensorsDataTransformer(feature_calculators.cwt_coefficients)),
                # ('ar_coefficient', SensorsDataTransformer(feature_calculators.ar_coefficient)),
                # ('autocorrelation', SensorsDataTransformer(feature_calculators.autocorrelation)),
                # ('time_reversal_asymmetry_statistic', SensorsDataTransformer(feature_calculators.time_reversal_asymmetry_statistic)),
                # ('binned_entropy', SensorsDataTransformer(feature_calculators.binned_entropy)),
                ], n_jobs=n_jobs)
            )
        ])
        self.filter_transformer = VarianceThreshold(threshold=0.)  # remove features with Var == 0
        sensor_names = DataReader.get_sensor_names()
        self.transformer_names, _ = zip(*self.transformer.named_steps['features'].transformer_list)
        self.feature_names = np.array(['{}_{}'.format(transformer_name, sensor_name)
                                       for transformer_name in self.transformer_names
                                       for sensor_name in sensor_names])

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

    @measure_time(context='filtering')
    def _filter_features(self, train_features, test_features):
        train_features = self.filter_transformer.fit_transform(train_features)
        test_features = self.filter_transformer.transform(test_features)
        feature_names = self.feature_names[self.filter_transformer.get_support(indices=True)]
        removed_features_count = len(self.feature_names) - len(feature_names)
        click.secho('Removed {} const features'.format(removed_features_count), fg='red')
        return train_features, test_features, feature_names

    def load_features(self):
        if self._cache_exists():
            click.secho('Loading features from cache', fg='blue')
            train_features, test_features, feature_names = self._load_from_cache()
        else:
            click.secho('Transforming data to features', fg='blue')
            train_features, test_features = self._transform_data_to_features()
            train_features, test_features, feature_names = self._filter_features(train_features, test_features)
            self._cache_arrays(train_features, test_features, feature_names)

        assert train_features.shape[1] == test_features.shape[1] == len(feature_names)
        click.secho('Train features shape: {}'.format(train_features.shape))
        click.secho('Test features shape: {}'.format(test_features.shape))
        return train_features, test_features, feature_names
