# -*- coding: utf-8 -*-
import os

import click
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from transformer import SensorsTransformer
from tsfresh.feature_extraction import feature_calculators
from reader import DataReader


class FeatureExtractor:
    TRAIN_FEATURES_CACHE_PATH = os.path.expanduser('~/train_features_cache.npy')
    TEST_FEATURES_CACHE_PATH = os.path.expanduser('~/test_features_cache.npy')
    FEATURE_NAMES_CACHE_PATH = os.path.expanduser('~/feature_names_cache.npy')

    def __init__(self, n_jobs):
        self.transformer = Pipeline([
            ('features', FeatureUnion([
                # ('number_peaks_5', SensorsDataTransformer(feature_calculators.number_peaks, n=5)),  # 640
                # ('number_peaks_10', SensorsDataTransformer(feature_calculators.number_peaks, n=10)),  # 870
                # ('number_peaks_20', SensorsDataTransformer(feature_calculators.number_peaks, n=20)),  # 1200
                ('count_above_mean', SensorsTransformer(feature_calculators.count_above_mean)),  # 500
                ('count_below_mean', SensorsTransformer(feature_calculators.count_below_mean)),  # 500
                ('autocorrelation_10', SensorsTransformer(feature_calculators.autocorrelation, lag=10)),  # 200
                ('autocorrelation_20', SensorsTransformer(feature_calculators.autocorrelation, lag=20)),  # 200
                ('autocorrelation_50', SensorsTransformer(feature_calculators.autocorrelation, lag=50)),  # 200
                ('max', SensorsTransformer(max)),  # 200
                ('min', SensorsTransformer(min)),  # 200
                ('mean_autocorrelation', SensorsTransformer(feature_calculators.mean_autocorrelation)),  # 117
                ('quantile_0.2', SensorsTransformer(feature_calculators.quantile, q=0.2)),  # 130
                ('quantile_0.4', SensorsTransformer(feature_calculators.quantile, q=0.4)),  # 130
                ('quantile_0.6', SensorsTransformer(feature_calculators.quantile, q=0.6)),  # 130
                ('quantile_0.8', SensorsTransformer(feature_calculators.quantile, q=0.8)),  # 130
                ('skewness', SensorsTransformer(feature_calculators.skewness)),  # 113
                ('kurtosis', SensorsTransformer(feature_calculators.kurtosis)),  # 120
                ('longest_strike_below_mean', SensorsTransformer(feature_calculators.longest_strike_below_mean)),
                ('longest_strike_above_mean', SensorsTransformer(feature_calculators.longest_strike_above_mean)),
                ('last_location_of_maximum', SensorsTransformer(feature_calculators.last_location_of_maximum)),
                ('first_location_of_maximum', SensorsTransformer(feature_calculators.first_location_of_maximum)),
                ('last_location_of_minimum', SensorsTransformer(feature_calculators.last_location_of_minimum)),
                ('first_location_of_minimum', SensorsTransformer(feature_calculators.first_location_of_minimum)),
                ('binned_entropy_5', SensorsTransformer(feature_calculators.binned_entropy, max_bins=5)),
                ('binned_entropy_10', SensorsTransformer(feature_calculators.binned_entropy, max_bins=10)),
                ('binned_entropy_20', SensorsTransformer(feature_calculators.binned_entropy, max_bins=20)),
                ('mean_second_derivate_central', SensorsTransformer(feature_calculators.mean_second_derivate_central)),
                ('mean', SensorsTransformer(feature_calculators.mean)),
                ('median', SensorsTransformer(feature_calculators.median)),
                ('variance', SensorsTransformer(feature_calculators.variance)),
                ('std', SensorsTransformer(feature_calculators.standard_deviation)),
                ('large_std', SensorsTransformer(feature_calculators.large_standard_deviation, r=0.5)),
                ('var_larger_than_std', SensorsTransformer(feature_calculators.variance_larger_than_standard_deviation)),
                ('sum_values', SensorsTransformer(feature_calculators.sum_values)),
                ('mean_change', SensorsTransformer(feature_calculators.mean_change)),
                ('mean_abs_change', SensorsTransformer(feature_calculators.mean_abs_change)),
                ('absolute_sum_of_changes', SensorsTransformer(feature_calculators.absolute_sum_of_changes)),
                ('abs_energy', SensorsTransformer(feature_calculators.abs_energy)),
                # ('fft_coefficient', SensorsDataTransformer(feature_calculators.fft_coefficient)),
                # ('index_mass_quantile', SensorsDataTransformer(feature_calculators.index_mass_quantile)),
                # ('number_cwt_peaks', SensorsDataTransformer(feature_calculators.number_cwt_peaks)),
                # ('cwt_coefficients', SensorsDataTransformer(feature_calculators.cwt_coefficients)),
                # ('ar_coefficient', SensorsDataTransformer(feature_calculators.ar_coefficient)),
                # ('time_reversal_asymmetry_statistic', SensorsDataTransformer(feature_calculators.time_reversal_asymmetry_statistic)),
                ], n_jobs=n_jobs)
            )
        ])
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
        X_train_partials = []
        for X_train_partial in DataReader.iter_train_files_data():
            X_train_partials.append(X_train_partial)
        X_train = np.concatenate(X_train_partials, axis=0)
        train_features = self.transformer.transform(X_train)
        assert train_features.shape[1] == len(self.transformer_names) * DataReader.SENSOR_NUM
        X_test = DataReader.read_test_data()
        test_features = self.transformer.transform(X_test)
        return train_features, test_features

    def load_features(self):
        if self._cache_exists():
            click.secho('Loading features from cache', fg='blue')
            train_features, test_features, feature_names = self._load_from_cache()
        else:
            click.secho('Transforming data to features', fg='blue')
            train_features, test_features = self._transform_data_to_features()
            feature_names = self.feature_names
            self._cache_arrays(train_features, test_features, feature_names)

        assert train_features.shape[1] == test_features.shape[1] == len(feature_names)
        click.secho('Train features shape: {}'.format(train_features.shape))
        click.secho('Test features shape: {}'.format(test_features.shape))
        return train_features, test_features, feature_names
