# -*- coding: utf-8 -*-
import os

import click
from sklearn.pipeline import FeatureUnion
from transformer import SensorTransformer, SensorMultiTransformer
from tsfresh.feature_extraction.feature_calculators import *
from reader import DataReader


class CustomFeatureUnion(FeatureUnion):

    def get_feature_names(self):
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not "
                                     "provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            feature_names.extend(trans.get_feature_names())
        return feature_names


class FeatureExtractor:
    TRAIN_FEATURES_CACHE_PATH = os.path.expanduser('~/train_features_cache.npy')
    TEST_FEATURES_CACHE_PATH = os.path.expanduser('~/test_features_cache.npy')
    FEATURE_NAMES_CACHE_PATH = os.path.expanduser('~/feature_names_cache.npy')

    def __init__(self, n_jobs):
        feature_transformers = [
            ('max', SensorTransformer(max)),
            ('min', SensorTransformer(min)),
            ('mean_autocorrelation', SensorTransformer(mean_autocorrelation)),
            ('skewness', SensorTransformer(skewness)),
            ('kurtosis', SensorTransformer(kurtosis)),
            ('longest_strike_below_mean', SensorTransformer(longest_strike_below_mean)),
            ('longest_strike_above_mean', SensorTransformer(longest_strike_above_mean)),
            ('last_location_of_maximum', SensorTransformer(last_location_of_maximum)),
            ('first_location_of_maximum', SensorTransformer(first_location_of_maximum)),
            ('last_location_of_minimum', SensorTransformer(last_location_of_minimum)),
            ('first_location_of_minimum', SensorTransformer(first_location_of_minimum)),
            ('binned_entropy_10', SensorTransformer(binned_entropy, max_bins=10)),
            ('mean_second_derivate_central', SensorTransformer(mean_second_derivate_central)),
            ('mean', SensorTransformer(mean)),
            ('median', SensorTransformer(median)),
            ('variance', SensorTransformer(variance)),
            ('std', SensorTransformer(standard_deviation)),
            ('var_larger_than_std', SensorTransformer(variance_larger_than_standard_deviation)),
            ('sum_values', SensorTransformer(sum_values)),
            ('mean_change', SensorTransformer(mean_change)),
            ('mean_abs_change', SensorTransformer(mean_abs_change)),
            ('absolute_sum_of_changes', SensorTransformer(absolute_sum_of_changes)),
            ('abs_energy', SensorTransformer(abs_energy)),
        ]
        for n in [1, 3, 5]:
            feature_transformers.append(('number_peaks_{}'.format(n), SensorTransformer(number_peaks, n=n)))
            feature_transformers.append(('large_num_of_peaks_{}'.format(n), SensorTransformer(large_number_of_peaks, n=n)))
            feature_transformers.append(('cwt_peaks_{}'.format(n), SensorTransformer(number_cwt_peaks, n=n)))

        for lag in range(1, 4):
            transformer = SensorTransformer(time_reversal_asymmetry_statistic, lag=lag)
            feature_transformers.append(('time_reversal_{}'.format(lag), transformer))

        for i in range(20):
            r = 0.05 * i
            feature_transformers.append(('symmetry_looking_{}'.format(r), SensorTransformer(symmetry_looking, r=r)))

        for i in range(10):
            r = 0.05 * i
            feature_transformers.append(('large_std_{}'.format(r), SensorTransformer(large_standard_deviation, r=r)))
            feature_transformers.append(('autocorrelation_{}'.format(i), SensorTransformer(autocorrelation, lag=i)))

        quantiles = [.1, .2, .3, .4, .6, .7, .8, .9]
        for q in quantiles:
            feature_transformers.append(('quantile_{}'.format(q), SensorTransformer(quantile, q=q)))

        feature_transformers.append(
            ('index_mass_quantile', SensorMultiTransformer(index_mass_quantile, param=[{'q': q} for q in quantiles]))
        )

        for ql, qh in itertools.product([0., .2, .4, .6, .8], [.2, .4, .6, .8, 1.]):
            feature_transformers.append(
                ('mean_abs_change_quantile_{}_{}'.format(ql, qh),
                 SensorTransformer(mean_abs_change_quantiles, ql=ql, qh=qh))
            )

        for r in [.1, .3, .5, .7, .9]:
            feature_transformers.append(
                ('approximate_entropy_{}'.format(r), SensorTransformer(approximate_entropy, m=2, r=r))
            )

        feature_transformers.append(
            ('ar_coefficient',
             SensorMultiTransformer(ar_coefficient, param=[{'coeff': coeff, 'k': 10} for coeff in range(5)]))
        )

        feature_transformers.append(
            ('cwt_coeff',
             SensorMultiTransformer(
                 cwt_coefficients,
                 param=[{'coeff': coeff, 'widths': (2, 5, 10, 20), 'w': w} for coeff in range(15) for w in (2, 5, 10, 20)]
             ))
        )

        feature_transformers.append(
            ('spkt_welch_density',
             SensorMultiTransformer(spkt_welch_density, param=[{'coeff': coeff} for coeff in [2, 5, 8]]))
        )

        feature_transformers.append(
            ('fft_coefficent',
             SensorMultiTransformer(fft_coefficient, param=[{'coeff': coeff} for coeff in range(10)]))
        )

        self.transformer = CustomFeatureUnion(feature_transformers, n_jobs=n_jobs)
        sensor_names = DataReader.get_sensor_names()
        for _, trans in self.transformer.transformer_list:
            trans.set_sensor_names(sensor_names)

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
        feature_names = np.asarray(self.transformer.get_feature_names())
        assert train_features.shape[1] == len(feature_names)
        X_test = DataReader.read_test_data()
        test_features = self.transformer.transform(X_test)
        return train_features, test_features, feature_names

    def load_features(self):
        if self._cache_exists():
            click.secho('Loading features from cache', fg='blue')
            train_features, test_features, feature_names = self._load_from_cache()
        else:
            click.secho('Transforming data to features', fg='blue')
            train_features, test_features, feature_names = self._transform_data_to_features()
            self._cache_arrays(train_features, test_features, feature_names)

        assert train_features.shape[1] == test_features.shape[1] == len(feature_names)
        click.secho('Train features shape: {}'.format(train_features.shape))
        click.secho('Test features shape: {}'.format(test_features.shape))
        return train_features, test_features, feature_names
