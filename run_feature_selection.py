# -*- coding: utf-8 -*-
import json
import os
from collections import Counter

from uuid import uuid4

import numpy as np

import click
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from utils import timeit, RESULTS_PATH
from wrappers import gini_index_wrapper, mrmr_wrapper

MAX_FEATURES = 40


def class_to_binary(x):
    return 1 if x == 'warning' else 0


def load_data(clear_cache, n_jobs, test):
    if test:
        return generate_data()
    feature_extractor = FeatureExtractor(n_jobs)
    if clear_cache:
        feature_extractor.clear_cache()
    train_features, test_features, feature_names = feature_extractor.load_features()
    y_train = DataReader.read_training_labels()
    y_test = DataReader.read_test_labels()
    assert train_features.shape[0] == y_train.shape[0]
    assert test_features.shape[0] == y_test.shape[0]
    class_to_binary_vec = np.vectorize(class_to_binary)
    y_train = class_to_binary_vec(y_train)
    y_test = class_to_binary_vec(y_test)
    return train_features, y_train, test_features, y_test, feature_names


def generate_data():
    n_features = 100
    X, y = make_classification(
        n_samples=2000,
        n_features=n_features,
        n_informative=7,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False
    )
    y = np.asarray([y]).T
    return X, y, X, y, np.array(list('f_{}'.format(i) for i in range(n_features)))


def print_labels_summary(y_train, y_test):
    assert y_train.shape == y_test.shape
    for label_ix in range(y_train.shape[1]):
        y_train_label = y_train[:, label_ix]
        y_test_label = y_test[:, label_ix]
        print('Label: {} Train: {} Test: {}'.format(label_ix, Counter(y_train_label), Counter(y_test_label)))


def save_results(prefix, y_true, predictions, score_fun, feature_num,
                 label_ix, select_time, clasiff_time, selected_features=None):
    auc_score = roc_auc_score(y_true, predictions)
    result = {
        'AUC': round(auc_score, 6),
        'model': 'SVM',
        'classif_time': round(clasiff_time),
        'select_time': round(select_time),
        'feature_num': feature_num,
        'score_fun': score_fun,
        'label_ix': label_ix
    }
    if selected_features is not None:
        selected_features = sorted(selected_features)
        result['selected_features'] = selected_features
    output_file = os.path.join(RESULTS_PATH, '{prefix}_{hash}'.format(prefix=prefix, hash=str(uuid4())))
    with open(output_file, 'w') as output:
        json.dump(result, output)


def validate_ranking_selection(selection_transformer, train_features, y_train, test_features, y_test,
                               feature_names, label_ix):
    print('Started selection: {}'.format(str(selection_transformer)))
    selection_duration, _ = timeit(selection_transformer.fit, train_features, y_train)
    for feature_num in [5, 10, 20, 40]:
        selection_transformer.k = feature_num
        X_train = selection_transformer.transform(train_features)
        assert X_train.shape[1] == feature_num
        X_test = selection_transformer.transform(test_features)
        assert X_train.shape[1] == feature_num
        svc = SVC()
        classification_duration, _ = timeit(svc.fit, X_train, y_train)
        predictions = svc.predict(X_test)
        selected_features = feature_names[selection_transformer.get_support(indices=True)]
        save_results(
            prefix='ranking',
            y_true=y_test,
            predictions=predictions,
            score_fun=selection_transformer.score_func.__name__,
            feature_num=feature_num,
            label_ix=label_ix,
            select_time=selection_duration,
            clasiff_time=classification_duration,
            selected_features=selected_features
        )


def run_ranking_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names):
    ranking_selectors = [
        SelectKBest(f_classif, k=MAX_FEATURES),
        SelectKBest(mutual_info_classif, k=MAX_FEATURES),
        SelectKBest(gini_index_wrapper, k=MAX_FEATURES),
        SelectKBest(mrmr_wrapper, k=MAX_FEATURES)
    ]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(
        delayed(validate_ranking_selection)(selection_transformer, train_features, y_train[:, label_ix], test_features,
                                            y_test[:, label_ix], feature_names, label_ix)
        for selection_transformer in ranking_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_dimensionality_reduction(reduction_transformer_cls, train_features, y_train, test_features, y_test,
                                      label_ix):
    print('Started reduction: {}'.format(reduction_transformer_cls.__name__))
    for n_components in [5, 10, 20, 40]:
        reduction_transformer = reduction_transformer_cls(n_components=n_components)
        selection_duration, _ = timeit(reduction_transformer.fit, train_features, y_train)
        X_train = reduction_transformer.transform(train_features)
        assert X_train.shape[1] == n_components
        X_test = reduction_transformer.transform(test_features)
        assert X_train.shape[1] == n_components
        svc = SVC()
        classification_duration, _ = timeit(svc.fit, X_train, y_train)
        predictions = svc.predict(X_test)
        save_results(
            prefix='reduction',
            y_true=y_test,
            predictions=predictions,
            score_fun=reduction_transformer_cls.__name__,
            feature_num=n_components,
            label_ix=label_ix,
            select_time=selection_duration,
            clasiff_time=classification_duration,
        )


def run_dimensionality_reduction_methods(n_jobs, train_features, y_train, test_features, y_test):
    dimensionality_reduction_selectors = [
        PCA,
        GaussianRandomProjection,
        SparseRandomProjection
    ]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(
        delayed(validate_dimensionality_reduction)(selection_transformer, train_features, y_train[:, label_ix],
                                                   test_features, y_test[:, label_ix], label_ix)
        for selection_transformer in dimensionality_reduction_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_model_selectors(model_selector, train_features, y_train, test_features, y_test, feature_names, label_ix):
    print('Started model selection: {}'.format(str(model_selector)))
    selection_duration, _ = timeit(model_selector.fit, train_features, y_train)
    for feature_threshold in ['mean', '2 * mean', '3 * mean']:
        selector = SelectFromModel(model_selector, prefit=True, threshold=feature_threshold)
        X_train = selector.transform(train_features)
        X_test = selector.transform(test_features)
        svc = SVC()
        classification_duration, _ = timeit(svc.fit, X_train, y_train)
        predictions = svc.predict(X_test)
        selected_features = feature_names[selector.get_support(indices=True)]
        save_results(
            prefix='model',
            y_true=y_test,
            predictions=predictions,
            score_fun=model_selector.__class__.__name__,
            feature_num=len(selected_features),
            label_ix=label_ix,
            select_time=selection_duration,
            clasiff_time=classification_duration,
            selected_features=selected_features
        )


def run_model_selection_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names):
    model_selectors = [
        RandomForestClassifier(n_estimators=100),
        ExtraTreesClassifier(n_estimators=100, n_jobs=3)
    ]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(
        delayed(validate_model_selectors)(model_selector, train_features, y_train[:, label_ix],
                                          test_features, y_test[:, label_ix], feature_names, label_ix)
        for model_selector in model_selectors
        for label_ix in range(y_train.shape[1])
    )


def pre_filter(self, train_features, test_features):
    filter_transformer = VarianceThreshold(threshold=0.)  # remove features with Var == 0
    train_features = filter_transformer.fit_transform(train_features)
    test_features = filter_transformer.transform(test_features)
    feature_names = self.feature_names[self.filter_transformer.get_support(indices=True)]
    removed_features_count = len(self.feature_names) - len(feature_names)
    click.secho('Removed {} const features'.format(removed_features_count), fg='red')
    return train_features, test_features, feature_names


@click.command()
@click.option('--clear-cache', '-cc', is_flag=True, help='Clear features cache')
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
@click.option('--test', '-t', is_flag=True)
def main(clear_cache, n_jobs, test):
    train_features, y_train, test_features, y_test, feature_names = load_data(clear_cache, n_jobs, test)
    print_labels_summary(y_train, y_test)
    run_ranking_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names)
    run_dimensionality_reduction_methods(n_jobs, train_features, y_train, test_features, y_test)
    run_model_selection_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names)

if __name__ == '__main__':
    main()
