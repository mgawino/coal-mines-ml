# -*- coding: utf-8 -*-
import json
import os
import pickle
from collections import Counter

from uuid import uuid4

import numpy as np

import click
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import scipy
from utils import timeit, RESULTS_PATH, MODEL_CACHE_PATH
from wrappers import gini_index_wrapper


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
    assert train_features.shape[0] == y_train.shape[0], (train_features.shape, y_train.shape)
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
    for label_ix in range(y_train.shape[1]):
        y_train_label = y_train[:, label_ix]
        y_test_label = y_test[:, label_ix]
        print('Label: {} Train: {} Test: {}'.format(label_ix, Counter(y_train_label), Counter(y_test_label)))


def save_results(prefix, classifier, y_true, predictions, score_fun, feature_num,
                 label_ix, select_time, clasiff_time, selected_features=None):
    auc_score = roc_auc_score(y_true, predictions)
    if hasattr(classifier, 'estimator'):
        result = {
          'model': classifier.estimator.__class__.__name__,
          'best_params': classifier.best_params_
        }
    else:
        result = {
            'model': classifier.__class__.__name__
        }
    result.update({
        'AUC': round(auc_score, 6),
        'c_time': round(clasiff_time),
        's_time': round(select_time),
        'f_num': feature_num,
        'score_fun': score_fun,
        'label': label_ix
    })
    if selected_features is not None:
        selected_features = sorted(selected_features)
        result['selected_features'] = selected_features
    output_file = os.path.join(RESULTS_PATH, '{prefix}_{hash}'.format(prefix=prefix, hash=str(uuid4())))
    with open(output_file, 'w') as output:
        json.dump(result, output)


def make_classifiers():
    svc_params = {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=.1),
        'kernel': ['rbf']
    }
    return [
        RandomizedSearchCV(SVC(cache_size=500), svc_params, cv=2, n_iter=10),
        RandomForestClassifier(n_estimators=500)
    ]


def fit_or_load_from_cache(model, X_train, y_train, label_ix):
    if isinstance(model, SelectKBest):
        model_file_name = '{}_{}'.format(label_ix, model.score_func.__name__)
    else:
        model_file_name = '{}_{}'.format(label_ix, model.__class__.__name__)
    model_file_path = os.path.join(MODEL_CACHE_PATH, model_file_name)
    if os.path.exists(model_file_path):
        print('Loaded model from cache: {}'.format(model_file_name))
        with open(model_file_path, 'rb') as model_file:
            model, duration = pickle.load(model_file)
        return model, duration
    duration = timeit(model.fit, X_train, y_train)
    with open(model_file_path, 'wb') as model_file:
        pickle.dump((model, duration), model_file)
    return model, duration


def validate_ranking_selection(selection_transformer, train_features, y_train, test_features, y_test,
                               feature_names, label_ix):
    print('Started selection: {} on label: {}'.format(str(selection_transformer), label_ix))
    selection_transformer, selection_duration = fit_or_load_from_cache(
        selection_transformer,
        train_features,
        y_train,
        label_ix
    )
    assert len(selection_transformer.scores_) == train_features.shape[1]
    for feature_num in [3, 5, 7, 9, 11, 14]:
        selection_transformer.k = feature_num
        X_train = selection_transformer.transform(train_features)
        assert X_train.shape[1] == feature_num
        X_test = selection_transformer.transform(test_features)
        assert X_train.shape[1] == feature_num
        selected_features = feature_names[selection_transformer.get_support(indices=True)]
        for classifier in make_classifiers():
            classification_duration = timeit(classifier.fit, X_train, y_train)
            predictions = classifier.predict(X_test)
            save_results(
                prefix='ranking',
                classifier=classifier,
                y_true=y_test,
                predictions=predictions,
                score_fun=selection_transformer.score_func.__name__,
                feature_num=feature_num,
                label_ix=label_ix,
                select_time=selection_duration,
                clasiff_time=classification_duration,
                selected_features=selected_features
            )
    print('Finished selection: {} on label: {}'.format(str(selection_transformer), label_ix))


def run_ranking_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names):
    ranking_selectors = [
        SelectKBest(gini_index_wrapper),
        SelectKBest(f_classif),
        SelectKBest(mutual_info_classif),
    ]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(
        delayed(validate_ranking_selection)(selection_transformer, train_features, y_train[:, label_ix], test_features,
                                            y_test[:, label_ix], feature_names, label_ix)
        for selection_transformer in ranking_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_dimensionality_reduction(reduction_transformer_cls, train_features, y_train, test_features, y_test,
                                      label_ix):
    print('Started reduction: {} on label: {}'.format(reduction_transformer_cls.__name__, label_ix))
    for n_components in [5, 10, 20, 40]:
        reduction_transformer = reduction_transformer_cls(n_components=n_components)
        selection_duration = timeit(reduction_transformer.fit, train_features, y_train)
        X_train = reduction_transformer.transform(train_features)
        assert X_train.shape[1] == n_components
        X_test = reduction_transformer.transform(test_features)
        assert X_train.shape[1] == n_components
        for classifier in make_classifiers():
            classification_duration = timeit(classifier.fit, X_train, y_train)
            predictions = classifier.predict(X_test)
            save_results(
                prefix='reduction',
                classifier=classifier,
                y_true=y_test,
                predictions=predictions,
                score_fun=reduction_transformer_cls.__name__,
                feature_num=n_components,
                label_ix=label_ix,
                select_time=selection_duration,
                clasiff_time=classification_duration,
            )
    print('Finished reduction: {} on label: {}'.format(str(reduction_transformer_cls.__name__), label_ix))


def run_dimensionality_reduction_methods(n_jobs, train_features, y_train, test_features, y_test):
    dimensionality_reduction_selectors = [
        PCA,
        GaussianRandomProjection,
    ]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(
        delayed(validate_dimensionality_reduction)(selection_transformer, train_features, y_train[:, label_ix],
                                                   test_features, y_test[:, label_ix], label_ix)
        for selection_transformer in dimensionality_reduction_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_model_selectors(model_selector, train_features, y_train, test_features, y_test, feature_names, label_ix):
    print('Started model selection: {} on label: {}'.format(str(model_selector.__class__.__name__), label_ix))
    model_selector, selection_duration = fit_or_load_from_cache(model_selector, train_features, y_train, label_ix)
    for feature_threshold in ['mean']:
        selector = SelectFromModel(model_selector, prefit=True, threshold=feature_threshold)
        X_train = selector.transform(train_features)
        X_test = selector.transform(test_features)
        selected_features = feature_names[selector.get_support(indices=True)]
        for classifier in make_classifiers():
            classification_duration = timeit(classifier.fit, X_train, y_train)
            predictions = classifier.predict(X_test)
            save_results(
                prefix='model',
                classifier=classifier,
                y_true=y_test,
                predictions=predictions,
                score_fun=model_selector.__class__.__name__,
                feature_num=len(selected_features),
                label_ix=label_ix,
                select_time=selection_duration,
                clasiff_time=classification_duration,
                selected_features=selected_features
            )
    print('Finished model selection: {} on label: {}'.format(str(model_selector), label_ix))


def run_model_selection_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names):
    model_selectors = [
        RandomForestClassifier(n_estimators=500),
        ExtraTreesClassifier(n_estimators=500)
    ]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(
        delayed(validate_model_selectors)(model_selector, train_features, y_train[:, label_ix],
                                          test_features, y_test[:, label_ix], feature_names, label_ix)
        for model_selector in model_selectors
        for label_ix in range(y_train.shape[1])
    )


def pre_filter(train_features, test_features, feature_names):
    print('Pre filtering data...')
    pipeline = Pipeline([
        ('imputer', Imputer()),
        ('variance_threshold', VarianceThreshold(threshold=0.)),
        ('scaler', StandardScaler())
    ])
    train_features = pipeline.fit_transform(train_features)
    test_features = pipeline.transform(test_features)
    features_count_before = len(feature_names)
    feature_names = feature_names[pipeline.named_steps['variance_threshold'].get_support(indices=True)]
    removed_features_count = features_count_before - len(feature_names)
    assert train_features.shape[1] == test_features.shape[1] == len(feature_names)
    click.secho('Removed {} features'.format(removed_features_count), fg='red')
    return train_features, test_features, feature_names


@click.command()
@click.option('--clear-cache', '-cc', is_flag=True, help='Clear features cache')
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
@click.option('--test', '-t', is_flag=True)
def main(clear_cache, n_jobs, test):
    train_features, y_train, test_features, y_test, feature_names = load_data(clear_cache, n_jobs, test)
    print_labels_summary(y_train, y_test)
    train_features, test_features, feature_names = pre_filter(train_features, test_features, feature_names)
    run_ranking_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names)
    run_model_selection_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names)
    run_dimensionality_reduction_methods(n_jobs, train_features, y_train, test_features, y_test)

if __name__ == '__main__':
    main()
