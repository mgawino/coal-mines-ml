# -*- coding: utf-8 -*-
import json
import os
import pickle
from collections import Counter

from uuid import uuid4

import itertools

import time

import numpy as np

import click
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC
import scipy
from sklearn.tree import DecisionTreeClassifier
from utils import timeit, RESULTS_PATH, MODEL_CACHE_PATH
from wrappers import gini_index_wrapper, corr_wrapper, mrmr


def class_to_binary(x):
    return 1 if x == 'warning' else 0


def load_data(clear_cache, n_jobs, test):
    if test:
        return generate_data()
    feature_extractor = FeatureExtractor(n_jobs)
    if clear_cache and click.confirm('Are you sure you want to clear cache?'):
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
        n_samples=20,
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
    svc_rbf_params = {
        'C': scipy.stats.expon(scale=400),
        'gamma': scipy.stats.expon(scale=.1),
    }
    svc_params = {
        'C': scipy.stats.expon(scale=400)
    }
    return [
        SVC(cache_size=500, kernel='rbf', class_weight='balanced'),
        LinearSVC(class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced'),
        LogisticRegression(class_weight='balanced')
    ]


def make_ranking_selectors():
    return [
        SelectKBest(gini_index_wrapper),
        SelectKBest(corr_wrapper),
        SelectKBest(f_classif),
        SelectKBest(mutual_info_classif)
    ]


def make_model_selectors():
    return [
        RandomForestClassifier(n_estimators=700, class_weight='balanced'),
        ExtraTreesClassifier(n_estimators=700, class_weight='balanced')
    ]


def _model_to_cache_path(model, label_ix):
    if isinstance(model, SelectKBest):
        model_file_name = '{}_{}'.format(label_ix, model.score_func.__name__)
    elif isinstance(model, RandomizedSearchCV):
        model_file_name = '{}_{}'.format(label_ix, model.estimator.__class__.__name__)
    else:
        model_file_name = '{}_{}'.format(label_ix, model.__class__.__name__)
    return os.path.join(MODEL_CACHE_PATH, model_file_name)


def load_from_cache(model_file_path):
    assert os.path.exists(model_file_path)
    print('Loaded model from cache: {}'.format(os.path.basename(model_file_path)))
    with open(model_file_path, 'rb') as model_file:
        model, duration = pickle.load(model_file)
        return model, duration


def fit_or_load_from_cache(model, X_train, y_train, label_ix):
    model_file_path = _model_to_cache_path(model, label_ix)
    if os.path.exists(model_file_path):
        model, duration = load_from_cache(model_file_path)
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
    for feature_num in range(1, 20):
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


def iter_ranking_methods(train_features, y_train, test_features, y_test, feature_names):
    ranking_selectors = make_ranking_selectors()
    yield from (
        delayed(validate_ranking_selection)(selection_transformer, train_features, y_train[:, label_ix], test_features,
                                            y_test[:, label_ix], feature_names, label_ix)
        for selection_transformer in ranking_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_dimensionality_reduction(reduction_transformer_cls, train_features, y_train, test_features, y_test,
                                      label_ix):
    print('Started reduction: {} on label: {}'.format(reduction_transformer_cls.__name__, label_ix))
    for n_components in [10, 15, 20]:
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


def iter_dimensionality_reduction_methods(train_features, y_train, test_features, y_test, _):
    dimensionality_reduction_selectors = [
        PCA,
        GaussianRandomProjection,
    ]
    yield from (
        delayed(validate_dimensionality_reduction)(selection_transformer, train_features, y_train[:, label_ix],
                                                   test_features, y_test[:, label_ix], label_ix)
        for selection_transformer in dimensionality_reduction_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_model_selectors(model_selector, train_features, y_train, test_features, y_test, feature_names, label_ix):
    model_selector, selection_duration = fit_or_load_from_cache(model_selector, train_features, y_train, label_ix)
    print('Started model selection: {} on label: {}'.format(str(model_selector.__class__.__name__), label_ix))
    sorted_feature_importances = sorted(model_selector.feature_importances_, reverse=True)
    for feature_num in range(1, 20):
        selector = SelectFromModel(
            model_selector,
            prefit=True,
            threshold=sorted_feature_importances[feature_num]
        )
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


def iter_model_selection_methods(train_features, y_train, test_features, y_test, feature_names):
    model_selectors = make_model_selectors()
    yield from (
        delayed(validate_model_selectors)(model_selector, train_features, y_train[:, label_ix],
                                          test_features, y_test[:, label_ix], feature_names, label_ix)
        for model_selector in model_selectors
        for label_ix in range(y_train.shape[1])
    )


def validate_mrmr_selector(selection_transformer, train_features, y_train, test_features, y_test,
                           feature_names, label_ix):
    print('Started MRMR on label: {} model: {}'.format(label_ix, selection_transformer))
    assert len(selection_transformer.scores_) == train_features.shape[1]
    start = time.process_time()
    selected_feature_indices = mrmr(
        train_features,
        y_train,
        selection_transformer.scores_,
        max_features=20
    )
    selection_duration = time.process_time() - start
    X_train = train_features[:, selected_feature_indices]
    X_test = test_features[:, selected_feature_indices]
    assert X_train.shape[1] == X_test.shape[1]
    selected_features = feature_names[selected_feature_indices]
    for classifier in make_classifiers():
        classification_duration = timeit(classifier.fit, X_train, y_train)
        predictions = classifier.predict(X_test)
        save_results(
            prefix='ranking',
            classifier=classifier,
            y_true=y_test,
            predictions=predictions,
            score_fun='MRMR_{}'.format(selection_transformer.score_func.__name__),
            feature_num=len(selected_feature_indices),
            label_ix=label_ix,
            select_time=selection_duration,
            clasiff_time=classification_duration,
            selected_features=selected_features
        )


def iter_mrmr_methods(train_features, y_train, test_features, y_test, feature_names):
    selectors = make_ranking_selectors()
    yield from (
        delayed(validate_mrmr_selector)(load_from_cache(_model_to_cache_path(selector, label_ix))[0],
                                        train_features, y_train[:, label_ix], test_features,
                                        y_test[:, label_ix], feature_names, label_ix)
        for selector in selectors
        for label_ix in range(y_train.shape[1])
    )


def pre_filter(train_features, test_features, feature_names):
    print('Pre filtering data...')
    pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.))
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
@click.option('--clear-cache', '-cc', is_flag=True)
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
@click.option('--test', '-t', is_flag=True)
def main(clear_cache, n_jobs, test):
    train_features, y_train, test_features, y_test, feature_names = load_data(clear_cache, n_jobs, test)
    print_labels_summary(y_train, y_test)
    train_features, test_features, feature_names = pre_filter(train_features, test_features, feature_names)
    methods = [
        # iter_ranking_methods,
        # iter_model_selection_methods,
        # iter_dimensionality_reduction_methods
        iter_mrmr_methods
    ]
    jobs = [method(train_features, y_train, test_features, y_test, feature_names) for method in methods]
    Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='n_jobs')(itertools.chain(*jobs))

if __name__ == '__main__':
    main()
