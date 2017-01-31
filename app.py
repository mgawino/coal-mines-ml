# -*- coding: utf-8 -*-
import json
import os
from collections import Counter

import time
from uuid import uuid4

import numpy as np

import click
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from wrappers import gini_index_wrapper, mrmr_wrapper

PROJECT_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')

MAX_FEATURES = 40


def class_to_binary(x):
    return 1 if x == 'warning' else 0


def load_data(clear_cache, n_jobs):
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
    n_features = 20
    X, y = make_classification(
        n_samples=2000,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=123,
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


def validate_ranking_selection(selection_transformer, label_ix, train_features, y_train, test_features, y_test, feature_names):
    print('Started selection: {}'.format(str(selection_transformer)))
    start_time = time.process_time()
    y_train_label = y_train[:, label_ix]
    y_test_label = y_test[:, label_ix]
    selection_transformer.fit(train_features, y_train_label)
    for feature_num in [5, 10, 20, 40]:
        selection_transformer.k = feature_num
        X_train = selection_transformer.transform(train_features)
        assert X_train.shape
        X_test = selection_transformer.transform(test_features)

        selected_features = feature_names[selection_transformer.get_support(indices=True)]
    predictions = pipeline.predict(test_features)
    auc_score = roc_auc_score(y_test_label, predictions)
    results = {
        'auc_score': auc_score,
        'selected_features': sorted(selected_features),
        'time': time.process_time() - start_time,
        'method': str(selection_transformer),
        'label_ix': label_ix
    }
    with open(os.path.join(RESULTS_PATH, str(uuid4())), 'w') as output:
        json.dump(results, output)


def run_ranking_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names):
    ranking_selectors = [
        SelectKBest(f_classif, k=MAX_FEATURES),
        SelectKBest(mutual_info_classif, k=MAX_FEATURES),
        SelectKBest(gini_index_wrapper, k=MAX_FEATURES),
        SelectKBest(mrmr_wrapper, k=MAX_FEATURES)
    ]
    Parallel(n_jobs=n_jobs, verbose=100, pre_dispatch='n_jobs')(
        delayed(validate_ranking_selection)(selection_transformer, label_ix, train_features, y_train, test_features,
                                            y_test, feature_names)
        for selection_transformer in ranking_selectors
        for label_ix in range(y_train.shape[1])
    )


def run_dimensionality_reduction_methods(n_jobs, train_features, y_train, test_features, y_test):
    dimensionality_reduction_selectors = [
        PCA(n_components=2),
        GaussianRandomProjection(n_components=10),
        SparseRandomProjection(n_components=10)
    ]


def run_model_selection_methods():
    random_forest_clf = RandomForestClassifier(n_estimators=100)
    # TODO: ExtraTrees
    SelectFromModel(random_forest_clf)


@click.command()
@click.option('--clear-cache', '-cc', is_flag=True, help='Clear features cache')
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
def main(clear_cache, n_jobs):
    train_features, y_train, test_features, y_test, feature_names = generate_data()  # FIXME load_data(clear_cache, n_jobs)
    print_labels_summary(y_train, y_test)
    run_ranking_methods(n_jobs, train_features, y_train, test_features, y_test, feature_names)


if __name__ == '__main__':
    main()
