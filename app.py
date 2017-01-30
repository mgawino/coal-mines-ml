# -*- coding: utf-8 -*-
import json
import os
from collections import Counter

import time
from uuid import uuid4

import numpy as np

import click
from joblib import Parallel, delayed
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.statistical_based.gini_index import gini_index
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from wrappers import gini_index_wrapper, mrmr_wrapper

PROJECT_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')


def make_selection_transformers():
    random_forest_clf = RandomForestClassifier(n_estimators=100)
    return [
        SelectKBest(f_classif, k=2),
        SelectKBest(mutual_info_classif, k=2),
        SelectKBest(gini_index_wrapper, k=2),
        SelectKBest(mrmr_wrapper, k=2),
        PCA(n_components=2),
        SelectFromModel(random_forest_clf)
        # GaussianRandomProjection(n_components='auto'),
        # SparseRandomProjection(n_components='auto')
    ]


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


def validate_selection(selection_transformer, label_ix, train_features,  y_train, test_features, y_test, feature_names):
    print('Started selection: {}'.format(str(selection_transformer)))
    start_time = time.process_time()
    y_train_label = y_train[:, label_ix]
    y_test_label = y_test[:, label_ix]
    pipeline = Pipeline([
        ('selection', selection_transformer),
        ('random_forest', SVC())
    ])
    pipeline.fit(train_features, y_train_label)
    selected_features = []
    if hasattr(selection_transformer, 'get_support'):
        selected_features = feature_names[selection_transformer.get_support(indices=True)]
    predictions = pipeline.predict(test_features)
    auc_score = roc_auc_score(y_test_label, predictions)
    results = {
        'auc_score': auc_score,
        'selected_features': sorted(selected_features),
        'time': time.process_time() - start_time,
        'method': str(selection_transformer)
    }
    with open(os.path.join(RESULTS_PATH, str(uuid4())), 'w') as output:
        json.dump(results, output)


@click.command()
@click.option('--clear-cache', '-cc', is_flag=True, help='Clear features cache')
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
def main(clear_cache, n_jobs):
    train_features, y_train, test_features, y_test, feature_names = generate_data()  # FIXME load_data(clear_cache, n_jobs)
    print('Finished generating data...')
    selection_transformers = make_selection_transformers()
    print_labels_summary(y_train, y_test)
    Parallel(n_jobs=n_jobs, verbose=100, pre_dispatch='n_jobs')(
        delayed(validate_selection)(selection_transformer, 0, train_features, y_train, test_features, y_test, feature_names)
        for selection_transformer in selection_transformers
    )


if __name__ == '__main__':
    main()
