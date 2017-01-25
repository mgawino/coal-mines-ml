# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np

import click
from skfeature.function.statistical_based.gini_index import gini_index
from skfeature.function.statistical_based.CFS import cfs
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def create_pipeline(selection_transformer):
    return Pipeline([
        ('selection', selection_transformer),
        ('random_forest', RandomForestClassifier())
    ])


def make_selection_transformers():
    random_forest_clf = RandomForestClassifier(n_estimators=100)
    return [
        SelectKBest(f_classif, k='all'),
        SelectKBest(mutual_info_classif, k='all'),
        SelectKBest(gini_index, k='all'),
        # MRMR
        # Genetic algorithms
        SelectFromModel(random_forest_clf, threshold=0.5),
        PCA(n_components=1),
        GaussianRandomProjection(n_components=1),
        SparseRandomProjection(n_components=1)
    ]


def class_to_binary(iterable):
    return list(map(lambda x: 1 if x == 'warning' else 0, iterable))


def load_data(clear_cache, n_jobs):
    feature_extractor = FeatureExtractor(n_jobs)
    if clear_cache:
        feature_extractor.clear_cache()
    train_features, test_features, feature_names = feature_extractor.load_features()
    y_train = DataReader.read_training_labels()
    y_test = DataReader.read_test_labels()
    assert train_features.shape[0] == y_train.shape[0]
    assert test_features.shape[0] == y_test.shape[0]
    return train_features, class_to_binary(y_train), test_features, class_to_binary(y_test), feature_names


def generate_data():
    features = np.array([
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
    ])
    y = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ])
    names = ['a', 'b', 'c', 'd', 'e']
    return features, y, features, y, names


@click.command()
@click.option('--clear-cache', '-cc', is_flag=True, help='Clear features cache')
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
def main(clear_cache, n_jobs):
    train_features, y_train, test_features, y_test, feature_names = generate_data() # FIXME load_data(clear_cache, n_jobs)
    selection_transformers = make_selection_transformers()
    for label_ix in range(3):
        y_train_label = y_train[:, label_ix]
        y_test_label = y_test[:, label_ix]
        print('Train labels: {} Test labels: {}'.format(Counter(y_train_label), Counter(y_test_label)))
    for selection_transformer in selection_transformers:
        for label_ix in range(1):  # FIXME
            y_train_label = y_train[:, label_ix]
            y_test_label = y_test[:, label_ix]
            pipeline = create_pipeline(selection_transformer)
            pipeline.fit(train_features, y_train_label)
            # selected_features = feature_names[selection_transformer.get_support(indices=True)]
            predictions = pipeline.predict(test_features)
            auc_score = roc_auc_score(y_test_label, predictions)
            print('Selection method: {} AUC: {}'.format(selection_transformer, auc_score))


if __name__ == '__main__':
    main()
