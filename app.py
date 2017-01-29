# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np

import click
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.statistical_based.gini_index import gini_index
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from wrappers import SelectKBestWrapper, cfs_wrapper


def create_pipeline(selection_transformer):
    return Pipeline([
        ('selection', selection_transformer),
        ('random_forest', RandomForestClassifier())
    ])


def make_selection_transformers():
    random_forest_clf = RandomForestClassifier(n_estimators=100)
    return [
        # SelectKBest(f_classif, k=5),
        # SelectKBest(mutual_info_classif, k=5),
        SelectKBest(gini_index, k=5),
        # SelectKBestWrapper(cfs_wrapper, k=5),
        # SelectKBestWrapper(mrmr, k=5),
        # Genetic algorithms
        # SelectFromModel(random_forest_clf, threshold=0.5),
        # PCA(n_components=1),
        # GaussianRandomProjection(n_components=1),
        # SparseRandomProjection(n_components=1)
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


@click.command()
@click.option('--clear-cache', '-cc', is_flag=True, help='Clear features cache')
@click.option('--n-jobs', '-j', type=click.INT, help='Feature extraction jobs', default='3')
def main(clear_cache, n_jobs):
    train_features, y_train, test_features, y_test, feature_names = load_data(clear_cache, n_jobs)
    selection_transformers = make_selection_transformers()
    for label_ix in range(3):
        y_train_label = y_train[:, label_ix]
        y_test_label = y_test[:, label_ix]
        print('Label: {} Train: {} Test: {}'.format(label_ix, Counter(y_train_label), Counter(y_test_label)))
    for selection_transformer in selection_transformers:
        for label_ix in range(3):
            y_train_label = y_train[:, label_ix]
            y_test_label = y_test[:, label_ix]
            pipeline = create_pipeline(selection_transformer)
            pipeline.fit(train_features, y_train_label)
            # selected_features = feature_names[selection_transformer.get_support(indices=True)]
            predictions = pipeline.predict(test_features)
            auc_score = roc_auc_score(y_test_label, predictions)
            print('Selection method: {} for label: {} AUC: {}'.format(selection_transformer, label_ix, auc_score))


if __name__ == '__main__':
    main()
