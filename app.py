# -*- coding: utf-8 -*-
from collections import Counter

from reader import DataReader, FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def create_pipline(selection_transformer):
    return Pipeline([
        ('selection', selection_transformer),
        ('random_forest', RandomForestClassifier())
    ])


def make_selection_transformers():
    return [SelectKBest(chi2, k=10)]


def class_to_binary(iterable):
    return list(map(lambda x: 1 if x == 'warning' else 0, iterable))


def load_data():
    feature_extractor = FeatureExtractor()
    # feature_extractor.clear_cache()
    train_features, test_features = feature_extractor.load_features()
    feature_names = feature_extractor.get_feature_names()
    assert len(feature_names) == train_features.shape[1]
    assert len(feature_names) == test_features.shape[1]
    y_train = DataReader.read_training_labels()
    y_test = DataReader.read_test_labels()
    assert train_features.shape[0] == y_train.shape[0]
    assert test_features.shape[0] == y_test.shape[0]
    return train_features, y_train, test_features, y_test


def main():
    train_features, y_train, test_features, y_test = load_data()
    selection_transformers = make_selection_transformers()
    for label_ix in range(3):
        y_train_label = y_train[:, label_ix]
        y_test_label = y_test[:, label_ix]
        print('Train labels: {} Test labels: {}'.format(Counter(y_train_label), Counter(y_test_label)))
        for selection_transformer in selection_transformers:
            pipeline = create_pipline(selection_transformer)
            pipeline.fit(train_features, y_train_label)
            predictions = pipeline.predict(test_features)
            print('AUC: ', roc_auc_score(class_to_binary(y_test_label), class_to_binary(predictions)))


if __name__ == '__main__':
    main()
