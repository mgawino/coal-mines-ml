# -*- coding: utf-8 -*-

from reader import DataReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from transformer import DataFrameTransformer, FeatureExtractorTransformer


def main():
    X_train = np.array(list(DataReader.iter_training_data()))
    y_train = np.array(list(DataReader.iter_training_labels()))
    print('Loaded training data: {} labels: {}'.format(X_train.shape, y_train.shape))

    X_test = np.array(list(DataReader.iter_test_data()))
    y_test = np.array(list(DataReader.iter_test_labels()))
    print('Loaded test data: {} labels: {}'.format(X_test.shape, y_test.shape))

    pipeline = Pipeline([
        ('dataframe', DataFrameTransformer()),
        ('features', FeatureExtractorTransformer()),
        ('random_forest', RandomForestClassifier())
    ])
    pipeline.fit(X_train, y_train[:, 0])
    print(pipeline.predict(X_test))

if __name__ == '__main__':
    main()
