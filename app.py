# -*- coding: utf-8 -*-

from reader import DataReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np
from transformer import SensorFunctionTransformer


def main():
    sensor_names = DataReader.get_sensor_names()
    combined_features = FeatureUnion([
        ('max', SensorFunctionTransformer(np.max, sensor_names)),
        ('min', SensorFunctionTransformer(np.min, sensor_names)),
        ('median', SensorFunctionTransformer(np.median, sensor_names)),
        ('mean', SensorFunctionTransformer(np.mean, sensor_names))
    ], n_jobs=3)

    X_train = np.array(list(DataReader.iter_training_data()))
    y_train = np.array(list(DataReader.iter_training_labels()))
    print('Loaded training data: {} labels: {}'.format(X_train.shape, y_train.shape))

    X_test = np.array(list(DataReader.iter_test_data()))
    y_test = np.array(list(DataReader.iter_test_data()))
    print('Loaded test data: {} labels: {}'.format(X_test.shape, y_test.shape))

    pipeline = Pipeline([
        ('features', combined_features),
        ('random_forest', RandomForestClassifier())
    ])
    pipeline.fit(X_train, y_train[:, 0])
    print(pipeline.predict(X_test))

if __name__ == '__main__':
    main()
