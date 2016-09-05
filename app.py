# -*- coding: utf-8 -*-

from reader import DataReader
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np
from sklearn.svm import SVC
from transformer import SensorFunctionTransformer


def main():
    sensor_names = DataReader.get_sensor_names()
    combined_features = FeatureUnion([
        ('max', SensorFunctionTransformer(np.max, sensor_names)),
        ('min', SensorFunctionTransformer(np.min, sensor_names)),
        ('median', SensorFunctionTransformer(np.median, sensor_names)),
        ('mean', SensorFunctionTransformer(np.mean, sensor_names))
    ], n_jobs=3)

    X_train = list(DataReader.iter_training_data())
    # y_train = list(DataReader.iter_training_labels())
    X_train_features = combined_features.transform(X_train)

    svm = SVC(kernel='linear')

    pipeline = Pipeline([
        ('features', combined_features),
        ('svm', svm)
    ])
    # test_data = np.asarray(list(reader.iter_test_data()))


if __name__ == '__main__':
    main()
