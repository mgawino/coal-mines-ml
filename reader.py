# -*- coding: utf-8 -*-

import csv

import numpy as np


class DataReader(object):
    SENSOR_DATA_COUNT_IN_ROW = 600
    SENSOR_NUM = 28
    SENSOR_NAMES_FILE_PATH = "data/column_names.txt"
    TRAINING_FILE_PATHS = ("data/trainingData{}.csv".format(i) for i in range(1, 6))
    TRAINING_LABEL_PATHS = ("data/trainingLabels{}.csv".format(i) for i in range(1, 6))
    TEST_FILE_PATH = "data/testData.csv"
    TEST_LABELS_PATH = "data/testLabels.csv"

    @classmethod
    def _iter_file_data(cls, filepaths):
        for filepath in filepaths:
            file_data = []
            with open(filepath) as file:
                print('Reading file: {} ...'.format(filepath))
                reader = csv.reader(file)
                for sensors_data in reader:
                    assert len(sensors_data) == cls.SENSOR_NUM * cls.SENSOR_DATA_COUNT_IN_ROW
                    file_data.append(sensors_data)
                    break
            yield np.asarray(file_data, dtype=np.float32)

    @classmethod
    def read_test_data(cls):
        return next(cls._iter_file_data((cls.TEST_FILE_PATH,)))

    @classmethod
    def iter_train_files_data(cls):
        yield from cls._iter_file_data(cls.TRAINING_FILE_PATHS)

    @staticmethod
    def _iter_labels(filepaths):
        for filepath in filepaths:
            with open(filepath) as file:
                reader = csv.reader(file)
                for sensor_labels in reader:
                    assert len(sensor_labels) == 3
                    yield np.asarray(sensor_labels)
                    break

    @classmethod
    def read_test_labels(cls):
        return np.array(list(cls._iter_labels((cls.TEST_LABELS_PATH,))))

    @classmethod
    def read_training_labels(cls):
        return np.array(list(cls._iter_labels(cls.TRAINING_LABEL_PATHS)))

    @classmethod
    def get_sensor_names(cls):
        with open(cls.SENSOR_NAMES_FILE_PATH) as file:
            seen_names = set()
            sensor_names = []
            for column_name in file:
                sensor_name = column_name.split('_')[0]
                if sensor_name not in seen_names:
                    sensor_names.append(sensor_name)
                seen_names.add(sensor_name)
            assert len(sensor_names) == cls.SENSOR_NUM
            return sensor_names
