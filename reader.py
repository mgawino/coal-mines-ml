# -*- coding: utf-8 -*-

import csv
from collections import namedtuple

from utils import grouped

SensorLabels = namedtuple('SensorLabels', ['label1', 'label2', 'label3'])
SensorData = namedtuple('SensorData', ['values'])


class DataReader(object):
    SENSOR_DATA_COUNT_IN_ROW = 600
    SENSOR_NUM = 28
    SENSOR_NAMES_FILE_PATH = "data/column_names.txt"
    TRAINING_FILE_PATHS = ("data/trainingData{}.csv".format(i) for i in range(1, 6))
    TRAINING_LABEL_PATHS = ("data/trainingLabels{}.csv".format(i) for i in range(1, 6))
    TEST_FILE_PATH = "data/testData.csv"
    TEST_LABELS_PATH = "data/testLabels.csv"

    def _iter_data(self, filepaths):
        for filepath in filepaths:
            with open(filepath) as file:
                reader = csv.reader(file)
                for line in reader:
                    sensor_data = list(grouped(line, self.SENSOR_DATA_COUNT_IN_ROW))
                    assert len(sensor_data) == self.SENSOR_NUM
                    yield SensorData(sensor_data)

    def iter_test_data(self):
        yield from self._iter_data((self.TEST_FILE_PATH,))

    def iter_training_data(self):
        yield from self._iter_data(self.TRAINING_FILE_PATHS)

    def _iter_labels(self, filepaths):
        for filepath in filepaths:
            with open(filepath) as file:
                reader = csv.reader(file)
                for sensor_labels in reader:
                    assert len(sensor_labels) == 3
                    yield SensorLabels(*sensor_labels)

    def iter_test_labels(self):
        yield from self._iter_labels((self.TEST_LABELS_PATH,))

    def iter_training_labels(self):
        yield from self._iter_labels(self.TRAINING_LABEL_PATHS)

    def iter_sensor_names(self):
        with open(self.SENSOR_NAMES_FILE_PATH) as file:
            for column_names in grouped(file, self.SENSOR_DATA_COUNT_IN_ROW):
                sensor_name = column_names[0].split('_')[0]
                yield sensor_name
