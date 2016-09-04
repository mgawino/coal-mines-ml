# -*- coding: utf-8 -*-
from reader import DataReader


def main():
    reader = DataReader()
    print(list(reader.iter_sensor_names()))
    print(len(list(reader.iter_test_data())))


if __name__ == '__main__':
    main()