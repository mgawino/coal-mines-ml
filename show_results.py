# -*- coding: utf-8 -*-
import glob
import json
import os
import textwrap
from collections import Counter

import click
from reader import DataReader
from terminaltables import AsciiTable
from utils import RESULTS_PATH


STATS_HEADER = ['AUC', 'model', 'f_num', 'score_fun', 'label']
TIMES_HEADER = ['c_time', 's_time']
FEATURES_HEADER = ['selected_features']
PARAMS_HEADER = ['best_params']
ALL_LABELS = 'ALL'
LABEL_TO_METHANOMETER = {
    0: 'MM263',
    1: 'MM264',
    2: 'MM256',
    3: ALL_LABELS
}


def iter_results():
    for result_path in glob.glob(os.path.join(RESULTS_PATH, '*')):
        with open(result_path) as result_file:
            yield json.load(result_file)


def _sort_list_of_lists(data, header, sort_keys, reverse):
    sort_indexes = [STATS_HEADER.index(sort_key) for sort_key in sort_keys]
    return sorted(
        data,
        key=lambda row: tuple(row[sort_index] for sort_index in sort_indexes),
        reverse=reverse
    )


def _read_data(header):
    data = []
    for result in iter_results():
        row = [result.get(key) for key in header]
        data.append(row)
    return data


def _wrap_selected_features(data, header, feature_limit):
    selected_features_ix = header.index('selected_features')
    for result in data:
        selected_features = result[selected_features_ix]
        if selected_features is not None:
            selected_features = ','.join(selected_features[:feature_limit])
            result[selected_features_ix] = textwrap.fill(selected_features, width=50)


def modify_data(data, header, include_features, feature_count, filter_label):
    model_ix = header.index('model')
    label_ix = header.index('label')
    for result in data:
        result[label_ix] = LABEL_TO_METHANOMETER[result[label_ix]]
    if filter_label != ALL_LABELS:
        data = [result for result in data if result[label_ix] == filter_label]
    for result in data:
        if result[model_ix] == 'RandomForestClassifier':
            result[model_ix] = 'RandomForest'
    if include_features:
        _wrap_selected_features(data, header, feature_count)
    return data


def show_results(sort_keys, sort_desc, include_features, feature_count, include_times, include_params, filter_label):
    header = STATS_HEADER
    if include_features:
        header.extend(FEATURES_HEADER)
    if include_times:
        header.extend(TIMES_HEADER)
    if include_params:
        header.extend(PARAMS_HEADER)
    data = _read_data(header)
    data = modify_data(data, header, include_features, feature_count, filter_label)
    if sort_keys:
        data = _sort_list_of_lists(data, header, sort_keys, sort_desc)
    print(AsciiTable([header] + data).table)


def show_summary():
    train_labels = DataReader.read_training_labels()
    test_labels = DataReader.read_test_labels()
    for label_ix in range(3):
        print('{} labels:'.format(LABEL_TO_METHANOMETER[label_ix]))
        train_counts = Counter(train_labels[:, label_ix])
        test_counts = Counter(test_labels[:, label_ix])
        print('Train -> ' + ' '.join('{} {}'.format(key, value) for key, value in train_counts.items()))
        print('Test -> ' + ' '.join('{} {}'.format(key, value) for key, value in test_counts.items()))


@click.command()
@click.option('--describe-data', '-dd', is_flag=True)
@click.option('--sort-keys', '-s', multiple=True, type=click.Choice(STATS_HEADER))
@click.option('--sort-desc', '-desc', is_flag=True)
@click.option('--include-features', '-if', is_flag=True)
@click.option('--feature-count', '-fc', type=click.INT, default=10)
@click.option('--include-times', '-it', is_flag=True)
@click.option('--include-params', '-ip', is_flag=True)
@click.option(
    '--filter-label', '-fl',
    default=ALL_LABELS,
    type=click.Choice(list(LABEL_TO_METHANOMETER.values()))
)
def main(describe_data, **kwargs):
    if describe_data:
        show_summary()
    else:
        show_results(**kwargs)


if __name__ == '__main__':
    main()
