# -*- coding: utf-8 -*-
import glob
import json
import os
import textwrap

import click
from terminaltables import AsciiTable
from utils import RESULTS_PATH


STATS_HEADER = ['AUC', 'model', 'f_num', 'score_fun', 'label']
TIMES_HEADER = ['c_time', 's_time']
FEATURES_HEADER = ['selected_features']
PARAMS_HEADER = ['best_params']


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


def _wrap_selected_features(data, header, feature_limit, feature_width):
    selected_features_ix = header.index('selected_features')
    for result in data:
        selected_features = result[selected_features_ix]
        if selected_features is not None:
            selected_features = ','.join(selected_features[:feature_limit])
            result[selected_features_ix] = textwrap.fill(selected_features, width=feature_width)


def modify_data(data, header, include_features, feature_limit, feature_width):
    model_ix = header.index('model')
    for result in data:
        if result[model_ix] == 'RandomForestClassifier':
            result[model_ix] = 'RandomForest'
    if include_features:
        _wrap_selected_features(data, header, feature_limit, feature_width)


@click.command()
@click.option('--sort-keys', '-s', multiple=True, type=click.Choice(STATS_HEADER))
@click.option('--sort-desc', '-desc', is_flag=True)
@click.option('--include-features', '-if', is_flag=True)
@click.option('--feature-limit', '-fl', type=click.INT, default=10)
@click.option('--feature-width', '-fw', type=click.INT, default=40)
@click.option('--include-times', '-it', is_flag=True)
@click.option('--include-params', '-ip', is_flag=True)
def main(sort_keys, sort_desc, include_features, feature_limit, feature_width, include_times, include_params):
    header = STATS_HEADER
    if include_features:
        header.extend(FEATURES_HEADER)
    if include_times:
        header.extend(TIMES_HEADER)
    if include_params:
        header.extend(PARAMS_HEADER)
    data = _read_data(header)
    modify_data(data, header, include_features, feature_limit, feature_width)
    if sort_keys:
        data = _sort_list_of_lists(data, header, sort_keys, sort_desc)
    print(AsciiTable([header] + data).table)


if __name__ == '__main__':
    main()
