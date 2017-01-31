# -*- coding: utf-8 -*-
import glob
import json
import os

import click
from terminaltables import AsciiTable
from utils import RESULTS_PATH


HEADER = ['AUC', 'model', 'feature_num', 'score_fun', 'label_ix', 'classif_time', 'select_time']


def iter_results():
    for result_path in glob.glob(os.path.join(RESULTS_PATH, '*')):
        with open(result_path) as result_file:
            yield json.load(result_file)


@click.command()
@click.option('--sort-keys', '-s', multiple=True, type=click.Choice(HEADER))
@click.option('--sort-desc', '-desc', is_flag=True)
def main(sort_keys, sort_desc):
    results = []
    for result in iter_results():
        row = [result.get(key, '') for key in HEADER]
        results.append(row)
    if sort_keys:
        sort_indexes = [HEADER.index(sort_key) for sort_key in sort_keys]
        results = sorted(
            results,
            key=lambda row: tuple(row[sort_index] for sort_index in sort_indexes),
            reverse=sort_desc
        )
    print(AsciiTable([HEADER] + results).table)


if __name__ == '__main__':
    main()
