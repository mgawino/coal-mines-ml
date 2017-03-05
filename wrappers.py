# -*- coding: utf-8 -*-
from operator import itemgetter

import numpy as np
from scipy.stats import pearsonr
from skfeature.function.statistical_based.gini_index import gini_index


def gini_index_wrapper(X, y):
    scores = gini_index(X, y)
    return np.negative(scores)


def corr_wrapper(X, y):
    scores = []
    pvalues = []
    for column in X.T:
        corr, pvalue = pearsonr(column, y)
        scores.append(abs(corr))
        pvalues.append(pvalue)
    return np.asarray(scores), np.asarray(pvalues)


def mrmr(X, y, scores, max_features):
    score_cache = dict()

    def _score(first_ix, second_ix):
        cache_key = tuple(sorted((first_ix, second_ix)))
        if cache_key in score_cache:
            return score_cache[cache_key]
        first_column = X[:, cache_key[0]]
        second_column = X[:, cache_key[1]]
        score, _ = pearsonr(first_column, second_column)
        score_cache[cache_key] = abs(score)
        return score

    def _redundancy(feature_ix):
        return sum(_score(feature_ix, selected_ix) for selected_ix in selected_feature_indices)

    top_feature_indices = sorted(enumerate(scores), key=itemgetter(1))
    top_feature_indices = set(ix for ix, _ in top_feature_indices)
    selected_feature_indices = list()

    while len(selected_feature_indices) != max_features:
        max_diff = -1000
        max_ix = -1
        for feature_ix in top_feature_indices:
            diff = scores[feature_ix] - _redundancy(feature_ix)
            if diff > max_diff:
                max_diff = diff
                max_ix = feature_ix
        if max_diff < 0:
            break
        top_feature_indices.remove(max_ix)
        selected_feature_indices.append(max_ix)

    return selected_feature_indices
