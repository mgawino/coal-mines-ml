# -*- coding: utf-8 -*-

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


def _prob_test(feature_column, target, feature_score, score_fun):
    random_scores = []
    for i in range(1000):
        random_column = np.random.permutation(feature_column)
        score = score_fun(random_column, target)
        random_scores.append(score)
    greater_scores = sum(1 if score > feature_score else 0 for score in random_scores)
    return greater_scores / 1000.0


def mrmr(X, y, scores, feature_names, max_features):
    score_cache = dict()

    def _score_fun(x, y):
        score, _ = pearsonr(x, y)
        return score

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

    feature_indices = set(range(X.shape[1]))
    max_ix = np.argmax(scores)
    selected_feature_indices = [max_ix]
    feature_indices.remove(max_ix)
    while len(selected_feature_indices) != max_features:
        max_diff = -1000
        max_ix = -1
        for feature_ix in feature_indices:
            diff = scores[feature_ix] - (_redundancy(feature_ix) / len(selected_feature_indices))
            if diff > max_diff:
                max_diff = diff
                max_ix = feature_ix
        if max_diff < 0 or _prob_test(X[:, max_ix], y, scores[max_ix], _score_fun) > 0.01:
            break
        feature_indices.remove(max_ix)
        selected_feature_indices.append(max_ix)

    return selected_feature_indices
