# -*- coding: utf-8 -*-
import numpy as np
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.statistical_based.gini_index import gini_index

MAX_FEATURES = 20


def mrmr_wrapper(X, y):
    indexes = mrmr(X, y, n_selected_features=MAX_FEATURES)
    scores = np.zeros(X.shape[1])
    for ix, feature_ix in enumerate(reversed(indexes)):
        scores[feature_ix] = ix
    return scores


def gini_index_wrapper(X, y):
    scores = gini_index(X, y)
    return np.negative(scores)
