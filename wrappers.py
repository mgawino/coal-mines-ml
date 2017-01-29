# -*- coding: utf-8 -*-
import numpy as np
from skfeature.function.statistical_based.CFS import cfs
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted


class SelectKBestWrapper(BaseEstimator, SelectorMixin):

    def __init__(self, index_function, k=None):
        self.k = k
        self.index_function = index_function
        self.indexes_ = None
        self.fitted_shape_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)

        if not callable(self.index_function):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.index_function, type(self.index_function)))

        self.indexes_ = np.asarray(self.index_function(X, y, n_selected_features=self.k))
        self.fitted_shape_ = X.shape[1]
        return self

    def _get_support_mask(self):
        check_is_fitted(self, ['indexes_', 'fitted_shape_'])
        mask = np.zeros((self.fitted_shape_, ), dtype=bool)
        mask[self.indexes_] = 1
        return mask


def cfs_wrapper(X, y, n_selected_features):
    indexes = cfs(X, y)
    assert len(indexes) > n_selected_features
    return indexes[:n_selected_features]