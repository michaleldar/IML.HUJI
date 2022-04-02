from __future__ import annotations
from typing import NoReturn

import sklearn.linear_model
from sklearn.linear_model import LogisticRegression

from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd
import datetime
import math


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters

        ----------


        Attributes
        ----------

        """
        self.estimator_ = LogisticRegression()
        self.fitted_ = False
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        data = AgodaCancellationEstimator.adjust_data(X)
        y_adjusted = AgodaCancellationEstimator.adjust_response(y)
        self.estimator_.fit(data, y_adjusted)
        self.fitted_ = True

    @staticmethod
    def adjust_response(X):
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if isinstance(X[sample], str):
                y[sample] = 1
            else:
                y[sample] = 0
        return y

    @staticmethod
    def adjust_data(X):
        # days between booking and check in
        col_1 = AgodaCancellationEstimator.date_to_days(X[:, 0]) - AgodaCancellationEstimator.date_to_days(X[:, 1])
        # days between checkout and check in
        col_2 = AgodaCancellationEstimator.date_to_days(X[:, 2]) - AgodaCancellationEstimator.date_to_days(X[:, 0])
        # payment method(now/later)
        col_3 = AgodaCancellationEstimator.pay_Now_or_later(X[:, 3])
        # is first booking
        col_4 = AgodaCancellationEstimator.first_booking(X[:, 4])

        # not sure but made better score
        # col_5 = X[:, 5]

        #cancellation policy
        col_6 = AgodaCancellationEstimator.canellation_policy(X[:, 6])

        col_7 = AgodaCancellationEstimator.first_booking(X[:, 7])
        # number of children
        # col_5 = X[:, 5]
        # number of adults
        #col_5 = X[:, 5]
        return np.array([col_1, col_2, col_3, col_4, col_6, col_7]).T

    @staticmethod
    def first_booking(X):
        false_scalar, true_scalar = 0, 1
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if X[sample]:
                y[sample] = true_scalar
            else:
                y[sample] = false_scalar
        return y

    @staticmethod
    def canellation_policy(X):
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if "100P" not in X[sample]:
                y[sample] = 100
            else:
                y[sample] = -1
        return y

    @staticmethod
    def pay_Now_or_later(X):
        now_scalar, later_scalar = -100, 100
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if X[sample] == "Pay Now":
                y[sample] = now_scalar
            else:
                y[sample] = later_scalar
        return y

    @staticmethod
    def date_to_days(X):
        def s_to_int(s):
            splited_by_space = s.split()
            splited_by_backslash = splited_by_space[0].split("-")
            return int(splited_by_backslash[0]) + int(splited_by_backslash[1]) * 30 + int(splited_by_backslash[2]) * 365

        return np.array([s_to_int(s) for s in X])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.estimator_.predict(AgodaCancellationEstimator.adjust_data(X))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return self.estimator_.score(AgodaCancellationEstimator.adjust_data(X),
                                     AgodaCancellationEstimator.adjust_response(y))
