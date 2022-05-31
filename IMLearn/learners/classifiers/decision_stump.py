from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        global_min_loss = 2
        for feature in range(X.shape[1]):
            plus_thresh, plus_loss = self._find_threshold(X[:, feature], y, 1)
            minus_thresh, minus_loss = self._find_threshold(X[:, feature], y, -1)
            if plus_loss <= global_min_loss:
                self.threshold_ = plus_thresh
                self.j_ = feature
                self.sign_ = 1
                global_min_loss = plus_loss
            if minus_loss <= global_min_loss:
                self.threshold_ = minus_thresh
                self.j_ = feature
                self.sign_ = -1
                global_min_loss = minus_loss
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        """
        min_loss = 2
        min_thresh = -1
        for thresh_candidate in np.unique(values):
            y_pred = np.where(values >= thresh_candidate, sign, -sign)
            curr_loss = misclassification_error(np.array(y_pred), np.sign(labels))
            if curr_loss < min_loss:
                min_loss = curr_loss
                min_thresh = thresh_candidate
        return min_thresh, min_loss
        """
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_labels = labels[sorted_idx]
        thresh_candidates = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(values) - 1)]
        thresh_candidates = np.concatenate([[-np.inf], thresh_candidates, [np.inf]])
        min_loss = np.sum(np.absolute(sorted_labels[np.sign(sorted_labels) == sign]))
        losses_candidates = np.append(min_loss, min_loss - np.cumsum((sorted_labels * sign)))
        min_loss_ind = np.argmin(losses_candidates)
        return thresh_candidates[min_loss_ind], losses_candidates[min_loss_ind]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        y_miss = (np.sign(y) != y_pred).astype(int)
        return float(np.sum(y_miss * np.absolute(y)))
