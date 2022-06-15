from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    validations = np.array_split(np.c_[X, y], cv)

    train_scores_sum, test_scores_sum = 0, 0

    for k in range(cv):
        xy_concatenated = np.concatenate([*validations[:k], *validations[k + 1:]], axis=0)
        estimator.fit(xy_concatenated[:, :-1], xy_concatenated[:, -1])
        train_scores_sum += scoring(estimator.predict(xy_concatenated[:, :-1]),
                                     xy_concatenated[:, -1])
        # test_scores_sum += scoring(estimator.predict(validations[k][:, :-1]), validations[k][:, -1])
        test_scores_sum += scoring(estimator.predict(validations[k][:, :-1]), validations[k][:, -1])
    return train_scores_sum / cv, test_scores_sum / cv
