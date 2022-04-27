from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        k_classes_values, k_classes_count = np.unique(y, return_counts=True)
        self.classes_ = k_classes_values
        self.pi_ = k_classes_count / y.shape[0]

        n_features = X.shape[1]
        mu_array = np.zeros((k_classes_values.shape[0], n_features))
        var_array = np.zeros((k_classes_values.shape[0], n_features))
        for idx, class_k in enumerate(self.classes_):
            X_k = X[y == class_k]
            mu_array[idx] = np.mean(X_k, axis=0)
            var_array[idx] = np.var(X_k, axis=0)
        self.vars_ = var_array
        self.mu_ = mu_array
        self.fitted_ = True

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
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        pdf = lambda X, var, mu: ((1 / (np.sqrt(2 * var * np.pi))) * (np.exp((- 1 / 2) * (((X - mu) / np.sqrt(var)) ** 2))))
        likelihood = np.empty((X.shape[0], self.classes_.shape[0]))
        for idx, k in enumerate(self.classes_):
            k_distribution = pdf(X, self.vars_[idx], self.mu_[idx]) * self.pi_[idx]
            likelihood[:, idx] = np.prod(k_distribution, axis=1)
        return likelihood

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
        raise NotImplementedError()
