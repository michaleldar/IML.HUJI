from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


def univariate_normal_pdf(sample, mu, sigma):
    """
    calculates normal PDF with given parameters of mu and var
    """
    return ((1 / (sigma * np.sqrt(2 * np.pi))) *
            (np.exp((- 1 / 2) * (((sample - mu) / sigma) ** 2))))


def multivariate_normal_pdf(sample, mu, cov):
    """
    calculates normal PDF with given parameters of mu and var
    """
    return ((1 / (cov * np.sqrt(((2 * np.pi) ** len(sample)) * np.linalg.det(cov)))) *
            (np.exp((- 1 / 2) * (np.matmul(np.matmul(sample - mu, np.linalg.inv(cov)), sample - mu)))))


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        mu = X.mean()
        
        sum = 0
        for sample in X:
            sum += (sample - mu)**2

        if self.biased_:
            var = sum / X.size
        else:
            var = sum / (X.size - 1)
           
        self.mu_ = mu
        self.var_ = var
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        samples_normal_values = []
        for sample in X:
            samples_normal_values.append(univariate_normal_pdf(sample, self.mu_, np.sqrt(self.var_)))

        return np.array(samples_normal_values)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        return sum(map(lambda x: np.log(univariate_normal_pdf(x, mu, sigma)), X))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def _calculate_covariance(self, X: np.ndarray, i, j):
        sum = 0
        for k in range(len(X)):
            sum += ((X[k][i] - self.mu_[i])*(X[k][j] - self.mu_[j]))
        return float(sum) / (len(X) - 1)

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.empty((len(X[0]), len(X[0])), float)

        for i in range(len(X[0])):
            for j in range(len(X[0])):
                self.cov_[i][j] = self._calculate_covariance(X, i, j)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        samples_normal_values = []
        for sample in X:
            samples_normal_values.append(multivariate_normal_pdf(sample, self.mu_, self.cov_))

        return np.array(samples_normal_values)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        samples_dimension = len(X[0])
        num_of_samples = len(X)
        first = (((samples_dimension * num_of_samples) / 2) * np.log(2 * np.pi))
        second = ((num_of_samples / 2) * np.log(np.linalg.det(cov)))
        third = 0

        inv_cov = np.linalg.inv(cov)
        for i in range(len(X)):
            third += (np.matmul(np.matmul(X[i] - mu, inv_cov), X[i] - mu) / 2)

        return - first - second - third
