from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y_true = response(X)
    y_noise = y_true + np.random.normal(scale=noise, size=n_samples)

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y_noise, name="y"), 2 / 3)
    train_X = np.array(train_X).reshape(-1)
    train_y = np.array(train_y).reshape(-1)
    test_X = np.array(test_X).reshape(-1)
    test_y = np.array(test_y).reshape(-1)

    go.Figure([
        go.Scatter(name='Polynomial Samples', x=X, y=y_true, mode='lines',
                   marker_color='rgb(0,0,0)'),
        go.Scatter(name='Train Samples', x=np.array(train_X).reshape(-1), y=np.array(train_y).reshape(-1),
                   mode='markers',
                   marker_color='rgb(162,70,208)'),
        go.Scatter(name='Test Samples', x=np.array(test_X).reshape(-1), y=np.array(test_y).reshape(-1), mode='markers',
                   marker_color='rgb(45,186,40)')
    ]).update_layout(title=r"$\text{}\text{Polynomial Model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) }$",
                     xaxis_title=r"$\text{X value}$",
                     yaxis_title=r"$\text{y polynomial response}$").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_range = list(range(11))
    errors = []
    for k in k_range:
        estimator = PolynomialFitting(k)
        errors.append(cross_validate(estimator, train_X, train_y, mean_square_error, 5))

    go.Figure([go.Scatter(name='Train Error', x=k_range, y=np.array(errors)[:, 0], mode='markers+lines',
                          marker_color='rgb(223,78,228)'),
               go.Scatter(name='Validation Error', x=k_range, y=np.array(errors)[:, 1], mode='markers+lines',
                          marker_color='rgb(135,243,62)')]) \
        .update_layout(title=r"$\text{Scored (Train/Validation) Error, as function of K}$",
                       xaxis_title=r"$\text{K value}$",
                       yaxis_title=r"$\text{Error Value}$").show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(np.array(errors)[:, 1])
    poly_best_k = PolynomialFitting(int(best_k))
    poly_best_k.fit(train_X, train_y)
    best_k_test_error = mean_square_error(test_y, poly_best_k.predict(test_X))
    print("----------")
    print("Best polynomial degrre is: ", best_k)
    print("Test error for k* model: {:.2f}".format(best_k_test_error))
    print("Validation error for k* model: {:.2f}".format(np.array(errors)[:, 1][best_k]))
    print("----------")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, text_y = split_train_test(pd.DataFrame(X), pd.Series(y, name="y"), train_proportion=0.5)


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range = np.linspace(0, 3, 500)
    ridge_errors = []
    lasso_errors = []
    for lam in lam_range:
        ridge = RidgeRegression(lam)
        lasso = Lasso(lam)
        ridge_errors.append(cross_validate(ridge, train_X, train_y, mean_square_error, 5))
        lasso_errors.append(cross_validate(lasso, train_X, train_y, mean_square_error, 5))

    go.Figure([
               go.Scatter(name='Lasso Train Error', x=lam_range, y=np.array(lasso_errors)[:, 0], mode='markers+lines',
                          marker_color='rgb(50,150,10)'),
               go.Scatter(name='Lasso Validation Error', x=lam_range, y=np.array(lasso_errors)[:, 1],
                          mode='markers+lines',
                          marker_color='rgb(135,243,62)')
               ]) \
        .update_layout(title=r"$\text{Scored (Train/Validation) Lasso Error, as function of lambda}$",
                       xaxis_title=r"$\text{lambda value}$",
                       yaxis_title=r"$\text{Error Value}$").show()

    go.Figure([go.Scatter(name='Ridge Train Error', x=lam_range, y=np.array(ridge_errors)[:, 0], mode='markers+lines',
                          marker_color='rgb(223,78,228)'),
               go.Scatter(name='Ridge Validation Error', x=lam_range, y=np.array(ridge_errors)[:, 1],
                          mode='markers+lines',
                          marker_color='rgb(150,20,100)')
               ]) \
        .update_layout(title=r"$\text{Scored (Train/Validation) Ridge Error, as function of lambda}$",
                       xaxis_title=r"$\text{lambda value}$",
                       yaxis_title=r"$\text{Error Value}$").show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lam = lam_range[np.argmin(np.array(ridge_errors)[:, 1])]
    lasso_best_lam = lam_range[np.argmin(np.array(lasso_errors)[:, 1])]
    print("Ridge Best lambda: ", ridge_best_lam, " With Errors: ", ridge_errors[np.argmin(np.array(ridge_errors)[:, 1])])
    print("Lasso Best lambda: ", lasso_best_lam, " With Errors: ", lasso_errors[np.argmin(np.argmin(np.array(lasso_errors)[:, 1]))])


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(noise=5)
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
