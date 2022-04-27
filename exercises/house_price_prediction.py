from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from utils import animation_to_gif

pio.templates.default = "simple_white"

# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    """
    return full_data.loc[:, full_data.columns != 'price'], full_data[["price"]]
    """
    full_data = full_data[full_data["price"] > 0]
    # full_data = full_data[full_data["bedrooms"] > 0 & full_data["bedrooms"] < 20]
    full_data = full_data[full_data["bathrooms"] > 0]
    full_data = full_data[full_data["sqft_living"] >= 0]
    full_data = full_data[full_data["sqft_lot"] > 10]
    full_data = full_data[(full_data.waterfront == 1) | (full_data.waterfront == 0)]
    full_data = full_data[full_data["view"] >= 0]
    # full_data = full_data[full_data["zipcode"] >= 90000]
    full_data = pd.get_dummies(full_data, columns=['zipcode'])
    full_data = full_data[(full_data.condition >= 1) & (full_data.condition <= 5)]
    full_data = full_data[(full_data.grade >= 0) & (full_data.grade <= 14)]
    full_data = full_data[(full_data.sqft_above >= 200) & (full_data.sqft_above <= 10000)]
    full_data = full_data[(full_data.sqft_basement >= 0) & (full_data.sqft_basement <= 5000)]
    full_data = full_data[(full_data.floors >= 0) & (full_data.floors <= 10)]
    full_data['last_work'] = full_data[['yr_built', 'yr_renovated']].max(axis=1)
    full_data = full_data[(full_data.last_work >= 0) & (full_data.last_work <= 2020)]

    full_data.drop(['long', 'date', 'lat', 'yr_renovated', 'yr_built', "condition", "sqft_lot", "sqft_lot15"], axis=1, inplace=True)

    prices = full_data.loc[:, "price"]

    return full_data.drop(['price'], axis=1), prices
    # return full_data.loc[:, ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "waterfront", "view", "zipcode"
    #                   , "condition", "grade", "sqft_above", "sqft_basement", "floors", "last_work"]], prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigma_y = np.std(y)
    features_corr = []
    for feature in X.columns:
        col = X.loc[:, feature]
        features_corr.append(np.cov(col.T, y.T)[0][1] / (np.std(col) * sigma_y))
        go.Figure([go.Scatter(x=X.loc[:, feature], y=y, mode='markers',
                              name=r'$\widehat\mu$')],
                  layout=go.Layout(
                      title="{feature_name} effect on price. Condition's correlation is {correlation}".format(feature_name=feature, correlation=features_corr[-1]),
                      xaxis_title=r"$\text{feature}$",
                      yaxis_title="Price",
                      height=300)).write_image(output_path + "/condition_to_price.png")

    fig = go.Figure([go.Bar(x=X.columns, y=features_corr,
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Features correlations of house price}$",
                  xaxis_title="$\text{ number of samples}$",
                  yaxis_title="Pearson Correlation",
                  height=300))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    lr = LinearRegression()
    percentages = np.linspace(10, 100, 91)
     # = load_data("../datasets/house_prices.csv")
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_losses = []
    std_losses = []
    for p in percentages:
        loss_for_p = []
        for i in range(10):
            lr = LinearRegression()
            p_train_X, p_train_y, p_drop_X, p_drop_y = split_train_test(train_X, train_y, (p / 100))
            lr.fit(p_train_X.to_numpy(), p_train_y.to_numpy())
            loss_for_p.append(lr.loss(test_X.to_numpy(), test_y.to_numpy()))
        mean_losses.append(np.mean(loss_for_p))
        std_losses.append(np.std(loss_for_p))

    fig = go.Figure([go.Scatter(x=percentages, y=mean_losses, mode="markers+lines",
                                name=r'$\widehat\mu$')],
                    layout=go.Layout(
                        title=r"$\text{Mean loss and std for percentage}$",
                        xaxis_title="$\text{p - percentage of training data}$",
                        yaxis_title="std of 10 samples with p percentage from training data",
                        height=300))
    fig.show()

    y_loss_std_plus = []
    y_loss_std_minus = []
    for i in range(len(mean_losses)):
        y_loss_std_plus.append(mean_losses[i] + 2 * std_losses[i])
        y_loss_std_minus.append(mean_losses[i] - 2 * std_losses[i])

    fig = go.Figure(data=(go.Scatter(x=percentages, y=mean_losses, mode="markers+lines",
                                     name="Mean Loss for percentage", line=dict(dash="dash"),
                                     marker=dict(color="green", opacity=.7)),
     go.Scatter(x=percentages, y=y_loss_std_plus, fill=None, mode="lines", line=dict(color="pink"),
                showlegend=False),
     go.Scatter(x=percentages, y=y_loss_std_minus, fill='tonexty', mode="lines", line=dict(color="pink"),
                showlegend=False)),
                    layout=go.Layout(
                        title=r"$\text{Mean loss and std for percentage}$",
                        xaxis_title="$\text{p - percentage of training data}$",
                        yaxis_title="std of 10 samples with \n p percentage from training data",
                        height=600))
    fig.show()

