import plotly.graph_objects as go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"
pio.renderers.default = 'browser'


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df["DayOfYear"] = pd.to_datetime(df["Date"]).dt.dayofyear
    df = df[(df.Temp > -50) & (df.Temp < 60)]
    df = df[(df.Year > 1945) & (df.Year < 2030)]
    df = df[(df.Month >= 1) & (df.Month <= 12)]
    df = df[(df.Day >= 1) & (df.Day <= 31)]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df.Country == "Israel"]
    years = israel_df.groupby("Year")
    fig = make_subplots(1, 4, subplot_titles="")
    for year in years.groups:
        fig.add_traces([go.Scatter(x=years.get_group(year)["DayOfYear"], y=years.get_group(year)["Temp"],
                                   mode="markers", name="year " + str(year))])

    fig.layout = go.Layout(title="Temperature over the year, and over different years (colored)",
                           xaxis_title="Day of the year",
                           yaxis_title="Temperature")
    fig.update_xaxes(showgrid=True, ticklabelmode="period")
    fig.show()

    months = israel_df.groupby("Month")
    months_std = []
    for month in months.groups:
        months_std.append(np.std(months.get_group(month)["Temp"]))

    go.Figure([go.Bar(x=list(months.groups), y=months_std)],
              layout=go.Layout(
                  title="std of Temperature For Month",
                  xaxis_title="month",
                  yaxis_title="std",
                  height=500)).show()

    # Question 3 - Exploring differences between countries
    country_month_df = df.groupby(['Country', 'Month'])
    country_month_df = country_month_df.Temp.agg(mean='mean', std='std').reset_index()
    px_fig = px.line(country_month_df, x=country_month_df.Month, y=country_month_df["mean"],
                     color=country_month_df.Country, error_y=country_month_df["std"],
                     title="Temperature for each month and country")
    px_fig.show()
    # Question 4 - Fitting model for different values of `k`
    y = israel_df.loc[:, "Temp"]
    X = israel_df.drop(columns=["Temp"])
    train_X, train_y, test_X, text_y = split_train_test(X, y)
    ks = np.linspace(1, 10, 10)
    k_loss = []
    for k in ks:
        pr = PolynomialFitting(int(k))
        pr._fit(train_X["DayOfYear"], train_y)
        loss_for_k = pr._loss(test_X["DayOfYear"], text_y)
        k_loss.append(loss_for_k)
        print(f"Loss for Polynomial fitting with degree {int(k)} is {loss_for_k:.2f}")

    go.Figure([go.Bar(x=ks, y=k_loss)],
              layout=go.Layout(
                  title="loss for k polinomyial fitting",
                  xaxis_title="k degree",
                  yaxis_title="loss",
                  height=500)).show()
    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)  # degree from question 4, with the best fit.
    model.fit(israel_df.DayOfYear, israel_df.Temp)
    df_countries = df.groupby("Country")
    country_to_loss = pd.DataFrame(columns=["Country", "Loss"])
    for country in df_countries.groups:
        if country == "Israel": continue
        country_to_loss.loc[len(country_to_loss.index)] = [country, model._loss(df_countries.get_group(country).
                                               DayOfYear, df_countries.get_group(country).Temp)]
    loss_by_country_bar = px.bar(country_to_loss, x="Country", y="Loss",
                                 title="Polynomial model with degree 5, Fitted on Israel data - to Loss")
    loss_by_country_bar.show()

