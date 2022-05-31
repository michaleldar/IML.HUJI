import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(lambda: DecisionStump(), n_learners)
    ada._fit(train_X, train_y)
    test_loss = []
    train_loss = []
    for i in range(n_learners):
        test_loss.append(ada.partial_loss(test_X, test_y, i+1))
        train_loss.append(ada.partial_loss(train_X, train_y, i+1))

    print(train_loss)
    print(test_loss)

    go.Figure([go.Scatter(x=list(range(250)), y=train_loss, mode="markers+lines", name="Train Data Loss",
                          marker=dict(color="red")),
                go.Scatter(x=list(range(250)), y=test_loss, mode="markers+lines", name="Test Data Loss",
                           marker=dict(color="green"))],
                layout=go.Layout(title=r"$\text{(1) Simulated Data}$",
                                 xaxis={"title": "x - Explanatory Variable"},
                                 yaxis={"title": "y - Response"},
                                 height=400)).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Iterations, loss: {ada.partial_loss(test_X, test_y, t)}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: ada.partial_predict(X, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol="circle", colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of AdaBoost Model}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()
    print(f"argmin test loss {np.argmin(np.array(test_loss))}, with error: {test_loss[np.argmin(np.array(test_loss))]}")
    # Question 3: Decision surface of best performing ensemble
    go.Figure([decision_surface(lambda X: ada.partial_predict(X, 238), lims[0], lims[1], showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, line=dict(color="black", width=1),
                                           colorscale=[custom[0], custom[-1]]))],
              layout=go.Layout(title=rf"$\textbf{{AdaBoost Decision Boundaries Iterations:238, Accuracy: "
                                     rf"{accuracy(ada.partial_predict(test_X, 238), test_y)}, Loss: "
                                     rf"{ada.partial_loss(test_X, test_y, 238)}}}$")).show()

    # Question 4: Decision surface with weighted samples
    go.Figure([decision_surface(lambda X: ada.predict(X), lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y, line=dict(color="black"),
                                      size=np.abs((ada.D_ / np.max(ada.D_)) * 5),
                                      colorscale=[custom[0], custom[-1]]))],
              layout=go.Layout(
                  title=rf"$\textbf{{AdaBoost Decision Boundaries With Wighted dots}}$")).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250)
    fit_and_evaluate_adaboost(0.4, 250)
