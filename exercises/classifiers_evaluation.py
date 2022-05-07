from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import utils
from math import atan2, pi
pio.templates.default = "simple_white"
pio.renderers.default = 'browser'


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    def callback_func(perceptron, x_i, y_i):
        losses.append(perceptron._loss(X, y))

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        Xy = np.load("../datasets/" + f)
        X = Xy[:, [0, 1]]
        y = Xy[:, 2]
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=callback_func)
        perceptron.fit(X, y)

        # Plot figure
        go.Figure([go.Scatter(x=list(range(len(losses))), y=losses, mode='markers+lines', name=n)],
                  layout=go.Layout(title=n,
                                   xaxis_title="training iterations",
                                   yaxis_title="Normalized Misclassification Error")).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "triangle-up", "square"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        Xy = np.load("../datasets/" + f)
        X = Xy[:, [0, 1]]
        y = Xy[:, 2]

        # Fit models and predict over training set

        nb = GaussianNaiveBayes()
        nb._fit(X, y)
        nb_y_pred = nb._predict(X)


        lda = LDA()
        lda._fit(X, y)
        lda_y_pred = lda._predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        modles_prediction = [lda.predict, nb.predict]
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        ys = [lda_y_pred, nb_y_pred]
        model_names = ["LDA", "Naive Bayes"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{Model: {model_names[idx]} Accuracy: {accuracy(y, y_pred)}}}$" for idx, y_pred in enumerate(ys)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(ys):
            fig.add_traces([
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=ys[i], symbol=symbols[y.astype(int)],
                                                   colorscale=[[0.1111111111111111, "rgb(215,48,39)"],
                                                               [0.5555555555555556, "rgb(224,243,248)"],
                                                               [0.8888888888888888, "rgb(69,117,180)"]],
                                                   line=dict(color="black", width=1)))],
                           rows=1, cols=(i + 1))

            fig.update_layout(title=rf"$\textbf{{{f} Dataset}}$",
                              margin=dict(t=100)) \
                .update_xaxes(visible=True).update_yaxes(visible=True)

            fig.add_traces([get_ellipse(lda.mu_[k, :], lda.cov_) for k in range(lda.mu_.shape[0])], rows=1, cols=1)
            fig.add_traces([get_ellipse(nb.mu_[k,:], np.diag(nb.vars_[k, :])) for k in range(nb.mu_.shape[0])], rows=1,
                           cols=2)
            fig.add_traces([go.Scatter(x=[lda.mu_[k][0]], y=[lda.mu_[k][1]], mode="markers", marker=dict(color="black", symbol="x")) for k in range(lda.mu_.shape[0])], rows=1, cols=1)
            fig.add_traces([go.Scatter(x=[nb.mu_[k][0]], y=[nb.mu_[k][1]], mode="markers", marker=dict(color="black", symbol="x")) for k in range(nb.mu_.shape[0])], rows=1, cols=2)
            fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
