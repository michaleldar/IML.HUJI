from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from utils import *
pio.templates.default = "simple_white"

import plotly.io as pio
pio.renderers.default = "browser"


def test_univariate_gaussian():
    mu = 10
    sigma = 1

    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(mu, sigma, 1000)
    u = UnivariateGaussian()
    u.fit(samples)
    print(u.mu_, u.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)

    estimated_mus = []
    for m in ms:
        X = samples[0:m]
        estimated_mus.append(np.abs(u.fit(X).mu_ - mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mus, mode='markers+lines',
                          name=r'$\widehat\mu$'),
               ], layout=go.Layout(barmode='overlay',
                          title=r"$\text{Estimator vs. true value}$",
                          xaxis_title="Number of samples",
                          yaxis_title="Absolute distance |estimates - true value|",
                          height=450)).show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    samples_normal_vals = u.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=samples_normal_vals, mode='markers',
                          name=r'$\widehat\mu$'),
               ], layout=go.Layout(barmode='overlay',
                                   title=r"$\text{PDF values for samples}$",
                                   xaxis_title="PDF sample values",
                                   yaxis_title="Sample value",
                                   height=450)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    m = MultivariateGaussian()
    multi_mu = [0, 0, 4, 0]
    multi_cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multi_x = np.random.multivariate_normal(multi_mu, multi_cov, 1000)
    m.fit(multi_x)
    print("Expectation: ", m.mu_, "\n")
    print("Cov: ", m.cov_, "\n")

    # Question 5 - Likelihood evaluation
    ms_f1_f2 = np.linspace(-10, 10, 200).astype(float)
    samples_to_vals = np.array(np.meshgrid(ms_f1_f2, 0, ms_f1_f2, 0)).T.reshape(-1, 4)

    log_likelihood_func = lambda x: MultivariateGaussian.log_likelihood(x, multi_cov, multi_x)
    log_likelihood = np.apply_along_axis(log_likelihood_func, 1, samples_to_vals)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=ms_f1_f2, y=ms_f1_f2, z=log_likelihood.reshape(200, 200).T,
                             colorbar=dict(title="Log likelihood")))
    fig.update_layout(
        title="values of log likelihood for changed f1, f3 mean values (f1, 0, f3, 0)",
        xaxis_title="f3",
        yaxis_title="f1")
    fig.show()

    # Question 6 - Maximum likelihood
    print("f1, f3 for max likelihood: ")
    print("f1: %.3f" % samples_to_vals[np.argmax(log_likelihood)][0])
    print("f3: %.3f" % samples_to_vals[np.argmax(log_likelihood)][2])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
