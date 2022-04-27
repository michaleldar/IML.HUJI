from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
from IMLearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset

    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data[["checkin_date",
                          "booking_datetime",
                          "checkout_date",
                          "charge_option",
                          "is_first_booking",
                          "original_selling_amount",
                          "cancellation_policy_code",
                          "is_first_booking"
                          ]]
    labels = full_data["cancellation_datetime"]

    return features, labels


def load_test_set(file_name: str):
    full_data = pd.read_csv(file_name).drop_duplicates()
    features = full_data[["checkin_date",
                          "booking_datetime",
                          "checkout_date",
                          "charge_option",
                          "is_first_booking",
                          "original_selling_amount",
                          "cancellation_policy_code",
                          "is_first_booking"
                          ]]
    return features


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, real_values, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame({"predictrd_Values": estimator.predict(X), "real_values": real_values}).to_csv(filename, index=False)


def predict_test_set(estimator: BaseEstimator, X: np.ndarray, file_name: str):
    pd.DataFrame({"predictrd_Values": estimator.predict(X)}).to_csv(file_name, index=False)


if __name__ == '__main__':
    # Load data
    data, cancellation_labels = load_data("../../IML_challenges/datasets/agoda_cancellation_train.csv")
    train_X, test_X, train_y, test_y = split_train_test(data, cancellation_labels)
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(np.array(train_X), np.array(train_y))
    test_set = load_test_set("test_set_week_1.csv")

    # Store model predictions over test set
    evaluate_and_export(estimator, np.array(test_X), AgodaCancellationEstimator.adjust_response(test_y),
                        "check_prediction.csv")
    predict_test_set(estimator, np.array(test_set),"316080076_313598492_318456290.csv")
    print("score is:")
    print(estimator.loss(np.array(test_X), np.array(test_y)))
