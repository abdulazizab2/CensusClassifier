import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from CensusClassifier.constants import CAT_FEATURES


@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(call):
    raise call.excinfo.value


@pytest.hookimpl(tryfirst=True)
def pytest_internalerror(excinfo):
    raise excinfo.value


@pytest.fixture(scope="session")
def data():
    data = pd.read_csv("./data/cleaned_df.csv")[0:100]  # take few samples for testing
    train, test = train_test_split(data, test_size=0.20)
    return train, test


@pytest.fixture(scope="session")
def splitted_data(data):
    train, _ = data  # does not matter which split for unit tests
    encoder = joblib.load("./model/categorical_encoder.joblib")
    y = train["salary"]
    X = train.drop(["salary"], axis=1)
    X_categorical = X[CAT_FEATURES].values
    X_continuous = X.drop(*[CAT_FEATURES], axis=1)
    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y
