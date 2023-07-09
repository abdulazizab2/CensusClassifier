import pytest
from CensusClassifier.ml.model import (
    train_model,
    load_model,
    inference,
    compute_sliced_model_metrics_categorical,
)
import numpy as np


@pytest.fixture(scope="session")
def model():
    return load_model("./model/census-rf.joblib")


def test_train_model(splitted_data):
    X_train, y_train = splitted_data
    try:
        rf_classifier = train_model(X_train, y_train)
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")
    assert rf_classifier is not None


def test_compute_model_metrics_slices(data, splitted_data, model):
    # setup
    train, _ = data
    X, y = splitted_data
    preds = inference(model, X)
    preds = np.where(preds == "<=50K", 0, 1)
    y = np.where(y == "<=50K", 0, 1)
    #
    with pytest.raises(Exception):
        _ = compute_sliced_model_metrics_categorical(
            train, "capital-loss", y, preds
        )  # numerics

    try:
        metrics_slides_df = compute_sliced_model_metrics_categorical(
            train, "workclass", y, preds
        )
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")

    assert (
        metrics_slides_df.columns
        == ["feature", "slice", "slice_samples", "precision", "recall", "fbeta"]
    ).all()
