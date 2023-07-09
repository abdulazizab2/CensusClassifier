import pytest
from CensusClassifier.ml.data import load_data, process_data
from CensusClassifier.constants import CAT_FEATURES


def test_load_data():
    try:
        _ = load_data("./data/cleaned_df.csv")
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")


def test_process_data(data):
    train, test = data
    try:
        _, _, encoder, lb = process_data(
            train, CAT_FEATURES, label="salary", training=True
        )
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")
    try:
        _, _, _, _ = process_data(
            test,
            categorical_features=CAT_FEATURES,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")
