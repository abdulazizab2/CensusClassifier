import pytest
from CensusClassifier.ml.data import load_data


def test_load_data():
    try:
        _ = load_data("./data/cleaned_df.csv")
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")
