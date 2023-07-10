from fastapi.testclient import TestClient
from main import app, load_models

client = TestClient(app)
load_models()


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello CensusClassifier!"}


def test_predict_over_50K():
    data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    }

    response = client.post("/predict", json=data)
    assert response.json() == {"prediction": ">50K"}


def test_predict_under_or_equal_50K():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    response = client.post("/predict", json=data)
    assert response.json() == {"prediction": "<=50K"}


def test_wrong_query():
    data = {"foo": 1}

    response = client.post("/predict", json=data)
    assert response.status_code == 422
