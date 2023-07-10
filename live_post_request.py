import requests
import json

url = "https://census-classifier-api.onrender.com/predict/"
data = {
    "age": 35,
    "workclass": "Public",
    "fnlgt": 55555,
    "education": "Doctorate",
    "education_num": 16,
    "marital_status": "Married",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 20000,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "United-States",
}

response = requests.post(
    url, data=json.dumps(data), headers={"Content-Type": "application/json"}
)

print(f"status_code: {response.status_code}")
print(response.text)
