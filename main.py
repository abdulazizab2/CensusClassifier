from fastapi import FastAPI
from pydantic import BaseModel, Field
from CensusClassifier.constants import CAT_FEATURES
from CensusClassifier.ml.data import process_data
from CensusClassifier.ml.model import inference, load_model
from CensusClassifier.utils.logger import logging
import numpy as np
import pandas as pd
import json

app = FastAPI()

model = None
encoder = None


class ModelInput(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13)
    marital_status: str = Field(example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example="United-States")


@app.on_event("startup")
def load_models():
    global model, encoder
    model = load_model("./model/census-rf.joblib")
    encoder = load_model("./model/categorical_encoder.joblib")
    logging.info("Model and Encoder are loaded")


@app.get("/")
async def root():
    return {"message": "Hello CensusClassifier!"}


@app.post("/predict")
async def predict(data: ModelInput):
    data_dict = data.dict()
    data_dict = {key.replace("_", "-"): value for key, value in data_dict.items()}
    data_df = pd.DataFrame([data_dict])
    data_df, _, _, _ = process_data(
        data_df, CAT_FEATURES, training=False, encoder=encoder
    )
    result = inference(model=model, X=data_df)
    result = "<=50K" if result.item() == 0 else ">50K"
    logging.info(f"Request received: Predction: {result}")
    return {"prediction": result}
