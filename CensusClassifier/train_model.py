import os
import pandas as pd
from constants import CAT_FEATURES
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from CensusClassifier.utils.logger import logging
from CensusClassifier.ml.model import train_model
from CensusClassifier.ml.data import load_data


def main():
    data = load_data("./data/cleaned_df.csv")
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    rf_classifier = train_model(X_train, y_train)


if __name__ == "__main__":
    main()
