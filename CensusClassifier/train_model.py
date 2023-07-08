import os
import pandas as pd
import wandb
from constants import CAT_FEATURES
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from CensusClassifier.utils.logger import logging
from CensusClassifier.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    compute_sliced_model_metrics_categorical,
)
from CensusClassifier.ml.data import load_data


def main():
    data = load_data("./data/cleaned_df.csv")
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    rf_classifier = train_model(X_train, y_train)
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(rf_classifier, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    run = wandb.init(
        project="census-classification",
        group="train_random_forest",
        job_type="model_metrics",
    )
    run.summary["precision"] = precision
    run.summary["recall"] = recall
    run.summary["fbeta"] = fbeta
    run.log(
        {"confusion_matrix": wandb.plot.confusion_matrix(y_true=y_test, preds=preds)}
    )
    run.finish()
    run = wandb.init(
        project="census-classification",
        group="train_random_forest",
        job_type="metrics_slices",
    )
    metrics_slices_df = pd.DataFrame()
    for feature in CAT_FEATURES:
        metrics_slices_df = pd.concat(
            [
                metrics_slices_df,
                compute_sliced_model_metrics_categorical(test, feature, y_test, preds),
            ]
        )
    slices_table = wandb.Table(dataframe=metrics_slices_df)
    run.log({"slices_table": slices_table})
    run.finish()


if __name__ == "__main__":
    main()
