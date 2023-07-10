import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
)
from CensusClassifier.utils.logger import logging
from typing import Dict


def train_model(X_train, y_train, save_model=True):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    save_model: boolean
        A boolean flag to save a model or not. Model will be saved in mode/
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf_classifier = RandomForestClassifier(random_state=42)
    logging.info("INFO: Random forest classifier is training ...")
    rf_classifier.fit(X_train, y_train)
    if save_model:
        joblib.dump(rf_classifier, "./model/census-rf.joblib")
        logging.info("INFO: Random forest classifier is saved in model/")
    return rf_classifier


def load_model(model_path):
    return joblib.load(model_path)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_sliced_model_metrics_categorical(
    df: pd.DataFrame, feature: str, y: np.ndarray, preds: np.ndarray
) -> pd.DataFrame:
    """
    Compute the performance of a model on slices for a given categorical feature

    Args:
        df (pd.DataFrame): pre-processed dataframe
        feature (str): name of the column to perform slicing
        y (np.array): known labels
        preds (np.array): predictions

    Returns:
        Dict: A pandas dataframe containing the slice metrics
    """

    if not df[feature].dtype == object:
        raise ValueError(
            f"Function expects a categorical feature. A non-categorical feature: {feature} is given"
        )

    slices = df[feature].unique().tolist()
    metrics_slices_df = pd.DataFrame(
        columns=["feature", "slice", "slice_samples", "precision", "recall", "fbeta"]
    )
    for slice in slices:
        mask = df[feature] == slice
        slice_samples = np.array(mask).sum()
        y_slice = y[mask]
        preds_slice = preds[mask]

        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        metrics_slices_df.loc[len(metrics_slices_df)] = {
            "feature": feature,
            "slice": slice,
            "slice_samples": slice_samples,
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }

    return metrics_slices_df
