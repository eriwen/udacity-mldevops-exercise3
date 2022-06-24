import logging
import numpy as np
import os.path
import pickle
import warnings
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score

warnings.simplefilter("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
model_file_name = "xgboost-classifier.json"
encoder_file_name = "encoder.pkl"
lb_file_name = "lb.pkl"


def train_model(X_train, y_train, fixed_params):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    best_params = grid_search(X_train, y_train, fixed_params)
    return xgb.XGBClassifier(**best_params, **fixed_params).fit(X_train, y_train)


def load_model(file_path):
    """
    Load XGBoost model, encoder, and label binarizer from the supplied file path.

    Inputs
    ------
    file_path : String
        Absolute file path where model resides.
    Returns
    -------
    booster
        XGBoost model.
    encoder : OneHotEncoder
        Category encoder.
    lb : LabelBinarizer
        Label binarizer.
    """
    booster = xgb.Booster()
    booster.load_model(os.path.join(file_path, model_file_name))
    encoder = pickle.load(open(os.path.join(file_path, encoder_file_name), "rb"))
    lb = pickle.load(open(os.path.join(file_path, lb_file_name), "rb"))
    return booster, encoder, lb


def save_model(model, encoder, lb, file_path):
    """
    Save given model to the supplied file path.

    Inputs
    ------
    model : XGBModel
        XGBoost model.
    encoder : OneHotEncoder
        Category encoder.
    lb : LabelBinarizer
        Label binarizer.
    file_path : String
        Directory path to store model.
    """
    model.get_booster().save_model(os.path.join(file_path, model_file_name))
    pickle.dump(encoder, open(os.path.join(file_path, encoder_file_name), "wb"))
    pickle.dump(lb, open(os.path.join(file_path, lb_file_name), "wb"))


def grid_search(X_train, y_train, fixed_params, scorer='roc_auc'):
    cv_params = [{
        'n_estimators': range(50, 400, 50),
        'max_depth': range(3, 8, 1),
        'eta': [0.01, 0.05, 0.1]
    }]
    logger.info(f"Tuning hyperparameters for {scorer}")

    cvs = GridSearchCV(xgb.XGBClassifier(**fixed_params), cv_params, scoring=scorer, n_jobs=-1).fit(X_train, y_train)

    logger.info('Best parameters set found on development set:')
    logger.info(cvs.best_params_)
    return cvs.best_params_


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
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def model_performance(model, X_slice, y_slice):
    predictions = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, predictions)
    return {"precision": precision, "recall": recall, "fbeta": fbeta}


def generate_slices(df, categorical_features):
    slices = {}
    for feature in categorical_features:
        for cls in df[feature].unique():
            data_slice = df[df[feature] == cls]
            slices[f"{feature} => {cls}"] = data_slice
    return slices


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : XGBClassifier
        XGBoost model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(xgb.DMatrix(X))
