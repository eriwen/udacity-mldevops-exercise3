import numpy as np
import os.path
from sklearn.model_selection import train_test_split
import pandas as pd
import pytest
import sys
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.ml.data import load_data, process_data, get_categorical_features
from model.ml.model import save_model, load_model, model_performance, generate_slices, inference


@pytest.fixture
def data():
    return load_data(os.path.join(os.path.dirname(__file__), '..'))


def test_save_and_load_model(data, tmpdir):
    X_train, y_train, encoder, lb = process_data(data)
    model = xgb.XGBClassifier(use_label_encoder=False).fit(X_train, y_train)
    save_model(model, encoder, lb, tmpdir)
    loaded_model, loaded_encoder, loaded_lb = load_model(tmpdir)
    assert type(loaded_model) is xgb.Booster
    assert type(loaded_encoder) is OneHotEncoder
    assert type(loaded_lb) is LabelBinarizer


def test_inference(data):
    model, encoder, lb = load_model('model')
    X, _, _, _ = process_data(data, training=False, encoder=encoder, lb=lb)
    prediction = inference(model, X)
    assert prediction is not None


def test_generate_slices(data):
    slices = generate_slices(data, ["sex"])
    assert len(slices.keys()) == 2
    assert "sex => Female" in slices.keys()
    assert "sex => Male" in slices.keys()
