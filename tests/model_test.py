import os.path
import sys
from pathlib import Path

import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.ml.data import load_data, process_data
from model.ml.model import save_model, load_model, model_performance, generate_slices


@pytest.fixture
def data():
    return load_data(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def model():
    X_train, y_train, encoder, lb = process_data(load_data(os.path.join(os.path.dirname(__file__), '..')))
    model = xgb.XGBClassifier(use_label_encoder=False).fit(X_train, y_train)
    return model


def test_save_and_load_model(model, tmpdir):
    model_path = os.path.join(tmpdir, 'model.json')
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    assert type(loaded_model) is xgb.Booster


def test_compute_model_performance(model):
    X_test, y_test, _, _ = process_data(load_data(os.path.join(os.path.dirname(__file__), '..')))
    metrics = model_performance(model, X_test, y_test)
    for metric in ["precision", "recall", "fbeta"]:
        assert 0 < metrics[metric] <= 1


def test_generate_slices(data):
    slices = generate_slices(data, ["sex"])
    assert len(slices.keys()) == 2
    assert "sex => Female" in slices.keys()
    assert "sex => Male" in slices.keys()
