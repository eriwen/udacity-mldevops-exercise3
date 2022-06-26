import json
import sys
from fastapi.testclient import TestClient
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app, negative_sample, positive_sample

client = TestClient(app)


def test_get_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "Hello World"}


def test_post_predict_negative():
    res = client.post("/predict", data=json.dumps(negative_sample))
    assert res.status_code == 200
    assert res.json().get("prediction") == "<=50K"


def test_post_predict_positive():
    res = client.post("/predict", data=json.dumps(positive_sample))
    assert res.status_code == 200
    assert res.json().get("prediction") == ">50K"


def test_post_predict_malformed_data():
    res = client.post("/predict", {})
    assert res.status_code != 200
