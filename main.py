from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from model.ml.data import get_categorical_features, process_data
from model.ml.model import load_model, inference

negative_sample = {
    "age": 38,
    "workclass": "Private",
    "fnlgt": 215646,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Divorced",
    "occupation": "Handlers-cleaners",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

positive_sample = {
    "age": 49,
    "workclass": "Local-gov",
    "fnlgt": 268234,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Protective-serv",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}


class FeatureSet(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {"negative": negative_sample, "positive": positive_sample}


app = FastAPI()
model, encoder, lb = load_model('model')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(features: FeatureSet):
    df = pd.DataFrame({k: v for k, v in features.dict().items()}, index=[0])
    cat_features = [f.replace('-', '_') for f in get_categorical_features()]
    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, label=None, encoder=encoder, lb=lb)
    prediction = lb.inverse_transform(inference(model, X)).tolist()[0]
    return {"prediction": prediction}
