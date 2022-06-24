import json
import requests

get_response = requests.get('https://fathomless-wildwood-07611.herokuapp.com/')
print(get_response.status_code)
print(get_response.json())

sample = {
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
prediction_response = requests.post('https://fathomless-wildwood-07611.herokuapp.com/predict', data=json.dumps(sample))
print(prediction_response.status_code)
print(prediction_response.json())