import logging
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, model_performance, generate_slices, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info('Reading data')
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'census-cleaner.csv'))

logger.info('Splitting data')
train, test = train_test_split(data, test_size=0.20)

logger.info('Processing data')
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
target = "salary"
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target, training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label=target, training=False, encoder=encoder, lb=lb
)

logger.info('Training classifier')
fixed_params = {'use_label_encoder': False, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42}
model = train_model(X_train, y_train, fixed_params)

logger.info('Saving model')
save_model(model, os.path.join(os.path.dirname(__file__), 'xgboost-classifier.json'))

logger.info('Generating model performance for data slices')
with open("slice_output.txt", "w") as slice_output_file:
    for key, data_slice in generate_slices(test, cat_features).items():
        X_slice, y_slice, _, _ = process_data(data_slice, categorical_features=cat_features, label=target, training=False,
                                              encoder=encoder, lb=lb)
        metrics = model_performance(model, X_slice, y_slice)
        slice_output_file.write(f"{key} : Precision: {metrics['precision']}, Recall: {metrics['recall']}, FBeta: {metrics['fbeta']}\n")

logger.info('Analyzing model performance')
overall_metrics = model_performance(model, X_test, y_test)
logger.info(f"Overall test metrics:\n\tPrecision: {overall_metrics['precision']}\n\tRecall: {overall_metrics['recall']}\n\tFBeta: {overall_metrics['fbeta']}\n")
