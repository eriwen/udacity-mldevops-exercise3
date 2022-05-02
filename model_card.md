# Model Card

A binary classifier which predicts whether a person earns more than US$50000 based on US Census data.

This model was developed to complete the [Udacity Machine Learning DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

## Model Details

This model uses an [XGBoost](https://xgboost.readthedocs.io/en/stable/) Classifier. The best hyperparameters were found using Grid Search:
 * eta = 0.1
 * max_depth = 4
 * n_estimators = 350

The latest model version was last trained by Eric Wendelin in April 2022.

## Intended Use

The model is intended to predict whether a person earns more than US$50000 using US Census data such as age, marital status, level of education, and gender.

## Training Data

[This Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) was used for training.

Data was cleaned by removing rows having missing values.
All provided non-target variables were used for training.
Categorical features were encoded using OneHotEncoder from scikit-learn. 
The target variable, salary, was encoded using a LabelBinarizer from scikit-learn.

80% of the US Census data set was sampled at random for training.

XGBoost was configured to optimize the AUC metric

## Evaluation Data

20% of the US Census data set was sampled at random for testing.
The same encoder and label binarizer used for model training must be provided for inference. 

## Metrics

The overall performance of the model on the test data set:

 * Precision: 0.78196
 * Recall: 0.65574
 * F1 Score: 0.71331

Performance for all subsets of test data partitioned by all possible categories and their values can be found in [this file](TODO).

## Ethical Considerations

The training data set was not analyzed for population bias, nor was the resulting model after training. 
This model should not be used to identify individuals, or report any bias with regard to any individual's salary. 


## Caveats and Recommendations

None