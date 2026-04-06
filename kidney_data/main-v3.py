import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


## Use when target is a number (ie. CO2 Emissions = 100 or 250)
## These are considered "Regressors", they predict numbers
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## Use when target is a label (ie. Kidney Stage = "Healthy Kidney" or "Severe CKD (Stage 4)")
## These are considered "Classifier" estimator, they predict categories
from sklearn.linear_model import LogisticRegression ## Good for linear models, when relationship is fairly simple, small data
from sklearn.tree import DecisionTreeClassifier ## Good for non linear models, and feature interactions
from sklearn.ensemble import RandomForestClassifier ## Good for non linear models, and feature interactions
from sklearn.neighbors import KNeighborsClassifier ## Predict based on nearby training examples
from sklearn.svm import SVC ## Try to separate classes with an optimal boundary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import joblib

########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
## Next Steps
## - currently script runs with mutliple models for categorical data
## - Also self sorts categorical and numerical columns for pipeline
## - Next step is to add in a target that is numerical like titanic or emissions
## - Then have logic for script to check if target is numerical, then run numerical pipeline
##    - IF categorical, run categorical pipeline
########################################################
########################################################
########################################################
########################################################
########################################################

## Variables
test_size_percentage = 0.20
random_state_number = 42
target_column_name = 'Target'
working_dir = Path(__file__).resolve().parent
training_data_file = working_dir / "Training_CKD_dataset.csv"
testing_data_file = working_dir / "Testing_CKD_dataset.csv"

## Load the dataset from a CSV file
## kidney dataset can be found at link below
## https://www.kaggle.com/datasets/priyankabarik/chronic-kidney-disease-ckd-clinical-dataset
data = pd.read_csv(training_data_file)

target_data_type = data.dtypes[target_column_name]
if str(target_data_type) in ["str", "object"]:
    print("target data type is categorical")
    data_is_categorical = True
    data_is_numerical = False
    job_directory_name = working_dir / "categorical"
elif str(target_data_type) in ["int64", "float64"]:
    print("target data type is numerical")
    data_is_categorical= True
    data_is_numerical = False
    job_directory_name = working_dir / "numerical"

def preview_data(data_set):
    """Function to preview what the data looks like"""

    ## Print first couple lines of csv
    print("#"*50)
    print("\n###### Preview of csv ######\n")
    print(data_set.head())
    print("#"*50)

    ## Print all columns and types
    print("#"*50)
    print("\n###### Describe dataset and data type ######\n")
    data_set.info()
    print("#"*50)

    print("#"*50)
    print("\n###### Print occurrences of null values ######\n")
    null_counts = data_set.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("No missing values found")
    else:
        print(null_counts)
    print("#"*50)

preview_data(data)

## Separate features (X) and target variable (y)
## Think of like flash cards
## X = picture/question
## Y = answer
X = data.drop([target_column_name], axis=1)
y = data[target_column_name]

print("#"*50)
print("\n###### Print applicable Target values ######\n")
for value in y.unique().tolist():
    print(value)
print("#"*50)

numerical_cols = []
categorical_cols = []

for column_name, data_type in X.dtypes.items():
    # print("##############")
    # print(f"column_name: {column_name}, data_type: {data_type}")
    if str(data_type) in ["str", "object"]:
        categorical_cols.append(column_name)
    elif str(data_type) in ["int64", "float64"]:
        numerical_cols.append(column_name)
    else:
        print("ERROR - DATA TYPE NOT str, object, int64, or float64")
        break
    # print("##############")

print("#"*50)
print("\n###### Print Numerical and Categorical ######\n")
print("Numerical")
print(numerical_cols)
print("\nCategorical")
print(categorical_cols)
print("#"*50)

# Pipeline for preprocessing numerical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for preprocessing categorical data
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols), 
    ('cat', categorical_pipeline, categorical_cols) 
])

classifiers = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(random_state=random_state_number),
    RandomForestClassifier(random_state=random_state_number),
    KNeighborsClassifier(n_neighbors=5),
    SVC()
]

# Split dataset into training and testing sets
## test_size=0.02 --> means 20% of the data is held out for testing, and 80% is used for training.
## random_state=42 --> makes the split reproducible, so you get the same train/test split each time you run the script.
## X_train, y_train for training the model
## X_test, y_test for testing the mode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage, random_state=random_state_number)

print("#"*50)
print(f'\nTotal data shape: {data.shape}')
print(f'\nTraining on {(1 - test_size_percentage):.2%} of data.')
print(f'Training data shape: {X_train.shape}')
print(f'\nTesting on {test_size_percentage:.2%} of data.')
print(f'Testing data shape: {X_test.shape}\n')
print('Testing data preview:')
print(pd.DataFrame(X_train).head())
print("#"*50)

trained_pipelines = {}
for classifier in classifiers:
    print("#"*50)
    classifier_name_pipeline_name = type(classifier).__name__.lower() + "_pipeline"
    print(classifier_name_pipeline_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', classifier)
    ])

    pipeline.fit(X_train, y_train) ## train pipeline
    y_pred = pipeline.predict(X_test) ## use trained pipeline to predict
    trained_pipelines[classifier_name_pipeline_name] = pipeline

    ## Get the names of the encoded categorical columns
    ## This is useful for interpreting the transformed feature set
    ## Columns that dont have numerical values, get encoded into several columns
    ## Ex - 'Diabetes' column gets transformed into 'Diabetes_Yes' and 'Diabetes_No'
    # print("#"*50)
    # print("\nPrinting encoded categorical columns:\n")
    encoded_columns = pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat']['encoder'] \
        .get_feature_names_out(categorical_cols)
    # print(encoded_columns.tolist())
    # print("#"*50)

    ## Display importance of each variable
    # model = pipeline.named_steps['model']
    # transformed_feature_names = numerical_cols + encoded_columns.tolist()
    # original_feature_names = X_test.columns.tolist()

    # print("#"*50)
    # print("\nPrinting Parameters by importance level:\n")
    # if isinstance(model, LogisticRegression):
    #     coefficients = model.coef_

    #     if coefficients.shape[0] == 1:
    #         importance_values = np.abs(coefficients[0])
    #     else:
    #         importance_values = np.abs(coefficients).mean(axis=0)

    #     importance_df = pd.DataFrame({
    #         'feature': transformed_feature_names,
    #         'importance': importance_values
    #     }).sort_values(by='importance', ascending=False)

    #     print("Method: absolute value of coefficients")
    #     print(importance_df.head(5))

    # elif isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
    #     importance_df = pd.DataFrame({
    #         'feature': transformed_feature_names,
    #         'importance': model.feature_importances_
    #     }).sort_values(by='importance', ascending=False)

    #     print("Method: feature_importances_")
    #     print(importance_df.head(5))

    # elif isinstance(model, (KNeighborsClassifier, SVC)):
    #     result = permutation_importance(
    #         pipeline,
    #         X_test,
    #         y_test,
    #         n_repeats=10,
    #         random_state=42,
    #         scoring='accuracy'
    #     )

    #     importance_df = pd.DataFrame({
    #         'feature': original_feature_names,
    #         'importance': result.importances_mean,
    #         'importance_std': result.importances_std
    #     }).sort_values(by='importance', ascending=False)

    #     print("Method: permutation importance")
    #     print(importance_df.head(5))

    # else:
    #     print("No importance logic defined for this model.")


    ## Display results
    print("\nModel Performance Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
    # print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
    # print(classification_report(y_test, y_pred))
    print("#"*50)

    # os.makedirs(job_directory_name, exist_ok=True)
    # pipeline_file_name = job_directory_name / (classifier_name_pipeline_name + ".joblib")
    # joblib.dump(pipeline, pipeline_file_name)

new_data = pd.read_csv(testing_data_file)

X_new = new_data.drop([target_column_name], axis=1)
y_true = new_data[target_column_name]

# for file in os.listdir(job_directory_name):
#     print("#"*50)
#     print(file)
#     model = joblib.load(file)
#     predictions = model.predict(X_new)

#     ## Print prediction accuracy
#     print("\nModel Performance Metrics:")
#     print(f'\nTotal data shape: {new_data.shape}')
#     accuracy = accuracy_score(y_true, predictions)
#     print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
#     # print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
#     # print(classification_report(y_true, predictions))
# #     print("#"*50)

# for pipeline_name, pipeline in trained_pipelines.items():
#     print("#"*50)
#     print(pipeline_name)
#     predictions = pipeline.predict(X_new)

#     ## Print prediction accuracy
#     print("\nModel Performance Metrics:")
#     # print(f'\nTotal data shape: {new_data.shape}')
#     accuracy = accuracy_score(y_true, predictions)
#     print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
#     # print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
#     # print(classification_report(y_true, predictions))
#     print("#"*50)
#     print("#"*50)
