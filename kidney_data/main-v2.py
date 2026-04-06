import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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

## Variables
test_size_percentage = 0.20
random_state_number = 42

## Load the dataset from a CSV file
## kidney dataset can be found here - https://www.kaggle.com/datasets/priyankabarik/chronic-kidney-disease-ckd-clinical-dataset
data = pd.read_csv('kidney_data/Training_CKD_dataset.csv')
# new_data = pd.read_csv('kidney_data/Testing_CKD_dataset.csv')

## Print what data looks like
print("#"*50)
print("\n###### Preview of csv ######\n")
print(data.head())
print("#"*50)

## Print all columns and types
print("#"*50)
print("\n###### Describe dataset and data type ######\n")
data.info()
print("#"*50)

print("#"*50)
print("\n###### Print occurrences of null values ######\n")
null_counts = data.isnull().sum()
null_counts = null_counts[null_counts > 0]
if null_counts.empty:
    print("No missing values found")
else:
    print(null_counts)
print("#"*50)

## Separate features (X) and target variable (y)
## Think of like flash cards
## X = picture/question
## Y = answer
X = data.drop(['Target'], axis=1)
y = data['Target']

print("#"*50)
print("\n###### Print applicable Target values ######\n")
print(y.unique().tolist())
print("#"*50)

# Define numerical and categorical columns
numerical_cols = [
    'Age', 'Gender', 'BMI', 'Systolic_BP', 'Diastolic_BP',
    'Heart_Rate', 'Serum_Creatinine', 'Blood_Urea_Nitrogen', 'eGFR',
    'Urine_Albumin', 'Urine_Protein', 'Albumin_Creatinine_Ratio',
    'Urine_Specific_Gravity', 'Sodium', 'Potassium', 'Calcium',
    'Phosphorus', 'Chloride', 'Bicarbonate', 'Hemoglobin',
    'RBC_Count', 'WBC_Count', 'Platelet_Count', 'Packed_Cell_Volume',
    'Blood_Glucose_Random', 'Fasting_Glucose', 'HbA1c', 'Cholesterol',
    'Triglycerides', 'Serum_Albumin', 'Total_Protein',
]

categorical_cols = ['Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney']

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

# # Final pipeline that includes preprocessing and the machine learning model
# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', RandomForestClassifier(random_state=random_state_number))
# ])

classifiers = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(random_state=random_state_number),
    RandomForestClassifier(random_state=random_state_number),
    KNeighborsClassifier(n_neighbors=5),
    SVC()
]
for classifier in classifiers:
    print("#"*50)
    classifier_name_pipeline_name = type(classifier).__name__.lower() + "_pipeline"
    print(classifier_name_pipeline_name)
    classifier_name_pipeline_name = Pipeline([
        ('preprocessor', preprocessor),
        ('model', classifier)
    ])

    # Split dataset into training and testing sets
    ## test_size=0.02 --> means 20% of the data is held out for testing, and 80% is used for training.
    ## random_state=42 --> makes the split reproducible, so you get the same train/test split each time you run the script.
    ## X_train, y_train for training the model
    ## X_test, y_test for testing the mode
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage, random_state=random_state_number)
    classifier_name_pipeline_name.fit(X_train, y_train) ## train pipeline
    y_pred = classifier_name_pipeline_name.predict(X_test) ## use trained pipeline to predict

# ## Print the shape of training and testing datasets for verification
# print("#"*50)
# print(f'\nTotal data shape: {data.shape}')
# print(f'\nTraining on {(1 - test_size_percentage):.2%} of data.')
# print(f'Training data shape: {X_train.shape}')
# print(f'\nTesting on {test_size_percentage:.2%} of data.')
# print(f'Testing data shape: {X_test.shape}\n')
# print('Testing data preview:')
# print(pd.DataFrame(X_train).head())
# print("#"*50)

    ## Get the names of the encoded categorical columns
    ## This is useful for interpreting the transformed feature set
    ## Columns that dont have numerical values, get encoded into several columns
    ## Ex - 'Diabetes' column gets transformed into 'Diabetes_Yes' and 'Diabetes_No'
    # print("#"*50)
    # print("\nPrinting encoded categorical columns:\n")
    encoded_columns = classifier_name_pipeline_name.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    # print(encoded_columns.tolist())
    # print("#"*50)

    ## Display importance of each variable
    # model = classifier_name_pipeline_name.named_steps['model']
    # all_feature_names = numerical_cols + encoded_columns.tolist()
    # importances = model.feature_importances_

    # feature_importance_df = pd.DataFrame({
    #     'feature': all_feature_names,
    #     'importance': importances
    # }).sort_values(by='importance', ascending=False)

    # print("#"*50)
    # print("\nPrinting Parameters by importance level:\n")
    # print(feature_importance_df)
    # print("#"*50)

    ## Display results
    # print("#"*50)
    print("\nModel Performance Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
    # print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
    # print(classification_report(y_test, y_pred))
    print("#"*50)

# joblib.dump(pipeline, 'kidney_pipeline.joblib')

# ## Create heat map
# # class_labels = pipeline.named_steps['model'].classes_
# # cm = confusion_matrix(y_test, y_pred, labels=class_labels)

# # plt.figure(figsize=(10, 7))
# # sns.heatmap(
# #     cm,
# #     annot=True,
# #     fmt='d',
# #     cmap='Blues',
# #     xticklabels=class_labels,
# #     yticklabels=class_labels
# # )
# # plt.title('Confusion Matrix for Kidney Status Predictions')
# # plt.xlabel('Predicted Kidney Status')
# # plt.ylabel('Actual Kidney Status')
# # plt.tight_layout()
# # plt.show()

# new_data = pd.read_csv('kidney_data/Testing_CKD_dataset.csv')

# X_new = new_data.drop(['Target'], axis=1)
# y_true = new_data['Target']

# model = joblib.load('kidney_pipeline.joblib')
# # predictions = pipeline.predict(X_new)
# predictions = model.predict(X_new)


# ## Print prediction accuracy
# print("#"*50)
# print("\nModel Performance Metrics:")
# print(f'\nTotal data shape: {new_data.shape}')
# accuracy = accuracy_score(y_true, predictions)
# print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
# print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
# print(classification_report(y_true, predictions))
# print("#"*50)


# ### Interpreting other variables
# family_history_yes = data[data['Family_History_Kidney'] == 'Yes']
# ckd_cases = family_history_yes[family_history_yes['Target'] != 'Healthy Kidney']
# percentage = (len(ckd_cases) / len(family_history_yes)) * 100
# print(f"Percentage with family history who had CKD: {percentage:.2f}%")

# family_history_no = data[data['Family_History_Kidney'] == 'No']
# ckd_cases_no = family_history_no[family_history_no['Target'] != 'Healthy Kidney']
# percentage_no = (len(ckd_cases_no) / len(family_history_no)) * 100
# print(f"Percentage with family history: {percentage:.2f}%")
# print(f"Percentage without family history: {percentage_no:.2f}%")

# print("#"*50)
# print("#"*50)
# print("#"*50)
# print("#"*50)
# print("#"*50)
# print("#"*50)


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

# logistic_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', LogisticRegression(max_iter=1000))
# ])

# svm_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', SVC())
# ])

# tree_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', DecisionTreeClassifier(random_state=42))
# ])

# kneighbor_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', KNeighborsClassifier(n_neighbors=5))
# ])

# # Training the models
# logistic_pipeline.fit(X_train, y_train)
# svm_pipeline.fit(X_train, y_train)
# tree_pipeline.fit(X_train, y_train)
# kneighbor_pipeline.fit(X_train, y_train)

# # Making predictions with each model
# log_reg_preds = logistic_pipeline.predict(X_test)
# svm_preds = svm_pipeline.predict(X_test)
# tree_preds = tree_pipeline.predict(X_test)
# kneighbor_preds = kneighbor_pipeline.predict(X_test)

# # Store model predictions in a dictionary
# # this makes it easier to iterate through each model
# # and print the results. 
# model_preds = {
#     "Logistic Regression": log_reg_preds,
#     "Support Vector Machine": svm_preds,
#     "Decision Tree": tree_preds,
#     "KNeighbors": kneighbor_preds
# }

# for model, preds in model_preds.items():
#     accuracy = accuracy_score(y_test, preds)
#     print(f"{model} Accuracy: {accuracy:.4f}") 
#     print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
