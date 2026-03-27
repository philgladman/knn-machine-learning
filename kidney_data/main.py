import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

## Use when target is a number (ie. CO2 Emissions = 100 or 250)
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## Use when target is a label (ie. Kidney Stage = "Healthy Kidney" or "Severe CKD (Stage 4)")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import joblib

## Variables
test_size_percentage = 0.20
random_state_number = 42

## Load the dataset from a CSV file
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

# Final pipeline that includes preprocessing and the machine learning model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=random_state_number))
])

# Split dataset into training and testing sets
## test_size=0.02 --> means 20% of the data is held out for testing, and 80% is used for training.
## random_state=42 --> makes the split reproducible, so you get the same train/test split each time you run the script.
## X_train, y_train for training the model
## X_test, y_test for testing the mode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage, random_state=random_state_number)
pipeline.fit(X_train, y_train) ## train pipeline
y_pred = pipeline.predict(X_test) ## use trained pipeline to predict

## Print the shape of training and testing datasets for verification
print("#"*50)
print(f'\nTotal data shape: {data.shape}')
print(f'\nTraining on {(1 - test_size_percentage):.2%} of data.')
print(f'Training data shape: {X_train.shape}')
print(f'\nTesting on {test_size_percentage:.2%} of data.')
print(f'Testing data shape: {X_test.shape}\n')
print('Testing data preview:')
print(pd.DataFrame(X_train).head())
print("#"*50)

## Get the names of the encoded categorical columns
## This is useful for interpreting the transformed feature set
## Columns that dont have numerical values, get encoded into several columns
## Ex - 'Diabetes' column gets transformed into 'Diabetes_Yes' and 'Diabetes_No'
print("#"*50)
print("\nPrinting encoded categorical columns:\n")
encoded_columns = pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
print(encoded_columns.tolist())
print("#"*50)

print("#"*50)
print("\nModel Performance Metrics:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
print(classification_report(y_test, y_pred))
print("#"*50)

joblib.dump(pipeline, 'kidney_pipeline.joblib')

## Create heat map
# class_labels = pipeline.named_steps['model'].classes_
# cm = confusion_matrix(y_test, y_pred, labels=class_labels)

# plt.figure(figsize=(10, 7))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=class_labels,
#     yticklabels=class_labels
# )
# plt.title('Confusion Matrix for Kidney Status Predictions')
# plt.xlabel('Predicted Kidney Status')
# plt.ylabel('Actual Kidney Status')
# plt.tight_layout()
# plt.show()

new_data = pd.read_csv('kidney_data/Testing_CKD_dataset.csv')

X_new = new_data.drop(['Target'], axis=1)
y_true = new_data['Target']

model = joblib.load('kidney_pipeline.joblib')
# predictions = pipeline.predict(X_new)
predictions = model.predict(X_new)


## Print prediction accuracy
print("#"*50)
print("\nModel Performance Metrics:")
print(f'\nTotal data shape: {new_data.shape}')
accuracy = accuracy_score(y_true, predictions)
print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
print(classification_report(y_true, predictions))
print("#"*50)
