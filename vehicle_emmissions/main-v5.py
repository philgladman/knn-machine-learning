import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

## Use when target is a label (ie. Kidney Stage = "Healthy Kidney" or "Severe CKD (Stage 4)")
## These are considered "Classifier" estimator, they predict categories
from sklearn.linear_model import LogisticRegression ## Good for linear models, when relationship is fairly simple, small data
from sklearn.tree import DecisionTreeClassifier ## Good for non linear models, and feature interactions
from sklearn.ensemble import RandomForestClassifier ## Good for non linear models, and feature interactions
from sklearn.neighbors import KNeighborsClassifier ## Predict based on nearby training examples
from sklearn.svm import SVC ## Try to separate classes with an optimal boundary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## Use when target is a number (ie. CO2 Emissions = 100 or 250)
## These are considered "Regressors", they predict numbers
from sklearn.linear_model import LinearRegression ## Good for linear models, when relationship is fairly simple, small data
from sklearn.tree import DecisionTreeRegressor ## Good for non linear models, and feature interactions
from sklearn.ensemble import RandomForestRegressor ## Good for non linear models, and feature interactions
from sklearn.neighbors import KNeighborsRegressor ## Predict based on nearby training examples
from sklearn.svm import SVR ## Try to separate classes with an optimal boundary
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## Variables
start_time = time.perf_counter()
test_size_percentage = 0.20
max_iter_number = 1000
random_state_number = 42
n_neighbors_number = 5
working_dir = Path(__file__).resolve().parent
training_data_file = working_dir / "vehicle_emmissions.csv"
target_column_name = 'CO2_Emissions'
enable_pagination = False

## Load the dataset from a CSV file
data = pd.read_csv(training_data_file)

target_data_type = data.dtypes[target_column_name]
print("#"*50 + "\n")
if str(target_data_type) in ["str", "object"]:
    print("target data type is categorical\n")
    data_is_categorical = True
    data_is_numerical = False
    job_directory_name = working_dir / "categorical"
elif str(target_data_type) in ["int64", "float64"]:
    print("target data type is numerical\n")
    data_is_categorical= False
    data_is_numerical = True
    job_directory_name = working_dir / "numerical"
else:
    print("ERROR - Undefined data type")

def printer(header, print_data):
    print("#"*50)
    print(f"\n###### {header} ######\n")
    print(print_data)
    print("#"*50)

def paginator():
    if enable_pagination:
        input("Press Enter to continue...")

def preview_data(data_set):
    """Function to preview what the data looks like"""

    ## Print first couple lines of csv
    printer("Preview of csv", data_set.head())
    paginator()

    ## Print all columns and types
    printer("Describe dataset and data type", data_set.info())
    paginator()

    null_counts = data_set.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        printer("Print occurrences of null values", "No missing values found")
    else:
        printer("Print occurrences of null values", null_counts)
    paginator()

preview_data(data)

# Data Cleaning and Feature Engineering 
def preprocess_data(df):
    ## Drop columns that should have zero corelation with survival
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    ## Fill in any null values with averages / most frequents
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].mode()[0])

    ## Feature Engineering / AKA create new features from existing data
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    ## Split Fare into 4 equal sized groups
    ## Labels=false means set labels to integers such as 0,1,2,3
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    ## Split Age into 5 fixed numeric ranges 0-12, 12-20, etc.
    ## Labels=false means set labels to integers such as 0,1,2,3,4
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf],labels=False)

    return df

## Separate features (X) and target variable (y)
## Think of like flash cards
## X = picture/question
## Y = answer

# data = preprocess_data(data)
X = data.drop([target_column_name], axis=1)
y = data[target_column_name]

printer("Print applicable Target values", y.unique().tolist())
paginator()

numerical_cols = []
categorical_cols = []

for column_name, data_type in X.dtypes.items():
    if str(data_type) in ["str", "object"]:
        categorical_cols.append(column_name)
    elif str(data_type) in ["int64", "float64"]:
        numerical_cols.append(column_name)
    else:
        print("ERROR - DATA TYPE NOT str, object, int64, or float64")
        break

printer("Print Numerical and Categorical", (
        "Numerical\n"
        f"{numerical_cols}\n\n"
        "Categorical\n"
        f"{categorical_cols}"
    ))
paginator()

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

categorical_classifiers = [
    LogisticRegression(max_iter=max_iter_number),
    DecisionTreeClassifier(random_state=random_state_number),
    RandomForestClassifier(random_state=random_state_number),
    KNeighborsClassifier(n_neighbors=n_neighbors_number),
    SVC()
]

numerical_classifiers = [
    LinearRegression(),
    DecisionTreeRegressor(random_state=random_state_number),
    RandomForestRegressor(random_state=random_state_number),
    KNeighborsRegressor(n_neighbors=n_neighbors_number),
    SVR()
]

if data_is_categorical:
    classifiers = categorical_classifiers
elif numerical_classifiers:
    classifiers = numerical_classifiers
else:
    print("ERROR - Undefined data type")

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

paginator()

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
    encoded_columns = pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat']['encoder'] \
        .get_feature_names_out(categorical_cols)

    # printer("\nPrinting encoded categorical columns:\n", encoded_columns.tolist())

    # Display importance of each variable
    model = pipeline.named_steps['model']
    transformed_feature_names = numerical_cols + encoded_columns.tolist()
    original_feature_names = X_test.columns.tolist()

    ## Display results
    print("\nModel Performance Metrics:")
    if data_is_categorical:
        classifiers = categorical_classifiers
    elif numerical_classifiers:
        # print("data_is_numerical")
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)  # Root mean squared error
        # mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"R² Score: {r2:.4f}")  # Higher is better; 1.0 indicates perfect prediction
        # print(f"Root Mean Squared Error: {rmse:.2f}")  # Lower is better
        # print(f"Mean Absolute Error: {mae:.2f}")  # Lower is better
    else:
        # print("data_is_categorical")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}") ## Higher is better; 1.0 indicates perfect prediction
        # print("\nClassification Report:") ## precision, recall, and F1-score for each class -  ## Higher is better; 1.0 indicates perfect prediction
        # print(classification_report(y_test, y_pred))

    # os.makedirs(job_directory_name, exist_ok=True)
    # pipeline_file_name = job_directory_name / (classifier_name_pipeline_name + ".joblib")
    # joblib.dump(pipeline, pipeline_file_name)

# new_data = pd.read_csv(testing_data_file)

# X_new = new_data.drop([target_column_name], axis=1)
# y_true = new_data[target_column_name]

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


# ### Interpreting other variables
# males = data[data['Sex'] == 'male']
# surviving_males = males[males['Survived'] == 1]
# print(len(surviving_males))

# females = data[data['Sex'] == 'female']
# dead_females = females[females['Survived'] == 0]
# print(len(dead_females))

# male_survivors = data[(data['Sex'] == 'male') & (data['Survived'] == 1)]
# print(len(male_survivors))

end_time = time.perf_counter()
printer("Script Run time", (f"Runtime: {end_time - start_time:.4f} seconds"))
