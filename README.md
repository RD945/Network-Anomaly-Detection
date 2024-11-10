
# Network Anomaly Detection By Reetam Dan E23CSEU0283 Bennett University

This project detects anomalies in network traffic using machine learning. It utilizes supervised learning algorithms to classify traffic as either normal or anomalous based on patterns in network data.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `optuna`, `lightgbm`, `xgboost`, `tabulate`
  - Install all libraries with:
    ```bash
    pip install numpy pandas seaborn matplotlib scikit-learn optuna lightgbm xgboost tabulate
    ```

## Data

The dataset consists of features representing network traffic attributes and a target variable (`class`) indicating whether the instance is "normal" or "anomaly".

## Data Preprocessing

1. **Loading Data**: Training and test data are loaded from `Train_data_1.csv` and `Test_data_1.csv`.
2. **Exploration**: The notebook examines dataset structure, feature types, and initial statistics to understand the data distribution.

## Model Building

The notebook implements several classifiers to detect anomalies, including:
- Logistic Regression
- Random Forest
- SVM
- KNN
- Decision Tree
- Naive Bayes

## Results and Analysis

Model outputs and evaluation metrics are provided to interpret performance. Accuracy, precision, recall, F1 Score and Average metrics help evaluate how well the model distinguishes normal traffic from anomalies.
Best performance is given by Random Forest


![image](https://github.com/user-attachments/assets/b23af083-926e-4d4b-bfe6-300fbe7ba81f)

![image](https://github.com/user-attachments/assets/693ac9c3-b428-46b8-83ba-317a607364b2)

![image](https://github.com/user-attachments/assets/4705e4d6-1e2f-4a77-8f06-18fd71678a5a)


## Conclusion

The notebook successfully identifies anomalies in network traffic. Future improvements could include feature engineering or testing additional algorithms like Adaboost, Gradient Boosting, Linear SVC, LightBGM, XGBoost and hyperparamter tuning with Optuna and website deployment(work in progress, will upload after milestone 2).


## Data Preprocessing

The Data Preprocessing code in detail explaining each part's purpose and how it prepares the dataset for machine learning:


Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

    Purpose: Load necessary libraries for data handling, visualization, machine learning model training, and data preprocessing.

1. Loading the Data

train = pd.read_csv('Train_data_1.csv')
test = pd.read_csv('Test_data_1.csv')

    Purpose: Load the training and test datasets from CSV files into Pandas DataFrames. The train dataset will be used to train the models, and test data will later be used for predictions.

2. Exploring the Data

train.head()
train.info()
train.describe()
train.describe(include='object')

    Purpose: These commands give an overview of the data:
        .head() shows the first few rows of the dataset.
        .info() provides information about the data types and the presence of null values.
        .describe() provides summary statistics for numeric columns.
        .describe(include='object') provides summary statistics for categorical columns.

3. Checking for Missing Values

train.isnull().sum()
total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")

    Purpose: This checks for missing values:
        train.isnull().sum() calculates the total number of missing values in each column.
        For columns with missing values, it calculates the percentage of missing values, which helps decide if they should be filled or removed.

4. Removing Duplicate Rows

print(f"Number of duplicate rows: {train.duplicated().sum()}")

    Purpose: Check for duplicate rows in the dataset. Removing duplicates is essential to ensure data integrity and avoid biased training.

5. Visualizing the Class Distribution

sns.countplot(x=train['class'])
print('Class distribution Training set:')
print(train['class'].value_counts())

    Purpose: sns.countplot visualizes the distribution of the target class in the training data to check for class imbalance, which may affect model performance.

6. Encoding Categorical Variables

def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)

    Purpose: This function encodes categorical variables into numeric form using LabelEncoder:
        It iterates over each column, checking if the data type is categorical (object).
        If it is, it applies LabelEncoder, transforming each category into a unique integer. This is necessary because machine learning models typically require numerical data.

7. Dropping Unnecessary Columns

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    Purpose: Remove columns with little to no relevance or variance (num_outbound_cmds in this case), which can simplify the model and reduce noise in the data.

8. Splitting Features and Target

X_train = train.drop(['class'], axis=1)
Y_train = train['class']

    Purpose: Separate features (X_train) and the target variable (Y_train). This split is necessary for model training, where features are used to predict the target.

9. Feature Selection

rfc = RandomForestClassifier()
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

    Purpose: Use Recursive Feature Elimination (RFE) with a RandomForestClassifier to select the 10 most important features based on predictive power.
        RFE recursively removes less important features and retrains the model until only the specified number of features remains.
        This can improve model performance by reducing overfitting and focusing on the most relevant data.

10. Scaling the Data

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)

    Purpose: Apply standard scaling to the data:
        StandardScaler standardizes features by removing the mean and scaling to unit variance.
        Scaling is important because many machine learning algorithms (like SVM and KNN) are sensitive to feature magnitudes, and scaling brings all features to the same level.

11. Train-Test Split

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

    Purpose: Split the preprocessed training data into x_train, x_test, y_train, and y_test sets with a 70-30 train-test ratio:
        This split allows evaluating the model’s performance on unseen data (x_test and y_test) to assess generalization.

Summary

The preprocessing steps ensure that the data is clean, appropriately encoded, scaled, and ready for model training. These steps include:

    Loading and inspecting the data.
    Checking for and handling missing values.
    Encoding categorical variables.
    Dropping irrelevant features.
    Selecting the most informative features.
    Scaling the features.
    Splitting the data for training and testing.

Each step contributes to preparing the dataset for training various machine learning models. This setup can significantly improve model performance and accuracy.
