
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
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Voting Classifier with multiple models
- XGBoost and LightGBM for efficient gradient boosting

Hyperparameter tuning with Optuna enhances the model's performance.

## Results and Analysis

Model outputs and evaluation metrics are provided to interpret performance. Accuracy, precision, and recall metrics help evaluate how well the model distinguishes normal traffic from anomalies.

## Usage

1. **Run Notebook**: Open and run each cell sequentially to reproduce the results.
2. **Modify Parameters**: You can adjust model parameters and retrain to improve performance.

## Conclusion

The notebook successfully identifies anomalies in network traffic. Future improvements could include feature engineering or testing additional algorithms and website deployment(work in progress, will upload after milestone 2)
