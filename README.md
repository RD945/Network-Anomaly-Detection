
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

Model outputs and evaluation metrics are provided to interpret performance. Accuracy, precision, and recall metrics help evaluate how well the model distinguishes normal traffic from anomalies.
Best performance is given by Random Forest

![image](https://github.com/user-attachments/assets/b23af083-926e-4d4b-bfe6-300fbe7ba81f)

## Conclusion

The notebook successfully identifies anomalies in network traffic. Future improvements could include feature engineering or testing additional algorithms and website deployment(work in progress, will upload after milestone 2)
