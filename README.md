
Network Anomaly Detection

This project aims to analyze network traffic data and identify potential anomalies or attack patterns using clustering techniques. 
The code uses the KMeans algorithm to categorize network traffic data into clusters, identifying unusual activity by assessing clusters 
with significantly smaller populations.

## Dependencies

The code requires the following Python libraries:
- **NumPy** (`numpy`) - for numerical operations
- **Pandas** (`pandas`) - for data handling and manipulation
- **Scikit-learn** (`sklearn`) - for preprocessing and clustering models
- **Matplotlib** (`matplotlib`) - for plotting data
- **Warnings** (`warnings`) - to suppress warnings for a clean output

Install all dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Data Overview

The data file (e.g., `Train.txt`) is a comma-separated file containing multiple features, such as:
- Network protocol and service types
- Source and destination bytes
- Error rates and rates of various network behaviors
- Attack labels

Each row represents a network traffic record, and the dataset includes both continuous and categorical variables. We have 43 columns, 
which represent both network features and a target variable, `attack`.

## Steps and Explanations

### 1. Load the Dataset

The dataset is loaded using Pandas, with specific column names provided to make the data more interpretable.

```python
col = ["duration","protocoltype","service",...,"lastflag"]
df = pd.read_csv('/path/to/Train.txt', sep=',', names=col)
```

### 2. Data Cleaning and Preprocessing

- **Drop Unnecessary Columns**: Four columns (`land`, `urgent`, `numfailedlogins`, `numoutboundcmds`) are dropped as they are either 
redundant or not useful for clustering.
  
- **Check for Missing Values**: `df.isna().sum()` is used to ensure no missing values are present in the dataset.

### 3. Encode Categorical Variables

The `LabelEncoder` is used to convert categorical features like `protocoltype`, `service`, `flag`, and `attack` into numerical values 
to ensure compatibility with the KMeans algorithm.

```python
le = LabelEncoder()
df['protocoltype'] = le.fit_transform(df['protocoltype'])
```

### 4. Determine Optimal Number of Clusters (Elbow Method)

The Elbow Method is used to identify the optimal number of clusters for KMeans by plotting the sum of squared distances (inertia) 
against a range of cluster numbers.

```python
K = range(1, 39)
a = [KMeans(n_clusters=i).fit(df).inertia_ for i in K]
plt.plot(K, a, marker='o')
```

### 5. Fit KMeans Model

Using the identified number of clusters (e.g., 5), the KMeans algorithm is fitted on the dataset to categorize data into clusters.

```python
kmeans = KMeans(n_clusters=5, random_state=111)
kmeans.fit(df)
```

### 6. Assign Cluster Labels

Each data point is assigned a cluster label, enabling us to see which data points belong to the same group. Points with unique or 
small cluster labels indicate possible anomalies.

```python
df['cluster_label'] = kmeans.fit_predict(df)
```

### 7. Analyze Cluster Centers for Feature Importance

The difference between cluster centroids reveals which features contribute the most to distinguishing clusters, indicating their 
importance in anomaly detection.

```python
centroid_diff = np.diff(kmeans.cluster_centers_, axis=0)
feature_importance = np.abs(centroid_diff).mean(axis=0)
```

### 8. Silhouette Score

The Silhouette Score measures the quality of clustering, with values close to 1 indicating well-defined clusters.

```python
metrics.silhouette_score(df, kmeans.labels_)
```

## Results and Insights

- **Elbow Plot**: Indicates the ideal number of clusters for separating normal and anomalous network behavior.
- **Cluster Analysis**: Identifies outliers, potentially malicious traffic, based on cluster labels.
- **Feature Importance**: Provides insight into which network features are most significant for distinguishing traffic types.

## Example Output


