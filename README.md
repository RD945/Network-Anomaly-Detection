# Network-Anomaly-Detection by Reetam Dan


# Network Anomaly Detection

## Overview

This project aims to detect anomalies in network traffic data using machine learning techniques. Network anomalies can indicate potential security threats or unusual behavior that may need further investigation.

## Features

- **Data Preprocessing**: Cleaning and preparing network traffic data for analysis.
- **Model Training**: Training a machine learning model to classify normal vs. anomalous network activity.
- **Anomaly Detection**: Real-time or batch detection of anomalies in new network traffic data.
- **Performance Evaluation**: Assessing the model's accuracy, precision, recall, and other relevant metrics.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/network-anomaly-detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd network-anomaly-detection
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**:
   - Run the preprocessing script to prepare the network data for training.
   - Ensure the data is saved in the expected format (specify format if needed).

2. **Training the Model**:
   - Execute the training script to train the anomaly detection model.
   - You can adjust hyperparameters in the script or through command-line arguments (explain how if relevant).

3. **Anomaly Detection**:
   - Use the trained model to perform anomaly detection on new network data.

   ```bash
   python detect_anomalies.py --input <input_data_file>
   ```

## Project Structure

- `data/`: Contains the raw and processed network traffic data.
- `src/`: Source code for preprocessing, model training, and anomaly detection.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `models/`: Stores trained model files.
- `results/`: Contains evaluation metrics and visualizations.

## LIST OF COLUMNS FOR THE DATA SET:

-["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
-"wrong_fragment","urgent","hot","num_failed_logins","logged_in",
-"num_compromised","root_shell","su_attempted","num_root","num_file_creations",
-"num_shells","num_access_files","num_outbound_cmds","is_host_login",
-"is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
-"rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
-"dst_host_diff_srv_rate","dst_host_same_src_port_rate",
-"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
-"dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]

## BASIC FEATURES OF EACH NETWORK CONNECTION VECTOR:

-1 Duration: Length of time duration of the connection
-2 Protocol_type: Protocol used in the connection
-3 Service: Destination network service used
-4 Flag: Status of the connection – Normal or Error
-5 Src_bytes: Number of data bytes transferred from source to destination in single connection
-6 Dst_bytes: Number of data bytes transferred from destination to source in single connection
-7 Land: if source and destination IP addresses and port numbers are equal then, this variable takes value 1
else 0
-8 Wrong_fragment: Total number of wrong fragments in this connection
-9 Urgent: Number of urgent packets in this connection. Urgent packets are packets with the urgent bit
Activated
