## UCI HAR
### Dataset Introduction: UCI HAR Dataset
The UCI HAR (Human Activity Recognition) dataset is a public dataset for human activity recognition research, collected and published in 2012 by Davide Anguita and colleagues from the University of Genoa, Italy, and hosted on the UCI Machine Learning Repository. It records motion data from 30 participants aged 19 to 48 performing daily activities, captured using a Samsung Galaxy SII smartphone’s accelerometer and gyroscope sensors. The dataset is widely used for time-series data classification and machine learning research. [](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)[](https://archive.ics.uci.edu/ml/datasets/Human%2BActivity%2BRecognition%2BUsing%2BSmartphones)

### Dataset Overview
- **Purpose**: To develop and test human activity recognition algorithms, particularly for time-series data classification tasks using sensor data.
- **Activity Categories**: The dataset includes 6 basic activities:
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying
- **Participants**: 30 participants, with data split into a training set (70%, 21 participants, 7,352 samples) and a test set (30%, 9 participants, 2,947 samples), with no participant overlap.
- **Sensors**: Data collected from a waist-mounted smartphone, including:
  - Triaxial accelerometer (x, y, z directions, capturing gravity and body motion components)
  - Triaxial gyroscope (angular velocity in x, y, z directions)
- **Data Sampling**:
  - Sampling frequency: 50 Hz (50 samples per second).
  - Data sampled using a 2.56-second fixed-width sliding window (128 readings per window) with 50% overlap.
  - Raw sensor signals preprocessed with a Butterworth low-pass filter to separate body acceleration and gravity components.
- **Features**:
  - Raw dataset includes 9 time-series signals (3 axes each for accelerometer and gyroscope, plus 3 axes for total acceleration).
  - A preprocessed feature set with 561 feature vectors is provided, derived through feature engineering (e.g., mean, standard deviation, frequency-domain features).
- **Data Scale**:
  - Training set: 7,352 samples
  - Test set: 2,947 samples
  - Total: 10,299 samples
- **License**: Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), freely usable and shareable with proper attribution. [](https://medium.com/data-science/a-guide-to-time-series-sensor-data-classification-using-uci-har-data-b7ac4f6ad251)

### Dataset Structure
The dataset is provided as a compressed file (~58 MB), containing the following key files and directories after decompression:
- **train/**: Training set data
  - `X_train.txt`: Training feature data (7,352 rows, 561 columns of features).
  - `y_train.txt`: Training activity labels (1-6, corresponding to 6 activities).
  - `subject_train.txt`: Training participant IDs (1-30).
- **test/**: Test set data
  - `X_test.txt`: Test feature data (2,947 rows, 561 columns of features).
  - `y_test.txt`: Test activity labels.
  - `subject_test.txt`: Test participant IDs.
- **features.txt**: List of names for the 561 features.
- **activity_labels.txt**: Mapping of activity labels (1-6) to activity names.
- **README.txt**: Technical description of the dataset. [](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)

### Data Collection Method
- Participants wore a smartphone on their waist and performed 6 activities, with the process recorded on video for manual annotation.
- Sensor data includes x, y, z axis signals from the accelerometer and gyroscope, capturing body motion and gravity components.
- Data was filtered for noise and preprocessed to generate time-series features suitable for machine learning.

### Applications and Research
The UCI HAR dataset is widely used in the following research areas:
- **Time-Series Classification**: Using machine learning (e.g., SVM, random forests) or deep learning (e.g., LSTM, CNN) for activity classification. [](https://medium.com/data-science/a-guide-to-time-series-sensor-data-classification-using-uci-har-data-b7ac4f6ad251)[](https://arxiv.org/html/2505.06730v1)
- **Feature Engineering**: Studying how to extract effective features from raw time-series data (e.g., using TSFresh). [](https://medium.com/data-science/a-guide-to-time-series-sensor-data-classification-using-uci-har-data-b7ac4f6ad251)
- **Cross-Domain Generalization**: Testing model generalization across different datasets or devices. [](https://www.nature.com/articles/s41597-024-03951-4)
- **Missing Data Handling**: Exploring activity recognition methods with missing sensor data. [](https://arxiv.org/html/2505.06730v1)
- **Personalized Recognition**: Attempting to identify individuals performing activities (participant classification). [](https://arxiv.org/html/2505.06730v1)

### Related Research Achievements
- The dataset was first used in the 2012 paper *Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine* to test SVM models. [](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)
- Subsequent studies using deep learning methods (e.g., LSTM, CNN-LSTM) achieved 93%-99% accuracy in activity classification. [](https://arxiv.org/html/2505.06730v1)[](https://www.researchgate.net/figure/Description-of-UCI-HAR-dataset_tbl1_349651970)
- For example, the ConvResBiGRU-SE model achieved 99.18% accuracy on the UCI HAR dataset, demonstrating the superiority of deep residual networks and attention mechanisms. [](https://www.researchgate.net/figure/Description-of-UCI-HAR-dataset_tbl1_349651970)

### Obtaining the Dataset
- **Download Link**: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- **Kaggle**: Also provides the dataset, but the UCI original dataset is recommended for consistency with published results. [](https://www.researchgate.net/post/Can_Anyone_help_me_in_understandingc_features_in_UCI_HAR_Dataset)

### Notes
- **Data Consistency**: The Kaggle version may have different data splits compared to the UCI original dataset; use the UCI version for comparability. [](https://www.researchgate.net/post/Can_Anyone_help_me_in_understandingc_features_in_UCI_HAR_Dataset)
- **Preprocessing**: Raw time-series data requires additional processing (e.g., standardization, missing value imputation) to suit specific models. [](https://arxiv.org/html/2505.06730v1)[](https://www.nature.com/articles/s41597-024-03951-4)
- **Challenges**: Activity recognition involves large volumes of sensor data (tens of observations per second), requiring handling of time-series complexity and individual motion pattern differences. [](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)

---

### Python Code to Download the UCI HAR Dataset
The UCI HAR dataset can be downloaded from the UCI Machine Learning Repository (~58 MB, containing training and test data after decompression). The following code uses `requests` to download and extract the dataset:
```python
import requests
import zipfile
import os
import shutil
def download_uci_har_dataset(save_dir='./UCI_HAR_Dataset'):
    """
    Download and extract the UCI HAR dataset to the specified directory
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    zip_path = os.path.join(save_dir, 'UCI_HAR_Dataset.zip')
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Download dataset
    print("Downloading UCI HAR dataset...")
    response = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete!")
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    print("Extraction complete!")
    # Delete zip file (optional)
    os.remove(zip_path)
    print(f"Dataset saved to {save_dir}")
# Execute download
download_uci_har_dataset()
```

### Description
- **Dependencies**: Requires `requests` (`pip install requests`).
- **Save Path**: The dataset is downloaded to the `./UCI_HAR_Dataset` directory, extracting to a `UCI HAR Dataset` folder.
- **File Structure**:
  - `train/X_train.txt`: Training features (7,352 samples, 561 features).
  - `train/y_train.txt`: Training labels (1-6, for 6 activities).
  - `train/subject_train.txt`: Training participant IDs.
  - `test/X_test.txt`, `test/y_test.txt`, `test/subject_test.txt`: Corresponding test set files.
  - `activity_labels.txt`: Activity label mapping (e.g., 1=Walking).
  - `features.txt`: Names of the 561 features.

---

### Simple Usage Example Code
The following is an example code for loading the UCI HAR dataset, performing basic preprocessing, and visualizing the data distribution using Pandas and Matplotlib:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
def load_uci_har_dataset(data_dir='./UCI_HAR_Dataset/UCI HAR Dataset'):
    """
    Load the UCI HAR dataset
    Return features, labels, and participant IDs for training and test sets
    """
    # Read feature names
    features = pd.read_csv(os.path.join(data_dir, 'features.txt'), sep='\s+', header=None, names=['index', 'feature_name'])['feature_name'].values
   
    # Read training data
    X_train = pd.read_csv(os.path.join(data_dir, 'train/X_train.txt'), sep='\s+', header=None, names=features)
    y_train = pd.read_csv(os.path.join(data_dir, 'train/y_train.txt'), sep='\s+', header=None, names=['activity'])
    subject_train = pd.read_csv(os.path.join(data_dir, 'train/subject_train.txt'), sep='\s+', header=None, names=['subject'])
   
    # Read test data
    X_test = pd.read_csv(os.path.join(data_dir, 'test/X_test.txt'), sep='\s+', header=None, names=features)
    y_test = pd.read_csv(os.path.join(data_dir, 'test/y_test.txt'), sep='\s+', header=None, names=['activity'])
    subject_test = pd.read_csv(os.path.join(data_dir, 'test/subject_test.txt'), sep='\s+', header=None, names=['subject'])
   
    # Read activity label mapping
    activity_labels = pd.read_csv(os.path.join(data_dir, 'activity_labels.txt'), sep='\s+', header=None, names=['id', 'activity_name'])
    activity_map = dict(zip(activity_labels['id'], activity_labels['activity_name']))
   
    # Map labels to activity names
    y_train['activity_name'] = y_train['activity'].map(activity_map)
    y_test['activity_name'] = y_test['activity'].map(activity_map)
   
    return X_train, y_train, subject_train, X_test, y_test, subject_test, activity_map
def visualize_activity_distribution(y_train, y_test, activity_map):
    """
    Visualize the distribution of activities in the training and test sets
    """
    plt.figure(figsize=(12, 5))
   
    # Training set activity distribution
    plt.subplot(1, 2, 1)
    y_train['activity_name'].value_counts().plot(kind='bar', title='Training Set Activity Distribution')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
   
    # Test set activity distribution
    plt.subplot(1, 2, 2)
    y_test['activity_name'].value_counts().plot(kind='bar', title='Test Set Activity Distribution')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
   
    plt.tight_layout()
    plt.show()
# Load dataset
X_train, y_train, subject_train, X_test, y_test, subject_test, activity_map = load_uci_har_dataset()
# Print dataset information
print(f"Training set feature shape: {X_train.shape}") # (7352, 561)
print(f"Training set label shape: {y_train.shape}") # (7352, 2)
print(f"Test set feature shape: {X_test.shape}") # (2947, 561)
print(f"Test set label shape: {y_test.shape}") # (2947, 2)
print("Activity label mapping:", activity_map)
# Visualize activity distribution
visualize_activity_distribution(y_train, y_test, activity_map)
# Example: Simple preprocessing (feature standardization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Standardized training set features (first 5 rows):")
print(X_train_scaled[:5, :5]) # Show first 5 rows and 5 columns
```

### Code Description
1. **Dependencies**:
   - Install required libraries: `pip install pandas numpy matplotlib requests scikit-learn`
2. **Download Code**:
   - Uses `requests` to download the UCI HAR dataset ZIP file and extract it.
   - Saves to the specified directory (default `./UCI_HAR_Dataset`).
3. **Load Code**:
   - Uses Pandas to read `X_train.txt` (features), `y_train.txt` (labels), `subject_train.txt` (participant IDs), etc.
   - Loads activity label mapping (`activity_labels.txt`) and converts numeric labels to activity names.
4. **Visualization**:
   - Plots bar charts of activity distributions for training and test sets to assess data balance.
5. **Preprocessing**:
   - Standardizes features using `StandardScaler` (mean=0, variance=1) for machine learning model input.
6. **Output Example**:
   - Displays dataset shapes, activity label mapping, and a snippet of standardized features.
   - Generates bar charts showing the distribution of the 6 activities (Walking, Sitting, etc.).

### Expected Results
- **Dataset Information**:
  - Training set: 7,352 samples, 561 features.
  - Test set: 2,947 samples, 561 features.
  - Activity labels: {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}.
- **Visualization**: Bar charts showing the sample counts for each activity in the training and test sets, confirming a relatively balanced distribution.
- **Standardized Features**: Outputs a portion of the standardized feature values for subsequent model training.

### Usage Recommendations
- **Model Training**: Use `X_train_scaled` and `y_train['activity']` to train classification models (e.g., SVM, random forests, or deep learning models like LSTM, CNN).
- **Deep Learning Example**: For PyTorch or TensorFlow, convert the data to tensor format for training.
- **Extensions**:
   - Load raw sensor data (`train/Inertial Signals/`) for time-series analysis.
   - Apply feature selection or dimensionality reduction (e.g., PCA) to reduce the complexity of the 561 features.

### Notes
- **Data Path**: Ensure `data_dir` points to the extracted dataset directory.
- **Memory Requirements**: The dataset is small (~58 MB), manageable on standard PCs.
- **Kaggle Alternative**: Kaggle provides the UCI HAR dataset, but its splits may differ; use the UCI official version for consistency.
- **Error Handling**: If downloading or extraction fails, check network connectivity or disk space.
