## Intermediate: Multi-Task Learning （MTL）

- **Importance**:  
  Multi-task learning trains ESM-2 simultaneously on multiple protein tasks (e.g., function classification, structure prediction), improving model generalization and efficiency.  
  It is a popular technique in bioinformatics, as protein tasks are often related (e.g., function and structure).  
- **Core Concept**:  
  The model shares most parameters, with task-specific output heads, jointly optimizing multiple objectives.
     
  <img width="685" height="494" alt="image" src="https://github.com/user-attachments/assets/4dd18183-6e9e-4418-ab2b-b0f9e8edb4bb" />

A minimal PyTorch-based Multi-Task Learning (MTL) example using a real dataset (UCI Wine Quality dataset) to implement two tasks: predicting wine quality (regression task) and predicting whether the wine is high-quality (classification task, quality ≥ 6 is high-quality). Results will be demonstrated through visualization (scatter plot of predicted quality) and evaluation metrics (MSE for regression, accuracy for classification).  

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd

# Define the multi-task learning model
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiTaskModel, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Regression head: Predict quality score
        self.regression_head = nn.Linear(hidden_dim, 1)
        # Classification head: Predict if high-quality
        self.classification_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        quality_pred = self.regression_head(shared_features)
        is_good_pred = self.classification_head(shared_features)
        return quality_pred, is_good_pred

# Data preparation
def prepare_data():
    # Load the Wine Quality dataset
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    X = data.drop('quality', axis=1).values
    y_quality = data['quality'].values
    y_class = (y_quality >= 6).astype(int)  # Quality ≥ 6 is high-quality
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test = train_test_split(
        X, y_quality, y_class, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_quality_train = torch.FloatTensor(y_quality_train).reshape(-1, 1)
    y_quality_test = torch.FloatTensor(y_quality_test).reshape(-1, 1)
    y_class_train = torch.FloatTensor(y_class_train).reshape(-1, 1)
    y_class_test = torch.FloatTensor(y_class_test).reshape(-1, 1)
    
    return X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test

# Train the model
def train_model(model, X_train, y_quality_train, y_class_train, epochs=100, lr=0.01):
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        quality_pred, is_good_pred = model(X_train)
        loss_reg = criterion_reg(quality_pred, y_quality_train)
        loss_cls = criterion_cls(is_good_pred, y_class_train)
        loss = loss_reg + loss_cls  # Simple sum of losses
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Regression Loss: {loss_reg.item():.4f}, Classification Loss: {loss_cls.item():.4f}')

# Evaluate and visualize
def evaluate_and_visualize(model, X_test, y_quality_test, y_class_test):
    model.eval()
    with torch.no_grad():
        quality_pred, is_good_pred = model(X_test)
        quality_pred = quality_pred.numpy()
        is_good_pred = (torch.sigmoid(is_good_pred) > 0.5).float().numpy()
        y_quality_test = y_quality_test.numpy()
        y_class_test = y_class_test.numpy()
    
    # Compute evaluation metrics
    mse = mean_squared_error(y_quality_test, quality_pred)
    accuracy = accuracy_score(y_class_test, is_good_pred)
    print(f'\nTest Set Evaluation:')
    print(f'Regression MSE: {mse:.4f}')
    print(f'Classification Accuracy: {accuracy:.4f}')
    
    # Visualize regression task predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_quality_test, quality_pred, alpha=0.5)
    plt.plot([y_quality_test.min(), y_quality_test.max()], [y_quality_test.min(), y_quality_test.max()], 'r--')
    plt.xlabel('True Quality')
    plt.ylabel('Predicted Quality')
    plt.title('Wine Quality Prediction (Regression Task)')
    plt.tight_layout()
    plt.savefig('wine_quality_prediction.png')
    plt.close()
    print("Prediction scatter plot saved as 'wine_quality_prediction.png'")
    
    # Print predictions for the first few samples
    print("\nSample Predictions (First 5):")
    for i in range(5):
        print(f"Sample {i+1}: True Quality={y_quality_test[i][0]:.2f}, Predicted Quality={quality_pred[i][0]:.2f}, "
              f"True Class={y_class_test[i][0]:.0f}, Predicted Class={is_good_pred[i][0]:.0f}")

def main():
    # Prepare data
    X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test = prepare_data()
    
    # Initialize model
    model = MultiTaskModel(input_dim=11, hidden_dim=64)
    
    # Train
    train_model(model, X_train, y_quality_train, y_class_train, epochs=100)
    
    # Evaluate and visualize
    evaluate_and_visualize(model, X_test, y_quality_test, y_class_test)

if __name__ == "__main__":
    main()
```


### Code Description:
1. **Dataset**:
   - Uses the UCI Wine Quality dataset (red wine, 1,599 samples), containing 11 chemical features and quality scores (3-8).
   - **Task 1 (Regression)**: Predict the quality score.
   - **Task 2 (Classification)**: Predict whether the wine is high-quality (quality ≥ 6).
   - Data is loaded via `pandas` from the UCI website, standardized, and split into training (80%) and test (20%) sets.

2. **Model Architecture**:
   - Shared layers: Two fully connected layers (ReLU activation), with 11-dimensional input features and a 64-dimensional hidden layer.
   - Regression head: Outputs a 1-dimensional quality score.
   - Classification head: Outputs a 1-dimensional binary classification probability (high-quality/non-high-quality).
   - Loss functions: MSELoss for regression, BCEWithLogitsLoss for classification, combined loss is the sum of both.

3. **Training**:
   - Uses Adam optimizer with a learning rate of 0.01, trained for 100 epochs.
   - Prints total loss, regression loss, and classification loss every 20 epochs.

4. **Evaluation and Visualization**:
   - Evaluates regression task with Mean Squared Error (MSE) and classification task with accuracy.
   - Plots a scatter plot showing the relationship between true and predicted quality scores, saved as `wine_quality_prediction.png`.
   - Prints true and predicted values (quality scores and classification results) for the first 5 test samples.

5. **Dependencies**:
   - Requires `torch`, `sklearn`, `pandas`, `matplotlib`, and `seaborn` (`pip install torch scikit-learn pandas matplotlib seaborn datasets`).
   - Dataset is loaded online, no manual download required.

### Results:
- Outputs loss values during training.
- Test set evaluation:
  - MSE for the regression task (reflecting prediction error for quality scores).
  - Accuracy for the classification task (reflecting correctness of high-quality/non-high-quality classification).
- Generates `wine_quality_prediction.png`, showing a scatter plot of predicted vs. true quality scores (red line represents the ideal prediction line).
- Prints prediction results for the first 5 samples, showing true and predicted quality scores and classification results.

### Notes:
- The heatmap is saved in the working directory and can be viewed with an image viewer.
- The model is simple (two shared layers), suitable for demonstrating Multi-Task Learning (MTL) concepts; for practical applications, consider adding more layers or using more complex architectures.
