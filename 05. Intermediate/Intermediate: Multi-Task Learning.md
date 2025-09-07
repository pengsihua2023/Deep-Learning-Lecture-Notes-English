# Multi-Task Learning (MTL)
## 📖 Definition  
Multi-Task Learning (MTL) is a machine learning training paradigm. The core idea is: a single model learns multiple related tasks simultaneously, instead of training separate models for each task like traditional methods. The model shares most parameters, each task has a specific output head, and multiple objectives are jointly optimized.  

<div align="center">
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/4dd18183-6e9e-4418-ab2b-b0f9e8edb4bb" />
</div>
<div align="center">
(This image is cited from the Internet)
</div>

## 📖 Mathematical Formulation of Multi-Task Learning

### 1. Basic Form of Single-Task Learning

Given dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N
$$

* $x_i \in \mathcal{X}$: input features of the $i$-th sample.  
* $y_i \in \mathcal{Y}$: supervision signal (label) corresponding to the $i$-th sample.  
* $N$: number of training samples.  

We train a model parameterized by $\theta$:

$$
f_\theta : \mathcal{X} \to \mathcal{Y},
$$

The objective is to minimize the expected loss:

$$
\min_\theta \ \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), y) \right].
$$



### 2. Extended Form of Multi-Task Learning

Suppose there are $T$ tasks, and for each task $t$, the dataset is:

$$
\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t},
$$

* $x_i^t$: input of task $t$.  
* $y_i^t$: label of task $t$.  
* $N_t$: number of samples in task $t$.  

Each task corresponds to a loss function $\mathcal{L}_t$. The optimization objective of multi-task learning is:

$$
\min_\theta \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \Big[ \mathcal{L}_t(f_\theta(x), y) \Big].
$$

* $\lambda_t$: task weight, controlling the importance of each task in the overall objective.  



### 3. Structured Representation of Parameter Sharing

In practice, a common approach is **shared representation layers + task-specific output layers**:

1. **Shared Representation Layer**:

$$
h = \phi_{\theta_s}(x),
$$

* $\phi_{\theta_s}$: feature extractor (e.g., the first few layers of a neural network), with parameters $\theta_s$ shared across all tasks.  
* $h$: shared latent representation.  

2. **Task-Specific Output Layer**:

$$
\hat{y}^t = f^t_{\theta_t}(h),
$$

* $f^t_{\theta_t}$: predictor for task $t$, with parameters $\theta_t$ specific to task $t$.  
* $\hat{y}^t$: model prediction for task $t$.  

Overall optimization objective:

$$
\min_{\theta_s, \{\theta_t\}_{t=1}^T} \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \left[ \mathcal{L}_t(f^t_{\theta_t}(\phi_{\theta_s}(x)), y) \right].
$$



### 4. Matrix / Regularization Perspective

If the task parameter matrix is assumed to be:

$$
W = [\theta_1, \dots, \theta_T] \in \mathbb{R}^{d \times T},
$$

Regularization constraints can be added outside the loss function:

### (a) Low-Rank Constraint

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \lambda \|W\|_*
$$

* $\|W\|_*$: nuclear norm, encouraging $W$ to be low-rank, meaning tasks share a low-dimensional subspace.  

### (b) Graph Regularization

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \gamma \sum_{(i,j)\in E} \|W_i - W_j\|^2
$$

* $E$: edge set of the task relationship graph.  
* $\|W_i - W_j\|^2$: encourages parameters of similar tasks to be close.  



### 5. Bayesian Perspective

Introduce a prior distribution over task parameters:

$$
p(\theta_1, \dots, \theta_T | \alpha) = \prod_{t=1}^T p(\theta_t | \alpha)
$$

* $\alpha$: shared hyperparameters, controlling the prior distribution for all tasks.  



### Summary

There are three main approaches to mathematical modeling of multi-task learning:

1. **Weighted loss function** (simple summation of tasks, with weights $\lambda_t$).  
2. **Parameter sharing** (shared layers $\theta_s$ + task-specific heads $\theta_t$).  
3. **Regularization / Probabilistic modeling** (modeling task relationships via nuclear norm, graph regularization, or shared priors).  

---
## 📖 Code
A simplest PyTorch-based Multi-Task Learning (MTL) example, using a real dataset (UCI Wine Quality dataset), implementing two tasks: predicting wine quality (regression task) and predicting whether the wine is good quality (classification task, quality ≥ 6 as good). Results are demonstrated through visualization (scatter plot of predicted quality) and evaluation metrics (MSE for regression, accuracy for classification).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Define Multi-Task Learning model
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiTaskModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.classification_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        quality_pred = self.regression_head(shared_features)
        is_good_pred = self.classification_head(shared_features)
        return quality_pred, is_good_pred

# Data preparation
def prepare_data():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    X = data.drop('quality', axis=1).values
    y_quality = data['quality'].values
    y_class = (y_quality >= 6).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test = train_test_split(
        X, y_quality, y_class, test_size=0.2, random_state=42
    )
    
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
        loss = loss_reg + loss_cls
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Regression Loss: {loss_reg.item():.4f}, Classification Loss: {loss_cls.item():.4f}')

# Evaluation and visualization
def evaluate_and_visualize(model, X_test, y_quality_test, y_class_test):
    model.eval()
    with torch.no_grad():
        quality_pred, is_good_pred = model(X_test)
        quality_pred = quality_pred.numpy()
        is_good_pred = (torch.sigmoid(is_good_pred) > 0.5).float().numpy()
        y_quality_test = y_quality_test.numpy()
        y_class_test = y_class_test.numpy()
    
    mse = mean_squared_error(y_quality_test, quality_pred)
    accuracy = accuracy_score(y_class_test, is_good_pred)
    print(f'\nTest Set Evaluation:')
    print(f'Regression MSE: {mse:.4f}')
    print(f'Classification Accuracy: {accuracy:.4f}')
    
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

    print("\nSample Predictions (First 5):")
    for i in range(5):
        print(f"Sample {i+1}: True Quality={y_quality_test[i][0]:.2f}, Predicted Quality={quality_pred[i][0]:.2f}, "
              f"True Class={y_class_test[i][0]:.0f}, Predicted Class={is_good_pred[i][0]:.0f}")

def main():
    X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test = prepare_data()
    model = MultiTaskModel(input_dim=11, hidden_dim=64)
    train_model(model, X_train, y_quality_train, y_class_train, epochs=100)
    evaluate_and_visualize(model, X_test, y_quality_test, y_class_test)

if __name__ == "__main__":
    main()
````

## 📖 Code Explanation:

1. **Dataset**:

   * Uses the UCI Wine Quality dataset (red wine, 1599 samples), with 11 chemical features and quality scores (3–8).
   * Task 1 (regression): predict the quality score.
   * Task 2 (classification): predict whether it is good quality (quality ≥ 6).
   * Data is loaded via `pandas` from the UCI website, standardized, and split into training (80%) and testing (20%).

2. **Model Structure**:

   * Shared layers: two fully connected layers (ReLU activation), input 11-dimensional features, hidden dimension 64.
   * Regression head: outputs 1-dimensional quality score.
   * Classification head: outputs 1-dimensional binary classification probability (good/bad).
   * Loss function: MSELoss for regression, BCEWithLogitsLoss for classification, combined as their sum.

3. **Training**:

   * Uses Adam optimizer, learning rate 0.01, trained for 100 epochs.
   * Every 20 epochs, prints total loss, regression loss, and classification loss.

4. **Evaluation & Visualization**:

   * Evaluates MSE for regression and accuracy for classification.
   * Generates a scatter plot of predicted vs. true quality, saved as `wine_quality_prediction.png`.
   * Prints predictions for the first 5 test samples (both quality scores and classification results).

5. **Dependencies**:

   * Requires installation of `torch`, `sklearn`, `pandas`, `matplotlib`, `seaborn`

     ```bash
     pip install torch scikit-learn pandas matplotlib seaborn datasets
     ```
   * Dataset is loaded online, no manual download needed.

## 📖 Results:

* Outputs training losses during training.
* Test set evaluation:

  * Regression task: MSE (reflecting error in predicted quality score).
  * Classification task: accuracy (reflecting correctness of good/bad classification).
* Generates `wine_quality_prediction.png`, showing scatter plot of predicted vs. true quality (red line = ideal prediction line).
* Prints predictions for the first 5 samples, showing true and predicted quality scores and classification results.


```
