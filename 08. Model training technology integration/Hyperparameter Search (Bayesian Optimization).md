## Hyperparameter Search (Bayesian Optimization)
## Hyperparameter Search (Bayesian Optimization)
### What is Bayesian Optimization?
Bayesian Optimization is a probabilistic model-based global optimization algorithm commonly used for hyperparameter tuning in machine learning. It is particularly suited for scenarios where evaluating the target function is computationally expensive, such as training a deep learning model. Unlike grid search or random search, Bayesian optimization builds a surrogate model (typically a Gaussian Process) to model the distribution of the target function and uses an acquisition function (e.g., Expected Improvement) to intelligently select the next hyperparameter point to evaluate, efficiently exploring the hyperparameter space.
#### Core Principle
- **Surrogate Model**: Uses a Gaussian Process or other probabilistic model to fit the performance of evaluated points, predicting the mean and uncertainty of unevaluated points.
- **Acquisition Function**: Calculates the "value" of each potential point based on the surrogate model, balancing exploration (high uncertainty points) and exploitation (points predicted to perform well).
- **Iterative Process**:
  1. Initialize: Randomly sample a few hyperparameter points and evaluate the target function (e.g., model accuracy).
  2. Update Model: Update the surrogate model with new data.
  3. Select Next Point: Use the acquisition function to choose the next hyperparameter point.
  4. Repeat until convergence or reaching an iteration limit.
- **Advantages**: More efficient in high-dimensional spaces, often finding near-optimal solutions with fewer evaluations; suitable for both continuous and discrete hyperparameters.
- **Disadvantages**: The computational overhead of the surrogate model may increase in very high dimensions; requires defining reasonable hyperparameter boundaries.
Bayesian optimization is widely used for tuning models like XGBoost and neural networks, significantly reducing search time and improving performance.
---
### Python Code Example
Below is an example of Bayesian optimization for hyperparameter search using the `skopt` library’s `BayesSearchCV` in Python. The example tunes hyperparameters (e.g., learning rate, max depth) for an XGBoost classifier on the digits dataset. Note: Running this code requires installing `skopt` and `xgboost` (`pip install scikit-optimize xgboost`) and ensuring base libraries like `numpy` and `sklearn` are available.
```python
import numpy as np
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
# Step 1: Load and split the dataset
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 2: Define the hyperparameter search space
# Use tuples to define ranges: (min, max, 'distribution type'), e.g., uniform or log-uniform
param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),  # Learning rate, log-uniform distribution
    'max_depth': (1, 50),  # Max depth, integer uniform distribution
    'gamma': (1e-9, 0.6, 'log-uniform'),  # Regularization parameter
    'n_estimators': (50, 100),  # Number of trees
    'degree': (1, 6),  # Kernel degree (if applicable)
    'kernel': ['linear', 'rbf', 'poly']  # Discrete categorical parameter
}
# Step 3: Initialize the Bayesian optimizer
# Use BayesSearchCV, specifying model, search space, scoring metric, cross-validation, and iterations
optimizer = BayesSearchCV(
    estimator=XGBClassifier(n_jobs=1),  # XGBoost classifier
    search_spaces=param_space,  # Search space
    scoring='accuracy',  # Evaluation metric: accuracy
    cv=3,  # 3-fold cross-validation
    n_iter=50,  # Total iterations (evaluate 50 points)
    random_state=42  # Random seed for reproducibility
)
# Step 4: Fit the model and search for optimal hyperparameters
optimizer.fit(X_train, y_train)
# Step 5: Output results
best_params = optimizer.best_params_
best_score = optimizer.best_score_
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_score:.4f}")
# Optional: Evaluate on test set
test_score = optimizer.score(X_test, y_test)
print(f"Test Set Accuracy: {test_score:.4f}")
```
#### Code Explanation
1. **Data Preparation**: Loads the sklearn digits dataset and splits it into training/test sets.
2. **Search Space**: Defines the range and distribution type (continuous, discrete) for hyperparameters.
3. **Optimizer Initialization**: `BayesSearchCV` encapsulates the Bayesian optimization process, using a Gaussian Process as the surrogate model.
4. **Fit and Results**: Runs optimization, automatically evaluating model performance across iterations. `n_iter=50` means evaluating 50 hyperparameter combinations.
5. **Output**: Prints the best hyperparameters and score. In practice, the best accuracy may exceed 0.98.
This example demonstrates the application of Bayesian optimization in practical hyperparameter tuning.
