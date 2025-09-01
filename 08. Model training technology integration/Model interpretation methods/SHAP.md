
# SHAP (SHapley Additive exPlanations) Model Interpretation Method

## 1. What is SHAP?

SHAP (SHapley Additive exPlanations) is a commonly used **model interpretation method**, based on the concept of **Shapley values** in game theory.
It quantifies the contribution of each input feature to the prediction result, making “black-box models” (such as deep learning, tree models, and ensemble methods) more transparent.


## 2. Why do we need SHAP?

* **Developers**: Help discover feature importance and debug the model.
* **Users/Business stakeholders**: Improve trust and understand prediction results.
* **Compliance/Safety**: In domains like finance and healthcare, models must be interpretable.

Example: In a medical prediction model, SHAP can explain why the prediction for “a certain patient is high risk,” possibly because of high blood pressure or older age.


## 3. Core Principles of SHAP

### 3.1 Based on Shapley Values

In game theory, **Shapley values** are used to measure each participant’s contribution to the outcome of cooperation.
In analogy to machine learning:

* **Players** → Features
* **Cooperation payoff** → Model prediction value
* **Allocation method** → Contribution value of each feature

### 3.2 Formula

The model output can be decomposed as:

$$
f(x) = \phi_0 + \sum_{i=1}^n \phi_i
$$

* \$\phi\_0\$: Baseline value (usually the average prediction of the training set).
* \$\phi\_i\$: Contribution of feature \$i\$ to the prediction.

### 3.3 Properties

SHAP values satisfy the following properties:

1. **Fairness**: The sum of contributions of all features equals the difference between the prediction and the baseline.
2. **Consistency**: If a feature’s contribution increases, its SHAP value will not decrease.
3. **Additivity**: Contribution values are linearly additive.


## 4. Applications of SHAP in Deep Learning

* **Local explanation**: Explaining the prediction of a single sample.
* **Global explanation**: Summarizing feature importance across all samples.
* **Model debugging**: Detecting whether the model relies on incorrect or biased features.
* **Business applications**: Interpretable AI in high-risk domains such as healthcare and finance.


## 5. Visualization Methods

SHAP provides multiple intuitive visualizations:

* **Summary Plot**: Global feature importance and the impact of feature values.
* **Force Plot**: Contributions of features for single or multiple samples.
* **Dependence Plot**: Relationship between feature values and their contributions.


## 6. Practical Code Example (PyTorch + SHAP)

Below is a **complete Notebook Demo** showing how to train a model and interpret it with SHAP.

```python
# =========================
# 1. Import dependencies
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 2. Define the model
# =========================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# 3. Generate data
# =========================
np.random.seed(42)
X_train = np.random.randn(200, 4).astype(np.float32)
y_train = (X_train[:, 0] + 0.5*X_train[:, 1] - X_train[:, 2]) > 0
y_train = y_train.astype(np.float32).reshape(-1, 1)

X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train)

# =========================
# 4. Train the model
# =========================
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

print("Training finished! Final loss:", loss.item())

# =========================
# 5. Explain the model with SHAP
# =========================
explainer = shap.DeepExplainer(model, X_train_torch[:50])   # Background data
shap_values = explainer.shap_values(X_train_torch[:50])     # Explain the first 50 samples

# =========================
# 6. Summary Plot (global feature importance)
# =========================
shap.summary_plot(
    shap_values[0], 
    X_train[:50], 
    feature_names=["f1", "f2", "f3", "f4"]
)

# =========================
# 7. Force Plot (single sample)
# =========================
sample_index = 0
shap.force_plot(
    base_value=explainer.expected_value[0].detach().numpy(),
    shap_values=shap_values[0][sample_index].detach().numpy(),
    features=X_train[sample_index],
    feature_names=["f1","f2","f3","f4"],
    matplotlib=True
)

# =========================
# 8. Force Plot (multiple samples)
# =========================
shap.force_plot(
    base_value=explainer.expected_value[0].detach().numpy(),
    shap_values=shap_values[0][:5].detach().numpy(),
    features=X_train[:5],
    feature_names=["f1","f2","f3","f4"],
    matplotlib=True
)

# =========================
# 9. Dependence Plot (relationship between feature values and contributions)
# =========================
shap.dependence_plot(
    "f1", 
    shap_values[0], 
    X_train[:50], 
    feature_names=["f1","f2","f3","f4"]
)
```



## 7. Summary

* **SHAP = Shapley values + model interpretation**
* It can explain single samples (local) and the entire model (global)
* Provides multiple visualization methods to intuitively show feature contributions
* In deep learning, `DeepExplainer` is commonly used, with optimized versions available for tree models and linear models as well


