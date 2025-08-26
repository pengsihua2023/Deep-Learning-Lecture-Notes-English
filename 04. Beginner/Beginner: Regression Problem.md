# Beginner: Regression Problem
## Mathematical Description of Regression

### 1. General Regression Problem

Given the training dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n
$$

where:

* \$x\_i \in \mathbb{R}^d\$ is the input feature vector;
* \$y\_i \in \mathbb{R}\$ is the corresponding target output.

Regression model assumption:

$$
y_i = f(x_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

where \$f(\cdot)\$ is the underlying true function and \$\varepsilon\_i\$ is noise.

The learning objective is to find an estimator function \$\hat{f}(\cdot)\$ such that the predicted value:

$$
\hat{y}_i = \hat{f}(x_i)
$$

can approximate the true value \$y\_i\$ as closely as possible.

Typically, the solution is obtained by minimizing the Mean Squared Error (MSE):

$$
\hat{f} = \arg \min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \left( y_i - f(x_i) \right)^2
$$

---

### 2. Linear Regression
<div align="center">
<img width="400" height="264" alt="image" src="https://github.com/user-attachments/assets/3a54ab56-20cb-4780-9d89-7af29f76956f" />    


Figure 1 Linear Regression
</div>

#### Model Form

Assume the output is a linear combination of input features:

$$
f(x) = w^\top x + b
$$

where:

* \$w \in \mathbb{R}^d\$ is the weight vector;
* \$b \in \mathbb{R}\$ is the bias term.

#### Loss Function

The objective function of linear regression is:

$$
L(w, b) = \frac{1}{n} \sum_{i=1}^n \left( y_i - (w^\top x_i + b) \right)^2
$$

#### Optimal Solution (Closed Form)

Let \$X = \[x\_1^\top; x\_2^\top; \dots; x\_n^\top] \in \mathbb{R}^{n \times d}\$,
\$y = \[y\_1, y\_2, \dots, y\_n]^\top \in \mathbb{R}^n\$,
then the closed-form solution for the optimal parameter is:

$$
\hat{w} = (X^\top X)^{-1} X^\top y
$$

---

### 3. Nonlinear Regression
<div align="center">
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/27696c25-d662-48ce-8bf2-0d9538119e8d" />     
</div>

Figure 2 Nonlinear Regression

When the relationship between features and target is not linear, nonlinear functions \$f(\cdot)\$ can be used.

#### (1). Polynomial Regression

Map input to higher-order features:

$$
f(x) = \sum_{j=0}^p w_j x^j
$$

The corresponding optimization objective is:

$$
L(w) = \frac{1}{n} \sum_{i=1}^n \left( y_i - \sum_{j=0}^p w_j (x_i)^j \right)^2
$$

---

#### (2). Kernel Regression

Using nonlinear feature mapping \$\phi(x)\$, the model is formulated as:

$$
f(x) = w^\top \phi(x)
$$

The optimization problem is:

$$
\hat{w} = \arg \min_w \frac{1}{n} \sum_{i=1}^n \left( y_i - w^\top \phi(x_i) \right)^2
$$

By introducing the kernel function \$k(x, x') = \phi(x)^\top \phi(x')\$, we can avoid explicitly constructing \$\phi(x)\$.

---

#### (3). Neural Network Regression

Assume the model is a multi-layer nonlinear function:

$$
f(x) = W^{(L)} \sigma \big( W^{(L-1)} \sigma ( \cdots \sigma(W^{(1)} x + b^{(1)}) \cdots ) + b^{(L-1)} \big) + b^{(L)}
$$

where:

* \$\sigma(\cdot)\$ is a nonlinear activation function;
* \${W^{(\ell)}, b^{(\ell)}}\$ are the parameters of the \$\ell\$-th layer.

The loss function is still Mean Squared Error:

$$
L(\theta) = \frac{1}{n} \sum_{i=1}^n \left( y_i - f(x_i; \theta) \right)^2
$$

where \$\theta\$ denotes the set of all parameters.

--- 

## First version of code 
- First version of code
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an improved neural network model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer
    layers.Dense(32, activation='relu'),  # Second hidden layer
    layers.Dense(1)  # Output layer
])

# Compile the model with a smaller learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model, increase epoch count, and add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X

```
### Training results
11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 5.4454 - mae: 1.7790 - val_loss: 7.5367 - val_mae: 1.9705  
Test Loss: 28.6256, Test MAE: 2.9774  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step  
Predictions for first 5 test samples: [21.798922 21.89294  20.243843 33.85427  21.61176 ]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  
- Second version of code
```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an optimized model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
    layers.Dropout(0.2),  # Dropout layer
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.
```
### training results
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 15.5459 - mae: 2.9512 - val_loss: 7.5879 - val_mae: 1.8665  
Test Loss: 28.6987, Test MAE: 3.1621  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step  
Predictions for first 5 test samples: [19.722134 21.551369 19.76431  34.056454 22.540033]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  

- Third version of code
```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an optimized model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.001)),  # Reduce L2 regularization
    layers.Dropout(0.1),  # Reduce Dropout rate
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2, 
          callbacks=[early_stopping], verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])
```
### training results
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.3365 - mae: 1.8211 - val_loss: 6.4201 - val_mae: 1.7893  
Test Loss: 25.8082, Test MAE: 2.9562  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step  
Predictions for first 5 test samples: [19.786161 21.211859 19.15384  34.56086  20.987986]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  

- Fourth version of code
```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an optimized model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.0005)),  # Lower L2 regularization
    layers.Dropout(0.05),  # Lower Dropout rate
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.05),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dense(1)
])

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    alpha=0.0001
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, 
          callbacks=[early_stopping], verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])
```
### training results
Epoch 87/1000
41/41 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 5.1438 - mae: 1.6906 - val_loss: 6.9840 - val_mae: 1.8983  
Test Loss: 26.1492, Test MAE: 3.0013  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step  
Predictions for first 5 test samples: [20.502605 21.119947 19.357008 34.657536 21.160563]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  
- Fifth version of code
```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an optimized model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.05),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.05),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dense(1)
])

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    alpha=0.0001
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, 
                   callbacks=[early_stopping], verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])

# Plot MAE curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Train and Validation MAE vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
```
### training results
Epoch 88/1000  
41/41 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0396 - mae: 1.7683 - val_loss: 6.8140 - val_mae: 1.9475  
Test Loss: 27.4981, Test MAE: 3.0507  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step  
Predictions for first 5 test samples: [20.937302 21.09048  19.416035 35.247955 21.223663]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  
<img width="911" height="480" alt="image" src="https://github.com/user-attachments/assets/c4351ded-28e5-4790-9771-85dd53fd5df0" />  
<img width="924" height="470" alt="image" src="https://github.com/user-attachments/assets/769de06c-a3bb-4333-a77b-ef0de3febdd2" />
