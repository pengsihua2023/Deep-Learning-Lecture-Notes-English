## Beginner: Regression Problem
<img width="400" height="264" alt="image" src="https://github.com/user-attachments/assets/3a54ab56-20cb-4780-9d89-7af29f76956f" />    

Figure 1 Linear regression      
<img width="583" height="432" alt="image" src="https://github.com/user-attachments/assets/27696c25-d662-48ce-8bf2-0d9538119e8d" />     

Figure 2 Nonlinear regression  

## First version of code 
- 第一版代码 
```
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载波士顿房价数据集
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建改进的神经网络模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # 隐藏层
    layers.Dense(32, activation='relu'),  # 第二个隐藏层
    layers.Dense(1)  # 输出层
])

# 编译模型，使用更小的学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 训练模型，增加 epoch 数并添加早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# 评估模型
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 预测
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])

```
### 训练结果
11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 5.4454 - mae: 1.7790 - val_loss: 7.5367 - val_mae: 1.9705  
Test Loss: 28.6256, Test MAE: 2.9774  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step  
Predictions for first 5 test samples: [21.798922 21.89294  20.243843 33.85427  21.61176 ]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  
- 第二版代码
```
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载波士顿房价数据集
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建优化模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.01)),  # L2正则化
    layers.Dropout(0.2),  # Dropout层
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.2, 
          callbacks=[early_stopping], verbose=1)

# 评估模型
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 预测
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])
```
### 训练结果
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 15.5459 - mae: 2.9512 - val_loss: 7.5879 - val_mae: 1.8665  
Test Loss: 28.6987, Test MAE: 3.1621  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step  
Predictions for first 5 test samples: [19.722134 21.551369 19.76431  34.056454 22.540033]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  

- 第三版代码
```
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载波士顿房价数据集
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建优化模型
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.001)),  # 降低L2正则化
    layers.Dropout(0.1),  # 降低Dropout率
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

# 学习率调度
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 编译模型
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2, 
          callbacks=[early_stopping], verbose=1)

# 评估模型
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 预测
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])
```
### 训练结果
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.3365 - mae: 1.8211 - val_loss: 6.4201 - val_mae: 1.7893  
Test Loss: 25.8082, Test MAE: 2.9562  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step  
Predictions for first 5 test samples: [19.786161 21.211859 19.15384  34.56086  20.987986]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  

- 第四版代码
```
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载波士顿房价数据集
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建优化模型
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.0005)),  # 更低L2正则化
    layers.Dropout(0.05),  # 更低Dropout率
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.05),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dense(1)
])

# 学习率调度
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    alpha=0.0001
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 编译模型
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, 
          callbacks=[early_stopping], verbose=1)

# 评估模型
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 预测
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])
```
### 训练结果
Epoch 87/1000
41/41 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 5.1438 - mae: 1.6906 - val_loss: 6.9840 - val_mae: 1.8983  
Test Loss: 26.1492, Test MAE: 3.0013  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step  
Predictions for first 5 test samples: [20.502605 21.119947 19.357008 34.657536 21.160563]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  
- 第五版代码
```
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 加载波士顿房价数据集
(X, y), (_, _) = tf.keras.datasets.boston_housing.load_data(test_split=0)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建优化模型
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.05),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.05),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dense(1)
])

# 学习率调度
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    alpha=0.0001
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 编译模型
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, 
                   callbacks=[early_stopping], verbose=1)

# 评估模型
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 预测
y_pred = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", y_pred.flatten())
print("Actual values:", y_test[:5])

# 绘制 MAE 曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Train and Validation MAE vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 Loss 曲线
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
### 训练结果
Epoch 88/1000  
41/41 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0396 - mae: 1.7683 - val_loss: 6.8140 - val_mae: 1.9475  
Test Loss: 27.4981, Test MAE: 3.0507  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step  
Predictions for first 5 test samples: [20.937302 21.09048  19.416035 35.247955 21.223663]  
Actual values: [18.2 21.4 21.5 36.4 20.2]  
<img width="911" height="480" alt="image" src="https://github.com/user-attachments/assets/c4351ded-28e5-4790-9771-85dd53fd5df0" />  
<img width="924" height="470" alt="image" src="https://github.com/user-attachments/assets/769de06c-a3bb-4333-a77b-ef0de3febdd2" />
