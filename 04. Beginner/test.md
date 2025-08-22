下面给出全连接神经网络（Fully Connected Neural Network，简称 FCNN 或 MLP）的数学描述：

---

## 1. 网络结构

一个典型的全连接神经网络由若干 **层 (layers)** 构成：

* 输入层（input layer）
* 一个或多个隐藏层（hidden layers）
* 输出层（output layer）

在全连接结构中，**每一层的每个神经元都与上一层的所有神经元相连**。

---

## 2. 数学符号

设：

* 输入向量为

$$
\mathbf{x} \in \mathbb{R}^{d}
$$

* 第 $l$ 层有 $n_l$ 个神经元，输出记为

$$
\mathbf{h}^{(l)} \in \mathbb{R}^{n_l}
$$

* 权重矩阵与偏置为

$$
\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}, \quad \mathbf{b}^{(l)} \in \mathbb{R}^{n_l}
$$

* 激活函数为

$$
\sigma(\cdot)
$$

---

## 3. 前向传播 (Forward Propagation)

输入层记为

$$
\mathbf{h}^{(0)} = \mathbf{x}
$$

对于第 $l$ 层 ($l=1,2,\dots,L$)，有：

1. **线性变换：**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
$$

2. **非线性激活：**

$$
\mathbf{h}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)
$$

最终，输出层结果为：

$$
\hat{\mathbf{y}} = \mathbf{h}^{(L)}
$$

---

## 4. 损失函数 (Loss Function)

训练时，给定目标输出 $\mathbf{y}$，常用损失函数包括：

* **回归问题：** 均方误差（MSE）

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)}\|^2
$$

* **分类问题：** 交叉熵损失（Cross-Entropy）

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{k=1}^K y_k \log \hat{y}_k
$$

---

## 5. 参数更新 (Backpropagation + Gradient Descent)

通过反向传播 (Backpropagation) 计算损失函数对参数的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

再用梯度下降或其变种（如 Adam, SGD, RMSProp）更新：

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

其中 $\eta$ 为学习率。

---

✅ 总结来说，全连接神经网络可以抽象为：

$$
\hat{\mathbf{y}} = f(\mathbf{x}; \Theta) = \sigma^{(L)}\Big(\mathbf{W}^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) \cdots ) + \mathbf{b}^{(L)}\Big)
$$

其中 $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)} \mid l=1,\dots,L\}$ 为模型参数。

---

要不要我再帮你画一个**MLP的结构示意图**（包含输入层、隐藏层、输出层和连接关系）？

