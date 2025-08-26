## CNN

Convolutional Neural Network (CNN)

* Importance: CNN is the cornerstone of computer vision, widely used in image recognition, autonomous driving, etc. It is suitable for demonstrating the practical power of deep learning.
* Core concepts:
  CNN uses the “convolution” operation, like a “magnifying glass” scanning the image to extract features (such as edges, shapes).
  The pooling layer reduces the size of data, retains important information, and decreases computation.
  Finally, the fully connected layer is used for classification or prediction.
* Applications: Image classification (cat vs. dog recognition), face recognition, medical image analysis.

  <img width="708" height="353" alt="image" src="https://github.com/user-attachments/assets/c404062e-9dc5-4c41-bf8d-93cf080c6181" />

---

## Mathematical Description of Convolutional Neural Network (CNN)

The core of CNN consists of the following basic operations: **Convolutional Layer**, **Activation Function**, **Pooling Layer**, and finally the **Fully Connected Layer**. We describe them one by one.

---

### 1. Convolutional Layer

Let the input feature map be

$$
\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}
$$

where \$H\$ is the height, \$W\$ is the width, and \$C\_{in}\$ is the number of input channels.

The convolution kernel (filter) is

$$
\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}
$$

where \$k\_h, k\_w\$ are the kernel size, and \$C\_{out}\$ is the number of output channels.

The convolution operation is defined as:

$$
Y_{i,j,c_{out}} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c_{in}=0}^{C_{in}-1} 
X_{i+m, j+n, c_{in}} \cdot K_{m,n,c_{in},c_{out}} + b_{c_{out}}
$$

where \$b\_{c\_{out}}\$ is the bias term. The output feature map is

$$
\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C_{out}}
$$

The specific dimensions depend on stride and padding.

---

### 2. Activation Function

A commonly used activation function is ReLU (Rectified Linear Unit):

$$
f(z) = \max(0, z)
$$

Applied to the convolution output:

$$
Z_{i,j,c} = f(Y_{i,j,c})
$$

---

### 3. Pooling Layer

The pooling operation is used to reduce the size of the feature map.
For example, in Max Pooling:

$$
P_{i,j,c} = \max_{0 \leq m < p_h,  0 \leq n < p_w} Z_{i \cdot s + m,  j \cdot s + n,  c}
$$

where \$p\_h, p\_w\$ are the pooling window sizes, and \$s\$ is the stride.

---

### 4. Fully Connected Layer

After several layers of convolution and pooling, we obtain a flattened feature vector:

$$
\mathbf{x} \in \mathbb{R}^d
$$

The fully connected layer output is:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

where \$W \in \mathbb{R}^{k \times d}\$, \$\mathbf{b} \in \mathbb{R}^k\$.

---

### 5. Classification Layer (Softmax)

In classification tasks, the final output is a probability distribution through Softmax:

![Softmax formula](https://latex.codecogs.com/png.latex?\hat{y}_i%20=%20\frac{\exp\(y_i\)}{\sum_{j=1}^{k}%20\exp\(y_j\)})

---

