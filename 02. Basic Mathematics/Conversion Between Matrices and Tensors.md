# Mathematical Fundamentals: Conversion Between Matrices and Tensors
In deep learning, conversion between matrices and tensors is a common operation because a matrix is a 2nd-order tensor, while a tensor is a more general multidimensional data structure (encompassing scalars, vectors, matrices, and higher dimensions).



## 📖 **1. Relationship Between Matrices and Tensors**
- **Matrix**: A two-dimensional array with shape `(m, n)`, representing a 2nd-order tensor. For example, a 4x3 matrix `[[0 1 0], [1 0 1], [0 0 1], [1 1 0]]` is a 2nd-order tensor with shape `(4, 3)`.
- **Tensor**: A mathematical object for multidimensional data, with its rank indicating the number of dimensions:
  - 0th-order: Scalar (e.g., `5`).
  - 1st-order: Vector (e.g., `[1, 2, 3]`).
  - 2nd-order: Matrix (e.g., `[[1, 2], [3, 4]]`).
  - 3rd-order and higher: Multidimensional arrays (e.g., `(2, 3, 4)`).
- **Relationship**:
  - A matrix is a special case of a tensor (2nd-order tensor).
  - In `NumPy`, matrices and tensors are both represented by `numpy.ndarray`, requiring no explicit conversion.
  - In deep learning frameworks (e.g., `TensorFlow`, `PyTorch`), tensors are specialized objects (`tf.Tensor` or `torch.Tensor`) that support automatic differentiation and GPU acceleration.



## 📖 **2. Conversion Between Matrices and Tensors**
#### **2.1 In NumPy**
In `NumPy`, matrices (2D arrays) and tensors (`ndarray`) are essentially the same, so a matrix can be directly used as a tensor without explicit conversion.
- **Matrix as Tensor**:
  - Any `NumPy` matrix (2nd-order tensor) can directly participate in tensor operations (e.g., matrix multiplication, norm computation).
  - Example (a 4x3 matrix):
    ```python
    import numpy as np
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]) # 4x3 matrix (2nd-order tensor)
    tensor = matrix # No conversion needed, directly used as a tensor
    print(tensor.shape) # (4, 3)
    print(tensor.ndim) # 2 (2nd-order tensor)
    # Tensor operation: matrix multiplication
    result = tensor @ tensor.T # 4x4 matrix
    print(result)
    ```
- **Tensor to Matrix**:
  - Higher-order tensors can be converted to matrices (2nd-order tensors) via reshaping (`reshape`) or slicing.
  - Example (3D tensor to matrix):
    ```python
    tensor_3d = np.random.randint(0, 2, size=(2, 4, 3)) # 3rd-order tensor
    matrix = tensor_3d[0, :, :] # Extract layer 0, yielding a 4x3 matrix
    print(matrix.shape) # (4, 3)
    ```
- **Shape Adjustment**:
  - Use `.reshape()` or `.expand_dims()` to adjust rank or shape.
  - Example (vector to matrix):
    ```python
    vector = np.array([1, 2, 3]) # 1st-order tensor
    matrix = vector.reshape(1, 3) # 2nd-order tensor (1x3 matrix)
    print(matrix.shape) # (1, 3)
    ```

#### **2.2 In Deep Learning Frameworks**
In `TensorFlow` and `PyTorch`, tensors are specialized objects supporting automatic differentiation and GPU acceleration. Matrices (typically `NumPy` arrays) require explicit conversion to tensors and vice versa.
- **Matrix (NumPy Array) to Tensor**:
  - **TensorFlow**:
    ```python
    import tensorflow as tf
    import numpy as np
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]) # 4x3 matrix
    tensor = tf.convert_to_tensor(matrix, dtype=tf.float32) # Convert to tf.Tensor
    print(tensor.shape) # (4, 3)
    print(tensor.dtype) # float32
    ```
  - **PyTorch**:
    ```python
    import torch
    import numpy as np
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]) # 4x3 matrix
    tensor = torch.from_numpy(matrix).float() # Convert to torch.Tensor
    print(tensor.shape) # torch.Size([4, 3])
    print(tensor.dtype) # torch.float32
    ```
- **Tensor to Matrix (NumPy Array)**:
  - **TensorFlow**:
    ```python
    import tensorflow as tf
    tensor = tf.constant([[0, 1, 0], [1, 0, 1]], dtype=tf.float32) # tf.Tensor
    matrix = tensor.numpy() # Convert to NumPy array
    print(matrix.shape) # (2, 3)
    ```
  - **PyTorch**:
    ```python
    import torch
    tensor = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32) # torch.Tensor
    matrix = tensor.numpy() # Convert to NumPy array
    print(matrix.shape) # (2, 3)
    ```
    - **Note**: In PyTorch, if the tensor is on a GPU, call `.cpu()` first: `tensor.cpu().numpy()`.
- **Shape and Rank Conversion**:
  - **Matrix (2nd-order Tensor) to Higher-order Tensor**:
    - Use `.reshape()` or `.expand_dims()` to increase dimensions.
    - Example:
      ```python
      import numpy as np
      matrix = np.array([[0, 1, 0], [1, 0, 1]]) # 2x3 matrix (2nd-order tensor)
      tensor_3d = matrix.reshape(1, 2, 3) # 1x2x3 3rd-order tensor
      print(tensor_3d.shape) # (1, 2, 3)
      ```
  - **Higher-order Tensor to Matrix**:
    - Extract a 2D portion via slicing or flatten to a matrix.
    - Example:
      ```python
      tensor_3d = np.random.randint(0, 2, size=(2, 4, 3)) # 3rd-order tensor
      matrix = tensor_3d[0] # Extract layer 0, yielding a 4x3 matrix
      print(matrix.shape) # (4, 3)
      ```



## 📖 **3. Applications in Deep Learning**
Conversion between matrices and tensors is widely used in deep learning for data processing, model training, and optimization. Using the example of a 4x3 matrix:
- **Feature Matrix**:
  - A 4x3 matrix may represent 4 samples, each with 3 features.
  - After conversion to a tensor, it can be used in neural network forward propagation.
  - Example:
    ```python
    import torch
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]) # 4x3 matrix
    tensor = torch.from_numpy(matrix).float() # Convert to tensor
    W = torch.randn(3, 2) # 3x2 weight tensor
    output = tensor @ W # Forward propagation
    print(output.shape) # torch.Size([4, 2])
    ```
- **Weight Tensor**:
  - A weight matrix (e.g., `(input dimension, output dimension)`) converted to a tensor participates in gradient computation.
- **Batch Data**:
  - Three-dimensional tensors (e.g., `(batch_size, number of samples, number of features)`) are used for batch processing. A matrix can be converted to a tensor by adding dimensions.
  - Example:
    ```python
    matrix = np.array([[0, 1, 0], [1, 0, 1]]) # 2x3 matrix
    batch_tensor = np.expand_dims(matrix, axis=0) # 1x2x3 tensor
    print(batch_tensor.shape) # (1, 2, 3)
    ```



## 📖 **4. Considerations**
- **Data Types**:
  - Ensure type compatibility during conversion (e.g., `NumPy`’s `int64` to `torch.float32`).
  - Example:
    ```python
    matrix = np.array([[0, 1, 0]], dtype=np.int64)
    tensor = torch.from_numpy(matrix).float() # Convert to float32
    print(tensor.dtype) # torch.float32
    ```
- **Memory Sharing**:
  - `torch.from_numpy` and `NumPy` arrays share memory, so modifying one affects the other.
  - Example:
    ```python
    matrix = np.array([1, 2, 3])
    tensor = torch.from_numpy(matrix)
    matrix[0] = 99
    print(tensor) # tensor([99, 2, 3])
    ```
- **GPU Support**:
  - Deep learning tensors can be moved to GPU (e.g., `tensor.to('cuda')`), while `NumPy` arrays do not support this.
- **Shape Consistency**:
  - Conversion does not alter the shape; a 4x3 matrix remains `(4, 3)` after conversion to a tensor.



## 📖 **5. Summary**
- **Matrix and Tensor Conversion**:
  - **NumPy**: Matrices (`ndarray`) are directly used as 2nd-order tensors without conversion; they can be reshaped into higher-order tensors using `.reshape()` or `.expand_dims()`.
  - **Deep Learning Frameworks**:
    - Matrix to tensor: `tf.convert_to_tensor` (TensorFlow) or `torch.from_numpy` (PyTorch).
    - Tensor to matrix: `tensor.numpy()`.
- **Applications in Deep Learning**:
  - Matrices (feature matrices, weights) converted to tensors participate in neural network training and gradient computation.
  - Shape adjustments (e.g., matrix to 3D tensor) support batch processing.
- **Considerations**:
  - Ensure data type compatibility (e.g., `float32`).
  - Be aware of memory sharing and GPU support.
