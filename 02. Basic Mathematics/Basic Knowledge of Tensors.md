# Mathematical Fundamentals: Basics of Tensors

## 📖 **1. Definition of Tensors**
- A **tensor** is a concept in mathematics and physics, generalizing scalars (0th-order), vectors (1st-order), and matrices (2nd-order) to describe the geometric and algebraic properties of multidimensional data. A tensor can be viewed as a multidimensional array, with elements accessed via multiple indices, suitable for representing complex data relationships and transformations.
- **Key Characteristics**:
  - **Rank (Order)**: The rank of a tensor indicates the number of dimensions:
    - 0th-order: Scalar (e.g., `5`).
    - 1st-order: Vector (e.g., `[1, 2, 3]`).
    - 2nd-order: Matrix (e.g., `[[1, 2], [3, 4]]`).
    - 3rd-order and higher: Multidimensional tensors (e.g., a 3D array with shape `(2, 3, 4)`).
  - **Shape**: Describes the size of each dimension, e.g., `(m, n)` for a matrix or `(d, h, w)` for a 3D tensor.
  - **Mathematical Properties**: Tensors support linear algebra operations (e.g., dot product, matrix multiplication, tensor product) and have transformation rules (e.g., covariance or contravariance under coordinate changes).
  - **Applications**: Widely used in physics (describing forces, stresses), machine learning (representing features, weights), and Meta-Learning (embedding vectors, task data).



## 📖 **2. Tensors in Python and NumPy**
In Python, tensors are typically implemented using `NumPy` arrays (`numpy.ndarray`) or tensor objects in deep learning frameworks (e.g., `TensorFlow`, `PyTorch`). Given your focus on `NumPy`, I’ll emphasize tensors in `NumPy`:
- **NumPy Tensors**:
  - In `NumPy`, tensors are represented by `ndarray`, and any multidimensional array can be considered a tensor.
  - Features:
    - **Efficiency**: Implemented in C, supporting vectorized operations.
    - **Multidimensional Support**: Can be 1st-order (vector), 2nd-order (matrix), or higher-order.
    - **Operations**: Supports linear algebra operations (e.g., `np.dot`, `np.linalg.norm`).
  - Example (a 4x3 array as a 2nd-order tensor):
    ```python
    import numpy as np
    tensor = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]) # 2nd-order tensor
    print(tensor)
    print("Shape:", tensor.shape) # (4, 3)
    print("Rank:", tensor.ndim) # 2 (2nd-order)
    ```
    **Output**:
    ```
    [[0 1 0]
     [1 0 1]
     [0 0 1]
     [1 1 0]]
    Shape: (4, 3)
    Rank: 2
    ```
- **Tensors in Deep Learning Frameworks**:
  - In `TensorFlow` (`tf.Tensor`) or `PyTorch` (`torch.Tensor`), tensors are specialized objects supporting automatic differentiation, GPU acceleration, etc.
  - Example:
    ```python
    import tensorflow as tf
    tensor = tf.constant([[0, 1, 0], [1, 0, 1]]) # TensorFlow tensor
    print(tensor.shape) # (2, 3)
    ```



## 📖 **3. Relationship Between Tensors and Arrays**
- **Array**: A computer science data structure, emphasizing storage and manipulation, capable of holding any data type (e.g., numbers, strings).
- **Tensor**: A mathematical concept, emphasizing linear algebra properties, typically containing numerical data to represent multidimensional relationships and transformations.
- **In NumPy**:
  - Arrays (`ndarray`) are directly used as tensors:
    - A 1D array (e.g., `[1, 2, 3]`) is a 1st-order tensor (vector).
    - A 2D array (e.g., `[[0, 1, 0], [1, 0, 1]]`) is a 2nd-order tensor (matrix).
    - A 3D array (e.g., shape `(2, 3, 4)`) is a 3rd-order tensor.
  - Your 4x3 array `[[0 1 0], [1 0 1], [0 0 1], [1 1 0]]` is a 2D array and also a 2nd-order tensor.
- **Conversion**:
  - `NumPy` arrays can be used as tensors without conversion.
  - They can be converted to deep learning tensors via `tf.convert_to_tensor` or `torch.from_numpy`.



## 📖 **4. Common Tensor Operations**
Here are some common operations for `NumPy` tensors, using your 4x3 tensor as an example:
- **Indexing and Slicing**:
  ```python
  tensor = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
  print(tensor[0, :]) # Row 0: [0 1 0]
  print(tensor[:, 1]) # Column 1: [1 0 0 1]
  ```
- **Linear Algebra Operations**:
  ```python
  print(np.dot(tensor, tensor.T)) # Matrix multiplication (4x4 result)
  print(np.linalg.norm(tensor[0, :])) # Norm of the first row
  ```
- **Statistical Operations**:
  ```python
  print(np.mean(tensor, axis=0)) # Mean per column: [0.5 0.5 0.5]
  print(np.sum(tensor, axis=1)) # Sum per row: [1 2 1 2]
  ```



## 📖 **5. Summary**
- **What is a Tensor?**:
  - A tensor is a multidimensional data object in mathematics, generalizing scalars, vectors, and matrices, with a rank (order) and shape, used to describe linear algebra relationships.
  - In `NumPy`, tensors are implemented as `ndarray`, supporting vector, matrix, and higher-order operations.
- **Relationship with Arrays**:
  - Arrays are general-purpose data structures, while tensors are mathematical concepts; in `NumPy`, arrays (like your 4x3 array) are directly used as tensors.
  - A 1st-order tensor (vector) corresponds to a 1D array, a 2nd-order tensor (matrix) corresponds to a 2D array, and so on.
- **Meta-Learning Context**:
  - Tensors are used to represent feature matrices (e.g., your 4x3 2nd-order tensor), labels (1st-order tensors), confusion matrices (2nd-order tensors), or multi-task data (3rd-order tensors).
  - Common operations include prototype computation, distance metrics, and confusion matrix generation.
- **Your Array**:
  - `[[0 1 0], [1 0 1], [0 0 1], [1 1 0]]` is a 4x3 2nd-order tensor, potentially representing a feature matrix, suitable for Meta-Learning tasks (e.g., ProtoNet).
