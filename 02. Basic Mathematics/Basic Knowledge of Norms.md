In mathematics and related fields, a **norm** is a function that assigns a non-negative value to a vector, measuring its "size" or "length" in a vector space. Norms are used in various areas like linear algebra, machine learning, and functional analysis to quantify distances or magnitudes. A norm must satisfy three properties:
1. **Non-negativity**: The norm is zero only for the zero vector, and positive otherwise.
2. **Scalability**: Scaling a vector by a constant scales the norm by the absolute value of that constant.
3. **Triangle inequality**: The norm of the sum of two vectors is at most the sum of their norms.

Here’s a simple explanation with code examples in Python to illustrate common norms for a vector \( v = [v_1, v_2, ..., v_n] \).

### Common Norms
1. **L1 Norm (Manhattan Norm)**: Sum of absolute values of the vector components.
   \[
   ||v||_1 = |v_1| + |v_2| + ... + |v_n|
   \]
2. **L2 Norm (Euclidean Norm)**: Square root of the sum of squared components (like the length of a vector in Euclidean space).
   \[
   ||v||_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}
   \]
3. **Infinity Norm (Max Norm)**: The maximum absolute value of the vector components.
   \[
   ||v||_\infty = \max(|v_1|, |v_2|, ..., |v_n|)
   \]

### Simple Python Code to Compute Norms
```python
import math

# Example vector
vector = [3, -4, 5]

# L1 Norm: Sum of absolute values
def l1_norm(v):
    return sum(abs(x) for x in v)

# L2 Norm: Square root of sum of squares
def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))

# Infinity Norm: Maximum absolute value
def inf_norm(v):
    return max(abs(x) for x in v)

# Compute norms
print("Vector:", vector)
print("L1 Norm:", l1_norm(vector))      # Output: 12
print("L2 Norm:", l2_norm(vector))      # Output: ~7.071
print("Infinity Norm:", inf_norm(vector)) # Output: 5
```

### Explanation of Code
- **Input**: The vector is a list of numbers (e.g., `[3, -4, 5]`).
- **L1 Norm**: Adds the absolute values: \( |3| + |-4| + |5| = 3 + 4 + 5 = 12 \).
- **L2 Norm**: Computes the square root of the sum of squares: \( \sqrt{3^2 + (-4)^2 + 5^2} = \sqrt{9 + 16 + 25} = \sqrt{50} \approx 7.071 \).
- **Infinity Norm**: Takes the maximum absolute value: \( \max(|3|, |-4|, |5|) = 5 \).

This code keeps it simple and demonstrates how norms work for a vector. Norms are foundational in measuring distances (e.g., in machine learning for error calculations) or ensuring stability in numerical methods. If you want a deeper dive into a specific norm or application, let me know!
