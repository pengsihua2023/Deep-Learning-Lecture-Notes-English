## Basic Knowledge of Norms

### What is a Norm?

A norm is a function in mathematics used to measure the "size" or "length" of a vector or a matrix. It has properties such as non-negativity, homogeneity (absolute scalability), and the triangle inequality. Norms are widely used in mathematics, physics, and computer science, for example, to measure errors, in optimization problems, or for regularization in machine learning.

---

For a vector

$$
x = [x_1, x_2, \ldots, x_n] \in \mathbb{R}^n,
$$

a norm is a function

$$
\|\cdot\| : \mathbb{R}^n \to \mathbb{R},
$$

satisfying the following properties:

1. **Non-negativity**

$$
\|x\| \geq 0, \quad \text{and} \quad \|x\| = 0 \iff x = 0
$$

2. **Absolute homogeneity**

$$
\|\alpha x\| = |\alpha| \ ⋅ \|x\|, \quad \forall \alpha \in \mathbb{R}
$$

3. **Triangle inequality**

$$
\|x + y\| \leq \|x\| + \|y\|, \quad \forall x, y \in \mathbb{R}^n
$$

---

### Common \$L\_p\$ Norm

$$
\lVert x \rVert_p = \left( \sum_{i=1}^n \lvert x_i \rvert^p \right)^{\frac{1}{p}}, p \ge 1
$$

---



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
<img width="824" height="216" alt="image" src="https://github.com/user-attachments/assets/551582aa-eefe-4774-bac8-9e399483e15f" />


This code keeps it simple and demonstrates how norms work for a vector. Norms are foundational in measuring distances (e.g., in machine learning for error calculations) or ensuring stability in numerical methods. If you want a deeper dive into a specific norm or application, let me know!
