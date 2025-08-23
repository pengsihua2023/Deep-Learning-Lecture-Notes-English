## Basic Knowledge of Norms
In mathematics and related fields, a **norm** is a function that assigns a non-negative value to a vector, measuring its "size" or "length" in a vector space. Norms are used in various areas like linear algebra, machine learning, and functional analysis to quantify distances or magnitudes. A norm must satisfy three properties:
1. **Non-negativity**: The norm is zero only for the zero vector, and positive otherwise.
2. **Scalability**: Scaling a vector by a constant scales the norm by the absolute value of that constant.
3. **Triangle inequality**: The norm of the sum of two vectors is at most the sum of their norms.

Here’s a simple explanation with code examples in Python to illustrate common norms for a vector \( v = [v_1, v_2, ..., v_n] \).

### Common Norms
<img width="950" height="378" alt="image" src="https://github.com/user-attachments/assets/e8bbe6bf-2cfc-40b6-8c0c-cc65e2f1a6dd" />
好的 ✅ 我已将中文部分翻译成英文，保持 LaTeX 公式 **完全不变**：

---

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

要不要我帮你也加上 **examples of \$L\_1\$, \$L\_2\$, and \$L\_\infty\$ norms** 的英文版本？这样你的英文版会更完整。


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
