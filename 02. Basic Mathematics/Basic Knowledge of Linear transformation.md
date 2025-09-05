# Linear Transformation
A linear transformation is a core concept in **linear algebra**, essentially a class of functions that preserve the "linear structure."


## 📖 1. Definition

Let $V, W$ be vector spaces over the same field $\mathbb{F}$, and consider the mapping

$$
T: V \to W
$$

If for any $u, v \in V$ and any scalars $a, b \in \mathbb{F}$, we have

$$
T(au + bv) = aT(u) + bT(v),
$$

then $T$ is called a **linear transformation**.



## 📖 2. Intuitive Understanding

The "linearity" of a linear transformation is reflected in the following two aspects:

* **Preservation of vector addition**:

$T(u+v) = T(u) + T(v)$

  
* **Preservation of scalar multiplication**:

$T(cu) = cT(u)$

In other words, **a linear transformation does not destroy the addition and scalar multiplication structure of a vector space**.



## 📖 3. Matrix Representation

In the finite-dimensional case, any linear transformation can be represented by a matrix.

For example:
If $T: \mathbb{R}^n \to \mathbb{R}^m$, then there exists an $m \times n$ matrix $A$ such that

$$
T(x) = Ax
$$

where $x \in \mathbb{R}^n$.



## 📖 4. Common Examples

1. **Scaling**:

$T(x) = 2x$  

   Multiply all vectors by 2.

2. **Rotation in $\mathbb{R}^2$**:



$$
T(x) = 
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

   Rotate the vector around the origin by angle $\theta$.

3. **Projection**:

${T(x,y) = (x,0)}$

   Project points in $\mathbb{R}^2$ onto the x-axis.



## 📖 5. Comparison with Nonlinear Mappings

For example $T(x) = x^2$ is not a linear transformation, because

$$
T(x+y) \neq T(x) + T(y)
$$



## 📖 Summary
A linear transformation is a **mapping that preserves vector addition and scalar multiplication**. In linear algebra, it is equivalent to matrix multiplication and serves as an important tool in the study of geometric transformations, signal processing, machine learning, and other fields.
