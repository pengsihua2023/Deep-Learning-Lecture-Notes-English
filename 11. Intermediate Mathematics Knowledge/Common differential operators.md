
# Common Differential Operators

### 📖 1. **Gradient**

* Object: Scalar field \$f(x,y,z)\$
* Definition:

$$
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right)
$$

* Result: A vector, representing the direction and rate of the fastest change at that point.
* Example: Gradient of a temperature field → direction of heat conduction.



### 📖 2. **Divergence**

* Object: Vector field \$\vec{A}(x,y,z)\$
* Definition:

$$
\nabla \cdot \vec{A} = \frac{\partial A_x}{\partial x} + \frac{\partial A_y}{\partial y} + \frac{\partial A_z}{\partial z}
$$

* Result: A scalar, describing the intensity of a "source" or "sink" at a point.
* Example: Divergence of the electric field is related to charge density.



### 📖 3. **Curl**

* Object: Vector field \$\vec{A}(x,y,z)\$
* Definition:

$$
\nabla \times \vec{A} =
\begin{vmatrix}
\hat{i} & \hat{j} & \hat{k} \\
\dfrac{\partial}{\partial x} & \dfrac{\partial}{\partial y} & \dfrac{\partial}{\partial z} \\
A_x & A_y & A_z
\end{vmatrix}
$$

* Result: A vector, describing the “rotation tendency” of the field.
* Example: Magnetic field is the result of the curl of current.



### 📖 4. **Laplacian**

* Object: Scalar field or vector field
* Definition:

$$
\nabla^2 f = \nabla \cdot (\nabla f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
$$

* Result: A scalar (for a scalar field) or a vector (for a vector field).
* Example: Poisson equation, heat conduction equation, wave equation.



### 📖 5. **Common Operator Identities in Physics**

* **\$\nabla \cdot (\nabla \times \vec{A}) = 0\$**
  (The divergence of a curl is always zero)
* **\$\nabla \times (\nabla f) = 0\$**
  (The curl of a gradient is always zero)
* **\$\nabla^2 f = \nabla \cdot (\nabla f)\$**
  (The Laplacian is the divergence of a gradient)

---

<div align="center">

## 📖 **Summary Table:**

| Operator   | Symbol                    | Input               | Output              | Physical Meaning        |
| ---------- | ------------------------- | ------------------- | ------------------- | ----------------------- |
| Gradient   | \$\nabla f\$              | Scalar field        | Vector field        | Direction of max growth |
| Divergence | \$\nabla \cdot \vec{A}\$  | Vector field        | Scalar field        | Source/sink strength    |
| Curl       | \$\nabla \times \vec{A}\$ | Vector field        | Vector field        | Local rotational trend  |
| Laplacian  | \$\nabla^2 f\$            | Scalar/Vector field | Scalar/Vector field | Second-order variation  |

</div>

---

# Vector Differential Operator Identities

### 📖 I. Basic Identities

1. **Curl of a gradient is zero**

\$\nabla \times (\nabla f) = 0\$

2. **Divergence of a curl is zero**

\$\nabla \cdot (\nabla \times \vec{A}) = 0\$

3. **Laplacian**

\$\nabla^2 f = \nabla \cdot (\nabla f)\$



### 📖 II. Common Expansion Formulas

1. **Product rule for gradient**

\$\nabla (fg) = f \nabla g + g \nabla f\$

2. **Product rule for divergence**

\$\nabla \cdot (f \vec{A}) = f(\nabla \cdot \vec{A}) + \nabla f \cdot \vec{A}\$

3. **Product rule for curl**

\$\nabla \times (f \vec{A}) = f(\nabla \times \vec{A}) + \nabla f \times \vec{A}\$



### 📖 III. Identities for Two Vector Fields

1. **Divergence of a cross product**

\$\nabla \cdot (\vec{A}\times \vec{B}) = \vec{B}\cdot (\nabla \times \vec{A}) - \vec{A}\cdot (\nabla \times \vec{B})\$

2. **Curl of a cross product**

\$\nabla \times (\vec{A}\times \vec{B}) = \vec{A}(\nabla \cdot \vec{B}) - \vec{B}(\nabla \cdot \vec{A}) + (\vec{B}\cdot \nabla)\vec{A} - (\vec{A}\cdot \nabla)\vec{B}\$

3. **Divergence of a product**

\$\nabla \cdot (\vec{A} f) = (\nabla \cdot \vec{A}) f + \vec{A}\cdot (\nabla f)\$



### 📖 IV. Vector Laplacian

For a vector field \$\vec{A}\$, the identity is:

\$\nabla^2 \vec{A} = \nabla (\nabla \cdot \vec{A}) - \nabla \times (\nabla \times \vec{A})\$


