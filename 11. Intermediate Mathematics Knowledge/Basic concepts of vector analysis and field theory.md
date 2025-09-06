# Basic concepts of vector analysis and field theory
## I. Basic Concepts of Vector Analysis

### 1. Scalars and Vectors

* **Scalar**: Has only magnitude, no direction, e.g., temperature, mass, density.
* **Vector**: Has both magnitude and direction, e.g., velocity, force, electric field intensity.

### 2. Vector Operations

* **Dot Product (scalar product):**

  $$
  \vec{A}\cdot \vec{B} = |\vec{A}||\vec{B}|\cos\theta
  $$

  Results in a scalar, commonly used for projection and work calculation.

* **Cross Product (vector product):**

$$
\vec{A}\times \vec{B} = |\vec{A}||\vec{B}|\sin\theta \hat{n}
$$

Results in a vector perpendicular to the plane of the two vectors, commonly used for torque and magnetic force.

### 3. Common Differential Operators

* **Gradient**:

$$
\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right)
$$

Describes the direction of the fastest change of a scalar field.

* **Divergence**:

$$
\nabla \cdot \vec{A}
$$

Describes the "source" or "sink" strength of a vector field at a point.

* **Curl**:

$$
\nabla \times \vec{A}
$$

Describes the rotational property of a vector field.

---

## II. Integral Theorems

These theorems connect the **differential form** with the **integral form** and form the foundation of field theory.

1. **Divergence Theorem (Gauss’s Theorem)**

<div align="center">
<img width="212" height="53" alt="image" src="https://github.com/user-attachments/assets/d2214d9a-3e50-48ab-a6a7-926722424bf2" />
</div>

Converts a volume integral into a surface integral over the boundary.

2. **Stokes’ Theorem**

$$
\iint_S (\nabla \times \vec{A}) \cdot d\vec{S} = \oint_{\partial S} \vec{A}\cdot d\vec{l}
$$

Converts a surface integral into a line integral over the boundary.

3. **Gradient Theorem (part of Green’s Theorem)**

$$
\int_{P_1}^{P_2} \nabla f \cdot d\vec{l} = f(P_2) - f(P_1)
$$

---

## III. Basic Concepts of Field Theory

### 1. Classification of Fields

* **Scalar Field**: Each point corresponds to a scalar (e.g., temperature field).
* **Vector Field**: Each point corresponds to a vector (e.g., electric field, velocity field).

### 2. Applications in Electromagnetic Fields

* **Electric Field**: \$\vec{E} = -\nabla \varphi\$
* **Magnetic Field**: Produced by current or changing electric field, satisfying \$\nabla \cdot \vec{B} = 0\$.
* **Maxwell’s Equations**: The core of field theory, unifying electricity and magnetism.

---

## IV. Physical Intuition

* **Divergence > 0**: The point is a "source," e.g., electric charge producing an electric field.
* **Divergence < 0**: The point is a "sink," e.g., fluid flowing in.
* **Nonzero curl**: Indicates a "rotational" effect in the field, e.g., vortex flow, magnetic field surrounding current.



