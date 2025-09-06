
# Five Common Mathematical Physics Equations

## 📖 1. **Laplace / Poisson Equation**

**Equation:**

$$
\nabla^2 u = 0 \quad \text{(Laplace)}, \quad \nabla^2 u = -f(x,y,z) \quad \text{(Poisson)}
$$

**Steps:**

1. Assume \$u(x,y)=X(x)Y(y)\$.
2. Separate variables, obtaining two ODEs (one trigonometric solution, one hyperbolic solution).
3. Apply boundary conditions, keeping the valid solutions.
4. Final solution expressed as **Fourier series expansion**.

**General solution:**

$$
u(x,y)=\sum_{n=1}^\infty C_n \sin\left(\frac{n\pi x}{L}\right)\sinh\left(\frac{n\pi}{L}y\right)
$$

**Features:**
Applicable to **steady-state problems** (electric field, temperature field).



## 📖 2. **Heat Conduction Equation**

**Equation:**

$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
$$

**Steps:**

1. Assume separation of variables \$u(x,t)=X(x)T(t)\$.

2. Obtain: \$T'(t)=-\alpha\lambda T(t),\quad X''+\lambda X=0\$

3. Spatial solution: sine series; temporal solution: exponential decay.

4. Use Fourier expansion to determine coefficients.

**General solution:**

$$
u(x,t)=\sum_{n=1}^\infty b_n \sin\left(\frac{n\pi x}{L}\right)e^{-\alpha (n\pi/L)^2 t}
$$

**Features:**
Temperature/concentration **gradually smooths out over time**, approaching stability.



## 📖 3. **Wave Equation**

**Equation:**

$$
\frac{\partial^2 u}{\partial t^2} = v^2 \nabla^2 u
$$

**Steps:**

1. Separate variables: \$u(x,t)=X(x)T(t)\$.

2. Obtain: \$X''+\lambda X=0,\quad T''+v^2\lambda T=0\$

3. Spatial solution: sine functions; temporal solution: sine/cosine.

4. Expand using initial conditions into Fourier series.

**General solution:**

$$
u(x,t)=\sum_{n=1}^\infty \Big[A_n\cos\Big(\tfrac{n\pi v}{L}t\Big)+B_n\sin\Big(\tfrac{n\pi v}{L}t\Big)\Big]\sin\left(\frac{n\pi x}{L}\right)
$$

**Features:**
**Standing wave solutions**, oscillation/wave propagation.



## 📖 4. **Schrödinger Equation**

**Equation:**

$$
-\frac{\hbar^2}{2m}\nabla^2 \psi + V\psi = E\psi
$$

**Steps:**

1. Identify potential field \$V(x)\$.
2. Solve the equation within the region (commonly sine or exponential solutions).
3. Boundary conditions: \$\psi\$ must be continuous and finite.
4. Obtain energy quantization condition.

**General solution (infinite potential well):**

$$
\psi_n(x)=\sin\left(\frac{n\pi x}{L}\right),\quad E_n=\frac{n^2\pi^2\hbar^2}{2mL^2}
$$

**Features:**
Essentially an **eigenvalue problem** → Energy is **quantized**.



## 📖 5. **Electromagnetic Wave Equation**

**Equation:**

$$
\nabla^2 \vec{E}-\frac{1}{c^2}\frac{\partial^2 \vec{E}}{\partial t^2}=0
$$

**Steps:**

1. Recognize as the standard wave equation.

2. Write traveling wave solution: \$E(x,t)=f(x-ct)+g(x+ct)\$

3. For monochromatic waves: \$E(x,t)=E\_0\cos(kx-\omega t),\quad \omega=ck\$

**Features:**
Describes light/electromagnetic waves propagating at **the speed of light**.


## 📖 Summary

| Equation             | Type       | General Solution Form                   | Features                    |
| -------------------- | ---------- | --------------------------------------- | --------------------------- |
| Laplace/Poisson      | Elliptic   | Trigonometric + Hyperbolic expansion    | Steady-state distribution   |
| Heat Conduction      | Parabolic  | Fourier sine series × exponential decay | Diffusion, stabilization    |
| Wave Equation        | Hyperbolic | Fourier sine series × sine/cosine       | Oscillation, standing waves |
| Schrödinger          | Elliptic   | Eigenfunctions (sine/exponential)       | Energy quantization         |
| Electromagnetic Wave | Hyperbolic | Traveling/standing waves                | Light-speed propagation     |



