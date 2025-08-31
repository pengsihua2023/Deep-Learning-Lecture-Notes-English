## Deep Backward Stochastic Differential Equation (Deep BSDE) Method

The Deep Backward Stochastic Differential Equation (Deep BSDE) Method is a deep learning–based numerical method for solving high-dimensional (even hundreds of dimensions) nonlinear parabolic partial differential equations (PDEs), particularly in scenarios where traditional grid-based methods (such as finite differences) fail due to the curse of dimensionality. It was proposed by Jiequn Han, Arnulf Jentzen, and Weinan E in 2017. The method transforms the PDE into a backward stochastic differential equation (BSDE) via the Feynman-Kac theorem, then uses neural networks to approximate the solution and gradient of the BSDE, and trains the model by minimizing the terminal condition loss. Deep BSDE is mesh-free, making it suitable for applications in financial pricing (e.g., high-dimensional Black-Scholes), quantum mechanics, and control problems, but it has high computational cost and its convergence depends on time discretization and network architecture.

Compared with the Deep Galerkin Method (DGM) or Deep Ritz Method (DRM), Deep BSDE is more suitable for time-dependent parabolic PDEs and naturally handles randomness, but requires simulating stochastic paths, which may introduce variance.

### Mathematical Description

Consider a general semilinear parabolic PDE:

$$
\partial_t u(t,x) + \mu(t,x) \cdot \nabla_x u(t,x) + \tfrac{1}{2} \text{Tr}\big(\sigma(t,x)\sigma(t,x)^* \text{Hess}_x u(t,x)\big)  +  f(t,x,u(t,x),[\sigma(t,x)]^* \nabla_x u(t,x)) = 0,
$$

for \$t \in \[0,T], ; x \in \mathbb{R}^d\$, with terminal condition \$u(T,x) = g(x)\$.
Here, \$\mu : \[0,T] \times \mathbb{R}^d \to \mathbb{R}^d\$ is the drift term,
\$\sigma : \[0,T] \times \mathbb{R}^d \to \mathbb{R}^{d \times d}\$ is the diffusion matrix,
\$f\$ is the nonlinear term, \$\text{Tr}(\cdot)\$ denotes the trace operator, \$A^\*\$ denotes transpose, and \$\text{Hess}\_x u\$ is the Hessian matrix.

By the Feynman-Kac theorem, this PDE can be represented as a forward–backward stochastic differential equation (FBSDE) system:

* **Forward SDE (path process):**

$$
X_t = x + \int_0^t \mu(s,X_s) ds + \int_0^t \sigma(s,X_s) dW_s,
$$

where \$W\_s\$ is a \$d\$-dimensional Wiener process.

* **Backward SDE (value process):**

$$
Y_t = g(X_T) + \int_t^T f(s,X_s,Y_s,Z_s) ds - \int_t^T Z_s^* dW_s,
$$

where \$Y\_t = u(t,X\_t), \quad Z\_t = \[\sigma(t,X\_t)]^\* \nabla\_x u(t,X\_t)\$ (gradient process).

---

Deep BSDE approximates the BSDE via time discretization (Euler scheme):
The time interval $\[0,T]\$ is divided into \$N\$ steps with step size \$\Delta t = T/N\$,
Brownian increments are simulated as \$\Delta W\_n \sim \mathcal{N}(0,\Delta t I\_d)\$.
Then, neural networks are used to parameterize the initial estimate \$Y\_0^\theta\$ (a scalar) and the gradient process \$Z\_n^\theta(t\_n,X\_n)\$ (a neural network at each time step).

The problem becomes minimizing the empirical loss:

$$
J(\theta) = \mathbb{E}\Big[ \big| Y_T^\theta - g(X_T) \big|^2 \Big],
$$

where \$Y\_T^\theta\$ is computed via backward iteration:

$$
Y_{n+1}^\theta = Y_n^\theta - f(t_n, X_n, Y_n^\theta, Z_n^\theta)\Delta t + (Z_n^\theta)^* \Delta W_n,
$$

starting from \$Y\_0^\theta\$.
The expectation is approximated by Monte Carlo sampling (batch simulation of paths), and \$\theta\$ is trained using optimizers such as Adam. This is equivalent to solving a stochastic control problem, where \$Z\$ is the control variable.

---

### Code Implementation

Below is a simple 1D Deep BSDE example implemented in PyTorch, used to solve a nonlinear PDE:

$$
\partial_t u + \tfrac{1}{2} \partial_{xx} u + u^3 = 0,
$$

for \$t \in \[0,1], ; x \in \mathbb{R}\$, with terminal condition

$$
u(1,x) = \cos\left(\tfrac{\pi x}{2}\right).
$$

This is a simplified version of the Allen-Cahn equation variant, with the true solution approximating a smooth function. The code includes neural network definition, path simulation, and training loop.
## Code 
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Neural network to approximate Z_t (gradient process), 
# can be shared or independent across time steps
class ZNet(nn.Module):
    def __init__(self, d=1, hidden_size=32):  # d is dimension, here 1D
        super(ZNet, self).__init__()
        self.fc1 = nn.Linear(d + 1, hidden_size)  # input (t, x)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d)   # output Z \in \mathbb{R}^d
    
    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        return self.fc_out(h)

# Deep BSDE solver
class DeepBSDE:
    def __init__(self, d=1, T=1.0, N=20, batch_size=256, lr=0.01):
        self.d = d  # dimension
        self.T = T  # terminal time
        self.N = N  # number of time steps
        self.dt = T / N
        self.batch_size = batch_size
        self.y0 = nn.Parameter(torch.tensor([0.5]))  # initial Y0 (u(0,0))
        self.z_net = ZNet(d=d)  # shared Z network
        self.optimizer = torch.optim.Adam(list(self.z_net.parameters()) + [self.y0], lr=lr)
    
    def f(self, y):  # nonlinear term f(y) = y^3
        return y ** 3
    
    def g(self, x):  # terminal condition g(x) = cos(pi x / 2)
        return torch.cos(np.pi * x / 2)
    
    def simulate_paths(self, x0=0.0):
        # Simulate Brownian paths
        dw = torch.sqrt(torch.tensor(self.dt)) * torch.randn((self.batch_size, self.N, self.d))
        x = torch.zeros((self.batch_size, self.N+1, self.d))
        x[:, 0, :] = x0
        for n in range(self.N):
            x[:, n+1, :] = x[:, n, :] + dw[:, n, :]  # sigma=1, mu=0
        return x, dw
    
    def loss(self, x, dw):
        y = self.y0.repeat(self.batch_size, 1)  # Y_0
        t = torch.zeros((self.batch_size, 1))
        for n in range(self.N):
            z = self.z_net(t, x[:, n, :])  # Z_n
            y = y - self.f(y) * self.dt + torch.sum(z * dw[:, n, :], dim=1, keepdim=True)
            t = t + self.dt
        return torch.mean((y - self.g(x[:, -1, :])) ** 2)
    
    def train(self, epochs=2000, x0=0.0):
        losses = []
        for epoch in range(epochs):
            x, dw = self.simulate_paths(x0)
            loss_val = self.loss(x, dw)
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            losses.append(loss_val.item())
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val.item():.4f}, y0: {self.y0.item():.4f}")
        return losses

# Main program
dbsde = DeepBSDE(d=1, T=1.0, N=50, batch_size=512, lr=0.005)
losses = dbsde.train()

# Test: estimate u(0,0)
print(f"Estimated u(0,0): {dbsde.y0.item()}")

# Visualize training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Note: To visualize u(t,x), one needs to evaluate at multiple points, 
# but here the focus is on y0
```

## Code Explanation

1. **ZNet**: 神经网络逼近 \$Z\_t(t, x)\$（梯度），输入为时间 \$t\$ 和位置 \$x\$。

2. **DeepBSDE**: 包含初始 \$Y\_0\$（可学习参数）和 \$Z\$ 网络。模拟 Brownian 路径（前向 SDE，这里 \$\mu = 0, \sigma = 1\$），然后前向迭代 \$Y\_t\$，使用 Euler 方案。

3. **loss**: 计算终端 MSE 损失 \$\mathbb{E}\left\[ \lvert Y\_T - g(X\_T)\rvert^2 \right]\$ 的期望。

4. **train**: 每轮生成新路径（随机步长），计算损失，反向传播优化。运行后，\$Y\_0\$ 收敛到 \$u(0,0)\$。

5. **扩展**: 对于高维，只需增大 \$d\$；对于完整 \$u(t,x)\$，可固定 \$t,x\$ 并模拟从那里开始，或使用多层网络。



这个例子是最简单的演示（1D，非线性）；实际应用中可添加方差减少技术（如重要采样）或多层 Z 网络。运行代码需要 PyTorch 环境，训练后损失下降，\$Y\_0\$ 接近真实值（对于此 PDE，约 \$0.8\$ ）。




