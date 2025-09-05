# Neural Operators
## 📖 Introduction
Neural Operators are a special type of neural network architecture designed to learn mappings between function spaces (i.e., to learn “operators”). Traditional neural networks usually handle point-to-point mappings (e.g., pixel-to-label in image classification), while neural operators can handle function-to-function mappings, such as rapidly approximating solutions to partial differential equations (PDEs) in scientific computing. They are resolution-invariant, meaning that data at one resolution can be used for training, while inference can be performed at different resolutions.


## 📖 Mathematical Description

A neural operator aims to approximate an operator \$G : \mathcal{A} \to \mathcal{U}\$, where \$\mathcal{A}\$ and \$\mathcal{U}\$ are Banach spaces (typically function spaces such as \$L^2(D)\$), and \$D \subset \mathbb{R}^d\$ is the domain. Given an input function \$a \in \mathcal{A}\$, the goal is to predict the output function \$u = G(a) \in \mathcal{U}\$.

The neural operator \$G^\theta\$ (with parameters \$\theta\$) is designed to be resolution-invariant, i.e., independent of the discretization resolution of the input function. Formally, it can be expressed in a hierarchical structure:

$$
v_0(x) = P(a(x)), \quad v_{t+1}(x) = \sigma \big( W v_t(x) + (\mathcal{K}(a;\phi) v_t)(x) \big), \quad t = 0, \ldots, T-1,
$$

$$
G^\theta(a)(x) = Q(v_T(x)),
$$

where:

* \$P : \mathbb{R}^{d\_a} \to \mathbb{R}^{d\_v}\$ is a lifting map, increasing the input dimension from \$d\_a\$ to a higher dimension \$d\_v\$.
* \$Q : \mathbb{R}^{d\_v} \to \mathbb{R}^{d\_u}\$ is a projection map, projecting the hidden dimension into the output dimension \$d\_u\$.
* \$W\$ is a local linear transformation (pointwise).
* \$\mathcal{K}\$ is a non-local integral kernel operator:

$$
(\mathcal{K}(a;\phi)v)(x) = \int_D \kappa_\phi(x,y,a(x),a(y)) v(y)\, dy,
$$

where \$\kappa\_\phi\$ is a kernel function parameterized by a neural network.

* \$\sigma\$ is an activation function (e.g., GELU).

---

A classic implementation is the **Fourier Neural Operator (FNO)**, which leverages Fourier transforms to efficiently parameterize \$\mathcal{K}\$ in the frequency domain, avoiding direct integration. Assuming periodic boundary conditions, in the 1D case:

$$
(\mathcal{K}v)(x) = \mathcal{F}^{-1} \Big( R_\phi \cdot (\mathcal{F}v) \Big)(x),
$$

where:

* \$\mathcal{F}\$ is the Fourier transform: \$(\mathcal{F}v)\_k = \int\_D v(x)e^{-2\pi i k \cdot x} dx \quad (\text{discretely via FFT})\$.
* \$\mathcal{F}^{-1}\$ is the inverse Fourier transform.
* \$R\_\phi\$ is a learnable (complex) parameter matrix, truncated to the first \$k\_{\max}\$ low-frequency modes: for each mode \$k\$, \$R\_\phi(k) \in \mathbb{C}^{d\_v \times d\_v}\$. This makes \$\mathcal{K}\$ a global convolution, with computational complexity \$O(N \log N)\$, where \$N\$ is the number of grid points.

The strength of FNO lies in resolution-invariance: it can be trained on coarse grids and used on fine grids during inference, since Fourier modes are continuous.

---
## 📖 Code
Below is the simplest 1D FNO example implemented from scratch in PyTorch. The task is to learn a simple operator: mapping the input function \$f(x) = \sin(kx)\$ to its integral form (cumulative integral). The code includes the spectral convolution layer (`SpectralConv1d`) and the FNO model, with synthetic training data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Spectral convolution layer (core component, uses FFT in frequency domain)
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x shape: (batch, in_channels, grid_size)
        x_ft = torch.fft.rfft(x)  # real FFT
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # inverse FFT
        return x

# Simple 1D FNO model
class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64):
        super(FNO1d, self).__init__()
        self.conv0 = SpectralConv1d(1, width, modes)  # input channel = 1
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(1, width, 1)  # linear layers
        self.w1 = nn.Conv1d(width, width, 1)
        self.fc = nn.Linear(width, 1)  # output layer

    def forward(self, x):
        # x shape: (batch, grid_size, 1) -> (batch, 1, grid_size)
        x = x.permute(0, 2, 1)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = x.permute(0, 2, 1)  # back to (batch, grid_size, width)
        x = self.fc(x)
        return x.squeeze(-1)  # output (batch, grid_size)

# Generate synthetic data: f(x) = sin(kx), target g(x) = -cos(kx)/k
grid_size = 256
n_train = 1000
x = torch.linspace(0, 2 * np.pi, grid_size).unsqueeze(0).repeat(n_train, 1)  # (n_train, grid_size)
k = torch.randint(1, 5, (n_train, 1))  # random frequencies
input = torch.sin(k * x)  # (n_train, grid_size)
target = -torch.cos(k * x) / k  # integral (n_train, grid_size)
input = input.unsqueeze(-1)  # (n_train, grid_size, 1)
target = target.unsqueeze(-1)

# Train model
model = FNO1d()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):  # train 100 epochs
    optimizer.zero_grad()
    out = model(input)
    loss = F.mse_loss(out.unsqueeze(-1), target)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test
test_input = torch.sin(3 * x[0]).unsqueeze(0).unsqueeze(-1)  # test sample k=3
pred = model(test_input).detach().numpy()
true = -np.cos(3 * x[0].numpy()) / 3
plt.plot(x[0].numpy(), pred[0], label='Prediction')
plt.plot(x[0].numpy(), true, label='True')
plt.legend()
plt.show()
```



## 📖 Code Explanation

1. **SpectralConv1d**: Core of FNO. Uses Fourier transform (FFT) to move input into frequency domain, applies learned weights on low-frequency modes, then inverse transform back. Captures global dependencies efficiently.
2. **FNO1d**: Stacks spectral convolution layers and linear layers with GELU activation. Input is a discretized function (values on grid points), output is also a function.
3. **Data Generation**: Input sine functions and their integrals as targets, simulating function-to-function mapping.
4. **Training**: Uses MSE loss to minimize the difference between prediction and true integral. After training, predictions align closely with the true function.

This example is a minimal demonstration. In practice, it can be extended to PDE solving (e.g., Burgers’ equation). Requires a PyTorch environment; after training, the loss decreases and predictions approach the true solution.



