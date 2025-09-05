# ReLoRA (Restarted Low-Rank Adaptation) Fine-Tuning Method

## 📖 1. Definition

**ReLoRA** is an improved fine-tuning method based on **LoRA (Low-Rank Adaptation)**. The core idea is:

* **LoRA**: when fine-tuning large pretrained models (LLMs), instead of updating the full parameter matrix \$W \in \mathbb{R}^{d \times k}\$, LoRA inserts a low-rank decomposition \$W + BA\$, where \$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)\$. This significantly reduces parameter count and memory usage.
* **ReLoRA**: during training, **periodically merge the LoRA low-rank increments into the main weights, then reset the LoRA parameters**.

  * This keeps the efficiency of low-rank updates while avoiding underfitting caused by relying on a low-rank approximation for too long;
  * At the same time, repeated "restarts" allow accumulating more information, improving convergence speed and final performance.

In other words, ReLoRA **periodically absorbs what LoRA has learned into the model weights, then gives LoRA a fresh space to continue learning**.



## 📖 2. Mathematical Formulation

Let:

* Original weight matrix: \$W \in \mathbb{R}^{d \times k}\$
* LoRA parameters: \$A\_t \in \mathbb{R}^{r \times k}, B\_t \in \mathbb{R}^{d \times r}\$
* Effective weights:

$$
W_t^{\text{eff}} = W + B_t A_t
$$

### ReLoRA update steps:

1. **Regular LoRA update** (within one cycle):

$$
(A_t, B_t) \leftarrow (A_{t-1}, B_{t-1}) - \eta \nabla_{A,B} L(W_{t-1}^{\text{eff}})
$$

2. **Periodic merge** (every \$T\$ steps):

$$
W \leftarrow W + B_t A_t
$$

$$
A_t, B_t \leftarrow \text{init}() \quad (\text{re-initialize randomly})
$$

Thus, model weights \$W\$ continuously absorb LoRA’s low-rank improvements, while LoRA parameters are reset to avoid early training limitations.

---

## 📖 3. Minimal Code Example

Here’s a minimal **PyTorch ReLoRA fine-tuning demo** (illustrating the mechanism, not a full library implementation):

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ===== Simple LoRA module =====
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))  # original weight
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)   # LoRA A
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)  # LoRA B
        self.rank = rank

    def forward(self, x):
        return nn.functional.linear(x, self.W + self.B @ self.A)

    def merge_lora(self):
        """Merge LoRA into the main weight"""
        with torch.no_grad():
            self.W += self.B @ self.A
            nn.init.normal_(self.A, std=0.01)
            nn.init.normal_(self.B, std=0.01)

# ===== Data and model =====
x = torch.randn(100, 10)
y = torch.randn(100, 5)

model = LoRALinear(10, 5, rank=4)
criterion = nn.MSELoss()
optimizer = optim.Adam([model.A, model.B], lr=1e-2)  # only train LoRA parameters

# ===== ReLoRA training =====
steps = 200
merge_every = 50  # merge every 50 steps

for step in range(steps):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (step + 1) % merge_every == 0:
        print(f"Step {step+1}: Loss = {loss.item():.4f}, merging LoRA...")
        model.merge_lora()
```



### 📖 Explanation

1. **LoRALinear**: implements a linear layer with LoRA.
2. **merge\_lora()**: merges \$BA\$ into the main weight \$W\$, then re-initializes \$A, B\$.
3. **Training loop**: calls `merge_lora()` every `merge_every` steps to realize ReLoRA’s periodic restart.
4. **Effect**: compared to plain LoRA, ReLoRA converges more stably.

---

## ReLoRA vs LoRA Convergence Comparison Example

We use a **simple regression task** to compare their loss curves under the same conditions.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===== LoRA module =====
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        return nn.functional.linear(x, self.W + self.B @ self.A)

    def merge_lora(self):
        """Merge LoRA into main weight and reset A, B"""
        with torch.no_grad():
            self.W += self.B @ self.A
            nn.init.normal_(self.A, std=0.01)
            nn.init.normal_(self.B, std=0.01)

# ===== Data =====
torch.manual_seed(42)
x = torch.randn(200, 10)
true_w = torch.randn(5, 10)
y = x @ true_w.T + torch.randn(200, 5) * 0.1  # linear task + small noise

# ===== Experiment setup =====
steps = 300
merge_every = 50

def train(model, relora=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([model.A, model.B], lr=1e-2)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if relora and (step + 1) % merge_every == 0:
            model.merge_lora()

        losses.append(loss.item())
    return losses

# ===== Train LoRA and ReLoRA =====
model_lora = LoRALinear(10, 5, rank=4)
losses_lora = train(model_lora, relora=False)

model_relora = LoRALinear(10, 5, rank=4)
losses_relora = train(model_relora, relora=True)

# ===== Plot =====
plt.plot(losses_lora, label="LoRA")
plt.plot(losses_relora, label="ReLoRA")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("LoRA vs ReLoRA Convergence Comparison")
plt.legend()
plt.show()
```



## 📖 Code Explanation

1. **Data**: creates a linear regression task \$y = Wx + \epsilon\$.
2. **LoRALinear**: same as before, implements LoRA weights.
3. **LoRA training**: only updates low-rank matrices \$A, B\$.
4. **ReLoRA training**: merges LoRA into \$W\$ every `merge_every=50` steps.
5. **Result**: plots convergence curves for LoRA and ReLoRA.

In practice you’ll see:

* **LoRA** curve decreases but may converge slowly or plateau;
* **ReLoRA** curve decreases more steadily and often reaches a lower loss.



