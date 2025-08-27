
# AR-SPINN Mathematical Description

AR-SPINN combines the dynamics of **Spiking Neurons**, the temporal dependence of **Recurrent Structures (RNN)**, and the **Adaptive Threshold Mechanism**.
It can be described by the following equations:

---

### 1. Membrane Potential Dynamics

The membrane potential of neuron \$i\$ at time \$t\$ is updated as:

$$
u_i(t) = \alpha u_i(t-1) + \sum_{j} W_{ij} s_j(t) + \sum_{k} R_{ik} s_k(t-1) - v \cdot s_i(t-1)
$$

where:

* \$\alpha \in (0,1)\$: leakage decay coefficient
* \$W\_{ij}\$: input weight
* \$R\_{ik}\$: recurrent connection weight
* \$s\_j(t) \in {0,1}\$: input spike
* \$s\_i(t-1)\$: self spike at the previous time step
* \$v\$: reset value

---

### 2. Spiking Function

$$
s_i(t) = H\big(u_i(t) - \theta_i(t)\big)
$$

where \$H(\cdot)\$ is the Heaviside step function, and \$\theta\_i(t)\$ is the dynamic threshold.

---

### 3. Adaptive Threshold Update

The threshold dynamically changes with spiking history:

$$
\theta_i(t) = \theta_0 + \beta \sum_{k < t} \gamma^{t-k} s_i(k)
$$

where:

* \$\theta\_0\$: baseline threshold
* \$\beta > 0\$: adaptation strength
* \$\gamma \in (0,1)\$: forgetting factor

---

# Minimal Python Implementation

Below is a minimal **AR-SPINN single-neuron implementation**, to illustrate the core idea:

```python
import numpy as np

class ARSPINN:
    def __init__(self, n_inputs, alpha=0.9, v=1.0,
                 theta0=1.0, beta=0.1, gamma=0.95):
        self.alpha = alpha      # decay coefficient
        self.v = v              # reset amplitude
        self.theta0 = theta0    # initial threshold
        self.beta = beta        # threshold adaptation strength
        self.gamma = gamma      # threshold forgetting factor
        self.u = 0.0            # membrane potential
        self.theta = theta0     # current threshold
        self.w_in = np.random.randn(n_inputs)  # input weights
        self.w_rec = np.random.randn()         # self-recurrent weight

    def step(self, x_t, s_prev):
        # Membrane potential update: input + recurrent + decay
        self.u = self.alpha * self.u \
                 + np.dot(self.w_in, x_t) \
                 + self.w_rec * s_prev
        
        # Spike or not
        s = 1 if self.u >= self.theta else 0
        
        # Reset after spike
        if s:
            self.u -= self.v
        
        # Adaptive threshold update
        self.theta = self.theta0 + self.beta * (self.gamma * (self.theta - self.theta0) + s)
        
        return s

# ====== Test ======
T = 20
x_seq = np.random.randint(0, 2, (T, 3))  # random input (20 steps, 3 dimensions)
neuron = ARSPINN(n_inputs=3)

s_prev = 0
output = []
for t in range(T):
    s_prev = neuron.step(x_seq[t], s_prev)
    output.append(s_prev)

print("Input sequence:\n", x_seq)
print("Output spikes:\n", output)
```

---

### This implementation demonstrates the three core mechanisms of **AR-SPINN**:

1. **Temporal Recurrence**: historical dependence introduced via `self.w_rec * s_prev`;
2. **Spiking Neuron Dynamics**: membrane potential accumulation, decay, reset;
3. **Adaptive Threshold**: threshold dynamically changes with spiking history.

---

