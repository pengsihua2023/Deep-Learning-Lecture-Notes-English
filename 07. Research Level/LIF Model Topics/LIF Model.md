
# **Leaky Integrate-and-Fire (LIF) Neuron Model**

## 📖 1. Definition

**LIF neuron** is one of the most common and simplified models in Spiking Neural Networks (SNNs).

* It abstracts the neuron as a **circuit with a capacitor and resistor**.
* The membrane potential \$V(t)\$ gradually accumulates with input current (integrate).
* The membrane potential simultaneously decays back to the resting potential over time (leaky).
* When \$V(t)\$ exceeds the threshold \$V\_{th}\$, the neuron fires a spike, and then the membrane potential is reset.
<div align="center">
<img width="4250" height="154" alt="image" src="https://github.com/user-attachments/assets/0f6b947c-44bf-4aa6-b402-0e1f017e1db4" />
</div>

## 📖 2. Mathematical Description

The dynamics of the LIF model is a first-order differential equation:

$$
\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{rest}) + R \cdot I(t)
$$

where:

* \$V(t)\$: membrane potential
* \$V\_{rest}\$: resting potential
* \$R\$: membrane resistance
* \$I(t)\$: input current
* \$\tau\_m = RC\$: membrane time constant (decay rate)

**Firing rule**:

$$
\text{if } V(t) \geq V_{th} \quad \Rightarrow \quad \text{emit a spike, and } V(t) \to V_{reset}
$$



## 📖 3. Minimal Code Example (Python + NumPy)

Below is the simplest simulation of a single LIF neuron, using Euler's method for numerical integration:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 100.0   # total time (ms)
dt = 1.0    # time step (ms)
steps = int(T/dt)

# LIF parameters
tau_m   = 10.0     # membrane time constant
V_rest  = -65.0    # resting potential
V_reset = -65.0    # reset potential
V_th    = -50.0    # threshold
R       = 1.0      # membrane resistance
I       = 1.5      # constant input current

# State variables
V = np.zeros(steps)
V[0] = V_rest
spikes = []

# Simulation
for t in range(1, steps):
    dV = (-(V[t-1] - V_rest) + R*I) / tau_m * dt
    V[t] = V[t-1] + dV
    if V[t] >= V_th:        # trigger spike
        V[t] = V_reset
        spikes.append(t*dt)

# Plot
time = np.arange(0, T, dt)
plt.plot(time, V, label="Membrane potential")
plt.axhline(V_th, color='r', linestyle='--', label="Threshold")
plt.xlabel("Time (ms)")
plt.ylabel("Potential (mV)")
plt.title("LIF Neuron Simulation")
plt.legend()
plt.show()

print("Spike times (ms):", spikes)
```



### Execution Result

* In the figure, you can see the membrane potential gradually rises to the threshold → triggers a spike → resets → accumulates again → forms periodic firing.
* The input current `I` controls whether spikes occur and the firing frequency.

---


