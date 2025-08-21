# Long Short-Term Memory (LSTM) Study Notes

## 1. Introduction

**Long Short-Term Memory (LSTM)** is a special type of Recurrent Neural Network (RNN) designed to solve the **vanishing gradient problem** of standard RNNs.

* **Strengths:** Capable of learning and remembering long-range dependencies in sequential data.
* **Applications:** Widely used in **speech recognition**, **time series forecasting**, **text generation**, **machine translation**, and more.

---

## 2. Core Concept

LSTM introduces a **gating mechanism** that regulates the flow of information:

* **Forget Gate:** Decides how much past information should be discarded.
* **Input Gate:** Controls how much new information is added to the cell state.
* **Output Gate:** Determines the output of the current hidden state.

Through these gates, LSTMs effectively retain useful information and mitigate gradient vanishing.

---

## 3. LSTM Cell Structure

At each time step $t$, an LSTM cell receives:

* the previous hidden state $h_{t-1}$,
* the previous cell state $c_{t-1}$, and
* the current input $x_t$.

The structure consists of:

* Input Gate
* Forget Gate
* Output Gate
* Cell State

---

## 4. Mathematical Formulation

### (1) Forget Gate

$$
f_t = \sigma \big( W_f [h_{t-1}, x_t] + b_f \big)
$$

*Controls what fraction of the previous cell state is retained.*

### (2) Input Gate

$$
i_t = \sigma \big( W_i [h_{t-1}, x_t] + b_i \big)
$$

$$
c_t = \tanh \big( W_c [h_{t-1}, x_t] + b_c \big)  
$$

*Determines what new information is stored in the cell state.*

### (3) Cell State Update

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

*Updates the cell state by combining retained and new information.*

### (4) Output Gate

$$
o_t = \sigma \big( W_o [h_{t-1}, x_t] + b_o \big)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

*Controls the output of the hidden state.*

---

## 5. Parameters

* $W_f, W_i, W_c, W_o$: Weight matrices
* $b_f, b_i, b_c, b_o$: Bias terms
* $h_t$: Hidden state
* $c_t$: Cell state

---

## 6. PyTorch Implementation

```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_new = nn.Linear(hidden_size, 20)
        self.fc = nn.Linear(20, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_new(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
```

*This model outputs a binary classification probability for sequential input data.*

---

## 7. Training & Results

* **Loss function:** Binary Cross-Entropy (BCELoss)
* **Optimizer:** Adam
* **Dataset:** UCI HAR (Human Activity Recognition), binary classification (Walking vs. Non-Walking)

**Training Results:**

* Loss converges rapidly
* Final test accuracy: **98.71%**

**Loss Curve:**
The training loss steadily decreases, confirming model convergence.

---

## 8. Key Takeaways

* LSTM is a powerful extension of RNN for handling **long-term dependencies**.
* The **gating mechanism** (forget, input, output) is crucial for controlling information flow.
* Practical implementations (e.g., PyTorch) are straightforward and widely used in real-world sequence modeling tasks.

---

✨ With these notes, you now have both the **mathematical foundation** and a **hands-on implementation** for LSTM.

---

If you want, I can now take this and turn it into a **fully compilable LaTeX document** with proper sections, theorem-style boxes for formulas, and nicely formatted Python code.

Do you want me to do that next?




