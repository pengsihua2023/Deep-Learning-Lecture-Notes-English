

# LoRA Fine-tuning (Low-Rank Adaptation)

## 1. Definition

**LoRA** is a **Parameter-Efficient Fine-Tuning (PEFT)** method proposed by Microsoft in 2021.

Core idea:

* **Freeze pre-trained model parameters** \$W\$, without directly updating them.
* Introduce a **low-rank decomposition parameter** \$\Delta W = BA\$ in the update term of each large weight matrix.
* During fine-tuning, **only train the low-rank matrices \$A, B\$**, while keeping the original weights frozen.

👉 Benefits:

* Significantly reduces trainable parameter count (since rank \$r \ll d,k\$).
* After training, \$\Delta W\$ can either be “merged” into the original model, or loaded as a “plugin”.



## 2. Mathematical Description

Let the original weight matrix be:

$$
W \in \mathbb{R}^{d \times k}
$$

During fine-tuning, LoRA modifies the weight as:

$$
W' = W + \Delta W, \quad \Delta W = B A
$$

Where:

* \$A \in \mathbb{R}^{r \times k}\$ , \$B \in \mathbb{R}^{d \times r}\$ , \$r \ll \min(d,k)\$.
* Initially \$BA = 0\$, ensuring the pre-trained model is not affected.
* Only \$A, B\$ are trained, while \$W\$ is frozen.

In Transformers, LoRA is usually applied to **attention projection matrices** \$W\_Q, W\_V\$.



## 3. Simple Code Example (PyTorch)

A minimal implementation of **linear layer + LoRA** in PyTorch:

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super(LoRALayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))  
        self.weight.requires_grad = False  # Freeze original weight

        # LoRA parameters (low-rank decomposition)
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x):
        W_eff = self.weight + self.B @ self.A  # Effective weight = frozen weight + LoRA increment
        return x @ W_eff.T

# Test
x = torch.randn(2, 10)  # batch=2, input_dim=10
layer = LoRALayer(in_dim=10, out_dim=6, rank=2)
y = layer(x)
print("Output shape:", y.shape)  # (2, 6)
```

In practice, LoRA is usually inserted into Transformer **Q (query projection)** and **V (value projection)** matrices, achieving efficient adaptation with few parameters.



## 4. Summary

* **Definition**: LoRA uses low-rank decomposition, training only incremental parameter matrices while freezing original weights.
* **Formula**:

$$
W' = W + BA, \quad A \in \mathbb{R}^{r \times k}, \; B \in \mathbb{R}^{d \times r}, \; r \ll \min(d,k)
$$

* **Features**:

  * Significantly reduces trainable parameter count.
  * Can be used in “merged” or “plugin” mode after training.
  * Highly compatible with Transformer architecture.
* **Code**: PyTorch can implement this via the `LoRALayer` class.

---

# 📘 Integrating LoRA into Transformer Attention Layer

## 1. Concept

* In **Self-Attention**, there are three projection matrices:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

* **LoRA** is usually applied to \$W\_Q, W\_V\$ (can also be extended to \$W\_K, W\_O\$).
* Replacement form:

$$
W_Q' = W_Q + B_Q A_Q, \quad W_V' = W_V + B_V A_V
$$

* This way, only the **low-rank matrices \$A, B\$** are trained, while original parameters remain frozen.

---

## 2. PyTorch Example Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super(LoRALinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.weight.requires_grad = False  # Freeze original weight

        # LoRA parameters
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x):
        W_eff = self.weight + self.B @ self.A
        return x @ W_eff.T


class LoRAAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=4):
        super(LoRAAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection matrices: Q, K, V (Q and V use LoRA)
        self.W_Q = LoRALinear(embed_dim, embed_dim, rank)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = LoRALinear(embed_dim, embed_dim, rank)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)


# Test
x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
attn = LoRAAttention(embed_dim=16, num_heads=2, rank=2)
y = attn(x)
print("Output shape:", y.shape)  # (2, 5, 16)
```



## 3. Summary

* **Q and V projection matrices** are replaced with **LoRA versions**, training only low-rank matrices \$A, B\$.
* **Parameter count greatly reduced**: e.g., \$d=1024, k=1024, r=8\$, LoRA params = \$16k\$, much smaller than full \$1M+\$.
* **Strong compatibility**: LoRA can be seamlessly inserted into existing Transformer models.

---

#  Hugging Face Transformers + LoRA Example

## 1. Install Dependencies

```bash
pip install transformers datasets peft accelerate
```



## 2. Load Data & Model

We use **Hugging Face Datasets** to load the `sst2` sentiment classification task, and `bert-base-uncased`.

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Dataset
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```



## 3. Configure LoRA

The `peft` library provides a simple interface.

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
config = LoraConfig(
    r=8,                      # Low-rank dimension
    lora_alpha=16,            # Scaling factor
    target_modules=["query", "value"],  # Apply to Self-Attention Q, V matrices
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# Inject LoRA
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Example output:

```
trainable params: 590,848 || all params: 109,483,778 || trainable%: 0.54
```

👉 Only **less than 1% of parameters** are trained.



## 4. Training

We use Hugging Face `Trainer` for fine-tuning.

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-bert",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-4,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(5000)),  # Small sample demo
    eval_dataset=encoded_dataset["validation"].select(range(1000)),
    tokenizer=tokenizer
)

trainer.train()
```



## 5. Inference & Save LoRA

```python
# Inference
text = "The movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print("Prediction:", "Positive" if pred == 1 else "Negative")

# Save only LoRA parameters
model.save_pretrained("./lora-bert")
```



## 🔑 Summary

* **LoRA in Hugging Face** is very convenient — just inject with `peft.LoraConfig`.
* Only trains **Q, V matrices** low-rank updates, greatly reducing parameter count.
* Can be applied to BERT, GPT-2, T5, and other large models for **efficient fine-tuning**.


