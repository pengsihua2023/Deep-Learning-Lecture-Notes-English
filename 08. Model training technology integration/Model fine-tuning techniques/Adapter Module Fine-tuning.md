# Adapter Modules Fine-tuning

## 📖 1. Definition

**Adapter fine-tuning** is a Parameter-Efficient Fine-Tuning (PEFT) method.
Core idea:

* Freeze most of the pre-trained model parameters;
* Insert a **small bottleneck structure (Adapter)** into each Transformer layer (usually after the feed-forward layer FFN or the attention layer);
* Only train the Adapter module’s parameters, without updating the original model weights.

**Advantage**: The number of trainable parameters is greatly reduced (often <5%), while maintaining model performance.



## 📖 2. Mathematical Description

Let the Transformer hidden state vector be \$h \in \mathbb{R}^d\$.
The core structure of the Adapter is a **down-projection – nonlinearity – up-projection** bottleneck:

$$
h' = h + W_{up}\,\sigma(W_{down}\,h)
$$

Where:

* \$W\_{down} \in \mathbb{R}^{r \times d}\$: down-projection mapping, \$r \ll d\$;
* \$W\_{up} \in \mathbb{R}^{d \times r}\$: up-projection mapping;
* \$\sigma(\cdot)\$: nonlinear activation function (e.g., ReLU, GELU);
* Residual connection ensures the adapter does not break the original representation.

During training, only \${W\_{down}, W\_{up}}\$ are updated, while the original Transformer parameters remain frozen.



## 📖 3. Minimal Code Example

Write a minimal Adapter module in **PyTorch** and use it inside a Transformer layer:

```python
import torch
import torch.nn as nn

# ===== Adapter Module =====
class Adapter(nn.Module):
    def __init__(self, d_model, r=16):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(r, d_model, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # Residual connection

# ===== Insert Adapter into Transformer Layer =====
class ToyTransformerLayer(nn.Module):
    def __init__(self, d_model=128, adapter_r=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.adapter = Adapter(d_model, r=adapter_r)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.ffn(x)
        x = self.adapter(x)  # Insert Adapter
        return x

# ===== Test =====
x = torch.randn(10, 32, 128)  # [seq_len, batch, hidden_dim]
layer = ToyTransformerLayer()
out = layer(x)
print("Output shape:", out.shape)
```

Execution result:

```
Output shape: torch.Size([10, 32, 128])
```

This shows the Adapter module works properly.



## 📖 4. Summary

* **Definition**: Insert small bottleneck layers into Transformer layers and train only the Adapter.
* **Formula**: \$h' = h + W\_{up},\sigma(W\_{down},h)\$.
* **Code**: A minimal Adapter module can be implemented in just a few lines of PyTorch.

---

A complete example of Hugging Face Transformers + PEFT Adapter fine-tuning on BERT for text classification (e.g., SST-2 sentiment classification).

---

## 📖 Adapter Fine-tuning with Hugging Face PEFT

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, AdapterConfig, TaskType

# 1. Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Define Adapter configuration
adapter_config = AdapterConfig(
    task_type=TaskType.SEQ_CLS,   # Task type: sequence classification
    r=16,                         # Bottleneck dimension (down-projection)
    alpha=16,                     # Scaling factor
    dropout=0.1
)

# 3. Wrap into Adapter model
model = get_peft_model(model, adapter_config)
print(model)  # You will see adapter modules injected into the model

# 4. Load dataset (GLUE/SST-2)
dataset = load_dataset("glue", "sst2")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(2000))  # Small-sample demo
eval_dataset = encoded_dataset["validation"].shuffle(seed=42).select(range(500))

# 5. Set training arguments
training_args = TrainingArguments(
    output_dir="./adapter_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-4,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    save_strategy="no",
    fp16=True
)

# 6. Hugging Face Trainer training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# 7. Test inference
text = "The movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print(f"Input: {text} → Predicted class: {pred}")
```

## 📖 Explanation

1. **AdapterConfig**: Defines bottleneck dimension `r=16`, meaning each Adapter layer has only a small number of trainable parameters.
2. **Freeze large model**: The PEFT library automatically freezes BERT weights and only trains Adapter parameters.
3. **Dataset**: SST-2 sentiment classification (binary: positive / negative).
4. **Efficiency**: Compared with full fine-tuning, Adapter fine-tuning trains <5% parameters, making it very suitable for multi-task transfer.



## 📖 Comparison: Adapter / LoRA / Prefix Tuning

| Method            | Core Idea                                                                                                   | Update Formula                                                  | Trainable Parameters                                       | Advantage                                                                                   | Use Cases                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Adapter**       | Insert a small bottleneck network (down→activation→up) in each Transformer layer, only train Adapter params | \$h' = h + W\_{up},\sigma(W\_{down} h)\$                        | \$O(d r)\$ (where \$r \ll d\$)                             | Stable, easy multi-task transfer, each task only stores small Adapters                      | Widely used in NLP tasks, text classification, sequence labeling          |
| **LoRA**          | Freeze original weights, add low-rank updates to weight matrices                                            | \$\Delta W = B A\$, effective weight \$W^{eff} = W + \Delta W\$ | \$O(d r + r k)\$                                           | Extremely small parameter count, efficient inference, widely used with quantization (QLoRA) | LLM fine-tuning, instruction tuning, chatbots                             |
| **Prefix Tuning** | Inject trainable virtual prefix tokens into attention key/value                                             | \$K' = \[P\_k; K], ; V' = \[P\_v; V]\$                          | Proportional to prefix length, independent of model params | Fixed parameter count, decoupled from model size, good for large models                     | Generation tasks (NLG, dialogue, translation), rapid multi-task switching |


## 📖 Final Summary

* **Adapter** → Like a “plugin mini-network”, stable, suitable for classification / sequence labeling tasks.
* **LoRA** → Low-rank approximation on weight matrices, ultra-efficient, mainstream method for LLMs.
* **Prefix Tuning** → Adds virtual tokens before attention, decoupled from model size, good for generative tasks.


