# QLoRA (Quantized Low-Rank Adaptation) Fine-Tuning Method

## 📖 1. Definition

**QLoRA (Quantized Low-Rank Adaptation)** is a method proposed in 2023 for efficiently fine-tuning large language models (LLMs). It combines **weight quantization** and **LoRA (Low-Rank Adaptation)**:

* **Weight quantization**: compress pretrained model weights from 16/32-bit floating point to 4-bit representation (usually NF4 format), significantly reducing memory usage.
* **LoRA**: add low-rank matrices (usually rank = 4–64) on top of quantized weights, training only these small matrices.
* **Result**:

  * Enables fine-tuning of models with tens of billions of parameters on a single GPU;
  * With almost no performance loss;
  * Efficient and memory-friendly.



## 📖 2. Mathematical Formulation

Let:

* Original weight matrix: \$W \in \mathbb{R}^{d \times k}\$
* Quantized weights: \$\hat{W} = Q(W)\$, where \$Q(\cdot)\$ is the quantization function (e.g., 4-bit NF4)
* LoRA parameters: \$A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}, r \ll \min(d,k)\$

**QLoRA effective weights representation**:

$$
W^{\text{eff}} = \hat{W} + BA
$$

Where:

* \$\hat{W}\$ is the frozen quantized weight;
* \$BA\$ is the trainable low-rank increment.

During training, only \$A,B\$ are updated while \$\hat{W}\$ remains frozen, thus reducing memory and compute overhead.

---

## 📖 3. Minimal Code Example

A minimal example of QLoRA fine-tuning using the Hugging Face **PEFT** library (assuming a small Transformers model):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load a pretrained model with 4-bit quantization
model_name = "facebook/opt-125m"  # small model example
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,                # enable 4-bit quantization
    device_map="auto",                # automatically allocate to GPU
    torch_dtype=torch.float16
)

# 2. Enable k-bit training support (freeze quantized weights)
model = prepare_model_for_kbit_training(model)

# 3. Define LoRA configuration
lora_config = LoraConfig(
    r=8,                       # LoRA rank
    lora_alpha=16,             # scaling factor
    target_modules=["q_proj","v_proj"],  # insert LoRA in these layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Add LoRA layers to the model (becomes QLoRA)
model = get_peft_model(model, lora_config)

# 5. Example input
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("QLoRA is great because", return_tensors="pt").to("cuda")

# 6. Forward pass
outputs = model(**inputs)
print("Logits shape:", outputs.logits.shape)
```


## 📖 Explanation

1. **`load_in_4bit=True`** → model weights are loaded in 4-bit, saving memory significantly.
2. **`prepare_model_for_kbit_training`** → freezes quantized weights and enables gradient support.
3. **`LoraConfig`** → defines LoRA parameters, e.g., rank `r=8`, applied to Transformer layers `q_proj, v_proj`.
4. **`get_peft_model`** → injects LoRA into the model, creating QLoRA.
5. **During training** → only LoRA parameters (`A, B`) are updated, while quantized weights remain frozen.

Below is a **full QLoRA fine-tuning training loop** example using Hugging Face 🤗 PEFT + Transformers, running on a small dataset (like `tiny_shakespeare` or a custom snippet of text) to demonstrate **loss decreasing**.

---

## Full QLoRA Fine-Tuning Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load model and tokenizer (small model example)
model_name = "facebook/opt-125m"  # can also be LLaMA/OPT/GPT-NeoX etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,           # enable 4-bit quantization
    device_map="auto",           # auto-assign to GPU
    torch_dtype=torch.float16
)

# 2. Enable k-bit training
model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA (QLoRA = 4-bit quantization + LoRA)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # common LoRA insertion points in attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Build a small dataset (tiny_shakespeare.txt or custom text)
with open("tiny_shakespeare.txt", "w") as f:
    f.write("To be, or not to be, that is the question.\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune...")

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="tiny_shakespeare.txt",
    block_size=64
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 5. Set training arguments
training_args = TrainingArguments(
    output_dir="./qlora_out",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=1,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs"
)

# 6. Train with Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

# 7. Test generation
inputs = tokenizer("To be, or not to", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```



## 📖 Code Explanation

1. **Quantized loading**: `load_in_4bit=True` → uses 4-bit NF4 weights.
2. **LoRA configuration**: inserts low-rank adapters only in attention layers `q_proj, v_proj`.
3. **Dataset**: uses a simplified `tiny_shakespeare` text file.
4. **Training loop**: powered by Hugging Face `Trainer`, loss decreases over iterations.
5. **Inference**: after training, the model can generate Shakespeare-style text.



