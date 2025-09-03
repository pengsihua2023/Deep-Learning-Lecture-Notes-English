

### Introduction to the PEFT Library

**PEFT** (Parameter-Efficient Fine-Tuning) is an open-source Python library developed by Hugging Face, focusing on parameter-efficient fine-tuning methods. Its goal is to reduce the computational and storage cost of fine-tuning large pre-trained models (such as Transformer, BERT, GPT, etc.), especially in resource-constrained scenarios or when fast adaptation to multiple downstream tasks is required. PEFT provides implementations of various parameter-efficient fine-tuning techniques, including **LoRA** (Low-Rank Adaptation), **Prefix Tuning**, **Prompt Tuning**, and more.

Below is a detailed introduction to the PEFT library, covering its core features, supported fine-tuning methods, advantages, application scenarios, and a code example related to Prefix Tuning.

---

### Core Features

The main goal of the PEFT library is to allow users to fine-tune large pre-trained models at an extremely low parameter cost while maintaining performance comparable to full fine-tuning. Its core features include:

1. **Parameter-Efficient Fine-Tuning**: Only update a small number of newly added parameters (usually <1% of model parameters), freeze original model weights, and reduce memory and computation requirements.
2. **Modular Design**: Supports multiple fine-tuning methods, allowing users to choose the right strategy for their task.
3. **Seamless Integration with Hugging Face Ecosystem**: Compatible with `transformers` and `datasets` libraries, supports mainstream pre-trained models (such as BERT, RoBERTa, T5, LLaMA, etc.).
4. **Cross-Task Support**: Suitable for classification, generation, QA, sequence labeling, and other NLP tasks.
5. **Model Saving and Loading**: Efficiently stores models containing only fine-tuned parameters, significantly reducing storage requirements.
6. **Ease of Use**: Provides simple APIs to implement complex fine-tuning logic with minimal code.

---

### Supported Fine-Tuning Methods

PEFT library supports the following main parameter-efficient fine-tuning methods:

1. **LoRA (Low-Rank Adaptation)**:

   * Fine-tunes models by introducing low-rank decomposition (Low-Rank Updates) in weight matrices.
   * Only a small number of low-rank matrix parameters are updated, suitable for various tasks.
   * Advantages: efficient, performance close to full fine-tuning.
2. **Prefix Tuning**:

   * Adds learnable prefix vectors in the attention layers of Transformers, affecting key (K) and value (V) computations.
   * Suitable for generation tasks and sequence classification tasks (the main focus previously discussed).
3. **Prompt Tuning**:

   * Adds learnable virtual tokens (prompts) at the input layer, optimizing only these prompt parameters.
   * Suitable for small-scale tasks or data-limited scenarios.
4. **P-Tuning v2**:

   * An improved version of Prompt Tuning, adds flexibility to prefix parameters, suitable for more complex tasks.
5. **Adapter Methods**:

   * Insert small fully-connected modules (Adapters) into each Transformer layer, optimizing only these modules.
   * Advantages: modular, easy to switch Adapters for different tasks.
6. **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**:

   * Fine-tunes by scaling Transformer internal activations, extremely parameter-efficient.
7. **LoHA (Low-Rank Hadamard Adaptation)** and **LoKr (Low-Rank Kronecker Adaptation)**:

   * Variants of LoRA that use different matrix decompositions for further efficiency.

---

### Advantages

* **Low Resource Requirements**: Only fine-tune a small number of parameters (usually tens of KB to a few MB), suitable for running on regular GPUs or CPUs.
* **Fast Adaptation**: Different tasks can use independent trainable parameters, no need to retrain the whole model when switching tasks.
* **Storage Efficient**: Saved models only contain fine-tuned parameters, drastically reducing storage space (compared to GB-sized full fine-tuned models).
* **Comparable Performance**: On many tasks (such as sentiment analysis, text generation), performance is close to or even better than full fine-tuning.
* **Open Source and Community Support**: PEFT is open-source (Apache 2.0 license), deeply integrated with the Hugging Face ecosystem, and has an active community.

---

### Application Scenarios

* **NLP Tasks**: Sentiment analysis, text classification, machine translation, question answering, text generation, etc.
* **Multi-Task Learning**: Train independent fine-tuning parameters (LoRA or Adapter) for each task to achieve efficient task switching.
* **Edge Deployment**: Fine-tune and deploy large models on resource-limited devices.
* **Domain Adaptation**: Quickly adapt general pre-trained models to specific domains (e.g., healthcare, legal).
* **Research and Experimentation**: Quickly test different fine-tuning strategies and explore the effects of parameter-efficient methods.

---

### Relationship with Prefix Tuning

In your earlier discussion, you focused on **Prefix Tuning**, which is one of the methods supported by PEFT. PEFT simplifies the implementation of Prefix Tuning through the `PrefixTuningConfig` and `get_peft_model` functions, automatically integrating learnable prefixes into the Transformer attention mechanism. Below is a real example based on the IMDB dataset, showing how to implement Prefix Tuning using PEFT for text classification.

---

### Code Example: Prefix Tuning with PEFT (IMDB Dataset)

The following code, based on the previously discussed IMDB dataset, demonstrates how to use the PEFT library to perform Prefix Tuning for sentiment classification.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from peft import PrefixTuningConfig, get_peft_model
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 1. Load IMDB dataset
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# For faster demonstration, use only 2000 training samples and 1000 test samples
train_texts, train_labels = train_texts[:2000], train_labels[:2000]
test_texts, test_labels = test_texts[:1000], test_labels[:1000]

# 2. Custom Dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 3. Load Tokenizer and DataLoader
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4. Configure Prefix Tuning
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
prefix_config = PrefixTuningConfig(
    task_type="SEQ_CLS",  # Sequence classification task
    num_virtual_tokens=20,  # Prefix length
    prefix_projection=True  # Enable projection layer
)
model = get_peft_model(model, prefix_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}% of total)")
print(f"Total parameters: {total_params}")

# 5. Optimizer and Scheduler
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
num_epochs = 3
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 6. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 7. Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 8. Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=["Negative", "Positive"])
    avg_loss = total_loss / len(dataloader)
    return accuracy, report, avg_loss

# 9. Training loop
for epoch in range(num_epochs):
    avg_train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
    test_accuracy, test_report, test_loss = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    print("Classification Report:\n", test_report)

# 10. Save model
model.save_pretrained("prefix_tuned_bert_imdb_peft")
tokenizer.save_pretrained("prefix_tuned_bert_imdb_peft")
print("Model and tokenizer saved!")

# 11. Inference example
loaded_model = BertForSequenceClassification.from_pretrained("prefix_tuned_bert_imdb_peft")
loaded_tokenizer = BertTokenizer.from_pretrained("prefix_tuned_bert_imdb_peft")
loaded_model.to(device)
loaded_model.eval()

test_text = "This movie was absolutely fantastic and thrilling!"
encoding = loaded_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
encoding = {k: v.to(device) for k, v in encoding.items()}

with torch.no_grad():
    outputs = loaded_model(**encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "Negative", 1: "Positive"}
    print(f"Predicted sentiment: {label_map[predicted_class]}")
```

---

### Code Explanation

1. **Dataset**:

   * Uses Hugging Face `datasets` library to load the IMDB dataset.
   * For faster demonstration, training data is limited to 2000 samples and test data to 1000 samples (remove the limit to use all 50,000 samples).
2. **PEFT Configuration**:

   * Configures Prefix Tuning with `PrefixTuningConfig`, setting `num_virtual_tokens=20` and `prefix_projection=True`.
   * Uses `get_peft_model` to automatically integrate prefix parameters into the BERT model.
3. **Training & Evaluation**:

   * Trains for 3 epochs using the AdamW optimizer and linear learning rate scheduler.
   * Evaluation includes accuracy, classification report, and test loss.
4. **Inference**:

   * Shows how to load the saved model and perform sentiment prediction.
5. **Dependencies**:

   ```bash
   pip install torch transformers peft datasets scikit-learn tqdm
   ```

---

### Installing and Using the PEFT Library

* **Installation**:

  ```bash
  pip install peft
  ```
* **Documentation**: The official PEFT documentation ([https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)) provides detailed configuration instructions and examples.
* **Supported Models**: Supports nearly all Transformer models in Hugging Face `transformers`.
* **Version Requirements**: Recommended to use the latest versions (as of August 2025, PEFT>=0.5.0, transformers>=4.30.0).

---

### Extensions and Further Needs

If you need any of the following, let me know, and I can further customize:

1. **Other PEFT Methods**: e.g., LoRA or Prompt Tuning implementations.
2. **Other Datasets**: e.g., Yelp, SST-2, Twitter.
3. **Multi-Task Support**: Configure independent PEFT parameters for multiple tasks.
4. **Snakemake Integration**: Incorporate PEFT training workflows into Snakemake (related to your earlier discussion).
5. **Mathematical Derivation**: Provide deeper explanations of Prefix Tuning or LoRA math.
6. **Deployment**: Deploy the fine-tuned model to GitHub or cloud platforms.



