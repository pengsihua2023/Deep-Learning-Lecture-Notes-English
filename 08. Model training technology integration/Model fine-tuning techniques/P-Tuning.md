

# P-Tuning Fine-Tuning

## 📖 1. Definition of P-Tuning

**P-Tuning** (a variant of Prompt Tuning) is a **parameter-efficient fine-tuning method** that inserts **learnable continuous prompt vectors** in front of the input sequence to guide the pre-trained language model in completing downstream tasks.

* The original model parameters remain frozen.
* Only the prompt vectors are trained (usually hundreds to thousands of parameters), which greatly reduces the cost of fine-tuning.

Unlike traditional “discrete prompts,” P-Tuning prompts are **continuous differentiable vectors** that can be directly optimized through gradient descent.

---

## 🔢 2. Mathematical Description

Assume:

* The pre-trained language model is \$f\_\theta\$, with parameters \$\theta\$ frozen.
* The original input sequence is:

$$
X = (x_1, x_2, \dots, x_n), \quad x_i \in \mathbb{R}^d
$$

where \$d\$ is the embedding dimension.

We define **learnable prompt vectors**:

$$
P = (p_1, p_2, \dots, p_m), \quad p_j \in \mathbb{R}^d
$$

The new input sequence is:

$$
X' = (p_1, p_2, \dots, p_m, x_1, x_2, \dots, x_n)
$$

During training, we only optimize \$P\$, and the objective function (e.g., cross-entropy for classification) is:

$$
\mathcal{L}(P) = - \sum_{(X, y)} \log p(y \mid X'; \theta, P)
$$

Where:

* \$\theta\$ is fixed (model parameters frozen).
* Only \$P\$ is updated through backpropagation.

---

## 💻 3. Simple Code Example

Here is a simplified **P-Tuning implementation** based on Hugging Face `transformers` (text classification as an example):

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class PTuningModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", prompt_length=10, num_labels=2):
        super().__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze pre-trained parameters
        
        # Define learnable prompt vectors
        self.prompt_length = prompt_length
        self.hidden_size = self.bert.config.hidden_size
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, self.hidden_size))
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Original input embeddings
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        # Expand batch dimension for prompt
        batch_size = input_ids.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prompt + original embeddings
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        # Extend attention mask
        prompt_mask = torch.ones(batch_size, self.prompt_length).to(attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Pass through BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Classification
        logits = self.classifier(pooled_output)
        return logits

# Test
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("P-Tuning is amazing!", return_tensors="pt", padding=True, truncation=True)
model = PTuningModel()

logits = model(**inputs)
print(logits)
```

---

## ✅ Summary

* **Definition**: Insert continuous learnable prompt vectors before the input sequence, freeze the pre-trained model, and only optimize prompt parameters.
* **Mathematical Form**: \$\mathcal{L}(P) = - \sum \log p(y \mid \[P; X]; \theta)\$.
* **Code**: Define prompt embeddings via `nn.Parameter` and concatenate them before input embeddings.



