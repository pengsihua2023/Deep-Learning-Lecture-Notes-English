# P-Tuning v2 Fine-Tuning

## 📖 1. Definition of P-Tuning v2

**P-Tuning v2** is a **Parameter-Efficient Fine-Tuning (PEFT)** method proposed by Tsinghua University, an improved version of P-Tuning.

* **Core idea**: Insert a set of **learnable continuous prefix vectors** into the **input of each Transformer layer**.
* **Different from Prompt Tuning**: Prompt Tuning only adds soft prompts at the input layer, while P-Tuning v2 injects **prefixes at multiple layers (prefix-tuning-like)**, enhancing expressive power.
* **Features**:

  * Achieves performance close to full fine-tuning.
  * Does not rely on complex LSTM/MHA structures (simpler than P-Tuning v1).
  * Applicable to various PLMs (GPT, BERT, T5).



## 📖 2. Mathematical Description

Let:

* The pre-trained Transformer model be \$f\_\theta\$, with parameters \$\theta\$ frozen.
* Input sequence embeddings:

$$
X = (x_1, x_2, \dots, x_n), \quad x_i \in \mathbb{R}^d
$$

At each layer \$l \in {1, \dots, L}\$, introduce **prefix vectors**:

$$
P^l = (p^l_1, p^l_2, \dots, p^l_m), \quad p^l_j \in \mathbb{R}^d
$$

In the attention calculation, prefix vectors are concatenated to the query/key/value inputs:

$$
\text{SelfAttn}(X^l) = \text{Softmax}\left( \frac{[X^l W_Q; P^l_Q][X^l W_K; P^l_K]^T}{\sqrt{d_k}} \right) [X^l W_V; P^l_V]
$$

Where:

* \$W\_Q, W\_K, W\_V\$ are frozen model weights.
* \$P^l\_Q, P^l\_K, P^l\_V\$ are results of projecting prefix vectors.

Training objective:

$$
\mathcal{L}(\{P^l\}) = - \sum_{(X, y)} \log p(y \mid X, \theta, \{P^l\})
$$

That is: freeze \$\theta\$, only train prefix parameters \${P^l}\$.



## 📖 3. Simple Code Demonstration

Below is a **simplified Hugging Face implementation of P-Tuning v2** (showing only core logic):

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class PrefixEncoder(nn.Module):
    """Use MLP to generate prefix embeddings"""
    def __init__(self, prefix_length, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(prefix_length, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, batch_size):
        prefix_tokens = torch.arange(self.embedding.num_embeddings).to(self.embedding.weight.device)
        prefix_embeds = self.embedding(prefix_tokens)  # [m, d]
        prefix_embeds = self.mlp(prefix_embeds)
        return prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, m, d]

class PTuningV2Model(nn.Module):
    def __init__(self, model_name="bert-base-uncased", prefix_length=10, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze pre-trained parameters

        self.prefix_encoder = PrefixEncoder(prefix_length, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.prefix_length = prefix_length

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)

        # Original word embeddings
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        # Prefix embeddings
        prefix_embeds = self.prefix_encoder(batch_size)

        # Concatenate prefix + original input
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # Extend attention mask
        prefix_mask = torch.ones(batch_size, self.prefix_length).to(attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Input to BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]

        # Classification
        logits = self.classifier(cls_output)
        return logits

# Example
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("P-Tuning v2 is powerful!", return_tensors="pt")

model = PTuningV2Model()
logits = model(**inputs)
print(logits)
```



## 📖 Summary

* **Definition**: P-Tuning v2 injects prefix vectors into **each Transformer layer**, training only prefix parameters while freezing the model.
* **Mathematical Form**: Add prefix Q/K/V in each self-attention layer, optimize \${P^l}\$.
* **Code Implementation**: Use a `PrefixEncoder` to generate prefix embeddings, concatenate them to input embeddings and attention computation.

