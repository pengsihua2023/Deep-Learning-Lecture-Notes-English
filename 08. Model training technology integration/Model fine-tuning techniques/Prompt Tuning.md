

# Prompt Tuning Fine-Tuning

## 📖 1. Definition of Prompt Tuning

**Prompt Tuning** is a **Parameter-Efficient Fine-Tuning (PEFT)** method.

* It adapts to downstream tasks by inserting a **trainable continuous prompt vector** (soft prompt) before the input sequence.
* Different from **P-Tuning**:

  * Prompt Tuning only adds prompts at the **embedding layer**, not inside the intermediate layers of the model.
  * P-Tuning v1 used LSTM/MHA to model prompts, which was more complex.
* During training, **only the soft prompt parameters are updated**, while the **entire pre-trained language model is frozen**.



## 📖 2. Mathematical Formulation

Let:

* The pre-trained language model be \$f\_\theta\$, with parameters \$\theta\$ frozen.
* The original input token sequence be:

  $$
  X = (x_1, x_2, \dots, x_n), \quad x_i \in \mathbb{R}^d
  $$

  where \$d\$ is the embedding dimension.

Define the soft prompt vectors:

$$
P = (p_1, p_2, \dots, p_m), \quad p_j \in \mathbb{R}^d
$$

The concatenated input becomes:

$$
X' = (p_1, p_2, \dots, p_m, x_1, x_2, \dots, x_n)
$$

The task objective function (e.g., cross-entropy for classification):

$$
\mathcal{L}(P) = - \sum_{(X, y)} \log p(y \mid X'; \theta, P)
$$

That is: freeze \$\theta\$, only optimize \$P\$.

---

## 📖 3. Simple Code Demonstration

Below is a simplified **Prompt Tuning** implementation using Hugging Face + PyTorch (example: text classification):

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class PromptTuningModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", prompt_length=20, num_labels=2):
        super().__init__()
        # 1. Pretrained model
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        
        # 2. Define soft prompt (trainable parameters)
        self.prompt_length = prompt_length
        self.hidden_size = self.bert.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, self.hidden_size))
        
        # 3. Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get original token embeddings
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        # Expand to batch dimension
        batch_size = input_ids.size(0)
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate soft prompt with original input
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # Adjust attention mask
        prompt_mask = torch.ones(batch_size, self.prompt_length).to(attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Feed into BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Classification
        logits = self.classifier(cls_output)
        return logits

# Example usage
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Prompt tuning is efficient!", return_tensors="pt")

model = PromptTuningModel()
logits = model(**inputs)
print(logits)
```



## 📖 Summary

* **Definition**: Prompt Tuning adds a **trainable soft prompt** before the input sequence, freezes the model, and only trains the prompt parameters.
* **Mathematical Form**: \$\mathcal{L}(P) = - \sum \log p(y \mid \[P; X]; \theta)\$.
* **Code Implementation**: Use `nn.Parameter` to define the soft prompt, and concatenate it before the embeddings.


