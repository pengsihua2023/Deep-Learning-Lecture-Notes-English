# BERT：Bidirectional Encoder Representations from Transformers
## 📖 Introduction
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model proposed by Google in 2018. It is widely used in natural language processing (NLP) tasks such as text classification, question answering, and named entity recognition. The core of BERT lies in using a **bidirectional Transformer Encoder**, capturing deep semantic information through large-scale unsupervised pre-training, and then fine-tuning to adapt to specific tasks.
<div align="center"> 
<img width="350" height="280" alt="image" src="https://github.com/user-attachments/assets/313fa320-c931-4fb3-8fcb-5d81de615a21" />
</div>

<div align="center">
(This picture was obtained from BERT paublished paper at https://arxiv.org/abs/1810.04805.)
</div>


## 📖 Mathematical Description of BERT Model

### 1. Input Representation

For an input sequence

$$
x = \{x_1, x_2, \dots, x_n\},
$$

the input vector of BERT consists of **word embeddings**, **position embeddings**, and **segment embeddings**:

$$
h_i^{(0)} = E(x_i) + P(i) + S(s_i),
$$

where:

* \$E(x\_i)\$: word embedding vector with dimension \$d\$.
* \$P(i)\$: position embedding.
* \$S(s\_i)\$: sentence segment embedding (used to distinguish sentence A/B).



### 2. Transformer Encoder Layer

BERT is composed of \$L\$ stacked Transformer Encoder layers. The input to the \$l\$-th layer is \${h\_1^{(l-1)}, \dots, h\_n^{(l-1)}}\$, and the output is \${h\_1^{(l)}, \dots, h\_n^{(l)}}\$.

* (a) Multi-Head Self-Attention

First, compute the Query, Key, and Value vectors for each token:

$$
Q = H^{(l-1)} W_Q, \quad K = H^{(l-1)} W_K, \quad V = H^{(l-1)} W_V,
$$

where \$H^{(l-1)} \in \mathbb{R}^{n \times d}\$, and the projection matrices \$W\_Q, W\_K, W\_V \in \mathbb{R}^{d \times d\_k}\$.

Single-head attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
$$

Multi-head attention is defined as:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O,
$$

$$
\text{head}_i = \text{Attention}(Q W_Q^{(i)}, K W_K^{(i)}, V W_V^{(i)}).
$$

* (b) Feed Forward Network

Each position independently passes through a two-layer feed-forward network:

$$
\text{FFN}(h) = \text{GELU}(h W_1 + b_1) W_2 + b_2.
$$



### 3. Residual Connection and Layer Normalization

Each sublayer is followed by residual connection and normalization:

$$
\tilde{H}^{(l)} = \text{LayerNorm}(H^{(l-1)} + \text{MultiHead}(Q,K,V)),
$$

$$
H^{(l)} = \text{LayerNorm}(\tilde{H}^{(l)} + \text{FFN}(\tilde{H}^{(l)})).
$$



### 4. Pre-training Objectives

BERT has two main pre-training tasks:

* (a) Masked Language Model (MLM)

Randomly mask 15% of the input tokens and predict the masked words:

$$
\mathcal{L}_ {MLM} = - \sum_{i \in M} \log P(x_i \mid x_{\setminus M}),
$$

where \$M\$ is the set of masked positions.

* (b) Next Sentence Prediction (NSP)

Determine whether the two input sentences are consecutive:

$$
\mathcal{L}_{NSP} = - \big[ y \log P(\text{IsNext}) + (1-y)\log P(\text{NotNext}) \big].
$$

* (c) Total Loss

\$\mathcal{L} = \mathcal{L}\_ {MLM} + \mathcal{L}\_ {NSP}. \$

## 📖 Applicable Scenarios:

Text classification, question answering, NER, translation, etc.

---

## 📖 Specific Implementation: Text Classification + Dataset Loading + Attention Visualization

Below is a complete PyTorch code example using the `transformers` library from Hugging Face to implement the BERT model for a **text classification task** (sentiment analysis example with the IMDb dataset), including:

* Dataset loading (IMDb dataset, simplified version).
* BERT model fine-tuning.
* Attention weight visualization (showing the attention distribution of the `[CLS]` token).

### Code
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# 1. Custom dataset
class IMDbDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.texts = dataset['text']
        self.labels = dataset['label']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Load dataset and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('imdb', split='train[:1000]')  # Use the first 1000 samples of IMDb dataset
train_dataset = IMDbDataset(dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 3. Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification: positive/negative sentiment
model.train()

# 4. Train (fine-tune) the model
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}")

# 5. Testing and attention visualization
model.eval()
test_text = "This movie is fantastic!"
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get attention weights
model_with_attentions = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=True)
model_with_attentions.load_state_dict(model.state_dict())
model_with_attentions.to(device)
model_with_attentions.eval()

with torch.no_grad():
    outputs = model_with_attentions(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    attentions = outputs.attentions  # Attention weights (num_layers, batch_size, num_heads, seq_len, seq_len)

# Prediction results
labels = ['Negative', 'Positive']
pred_label = labels[torch.argmax(probs, dim=1).item()]
print(f"Text: {test_text}")
print(f"Predicted sentiment: {pred_label}")
print(f"Probabilities: {probs.tolist()}")

# 6. Visualize attention weights (last layer, first attention head)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attn = attentions[-1][0, 0].detach().cpu().numpy()  # Last layer, head 1
plt.figure(figsize=(10, 8))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title('Attention Weights (Last Layer, Head 1)')
plt.xlabel('Tokens')
plt.ylabel('Tokens')
plt.show()

```


## 📖 Code Explanation

* **Dataset Loading**:

  * Use the `datasets` library to load the IMDb dataset (sentiment analysis, binary classification: positive/negative). Take the first 1000 samples for simplification.
  * Define a custom `IMDbDataset` class to encode text into `input_ids` and `attention_mask`, and provide labels.
  * Use `DataLoader` for batch data loading, with `batch_size=8`.

* **Model**:

  * `BertForSequenceClassification`: a pre-trained BERT model with an added classification head (fully connected layer), where `num_labels=2` denotes positive/negative classification.
  * Use `bert-base-uncased` (12 layers, 768 dimensions, 110M parameters).

* **Training**:

  * Use Adam optimizer with a learning rate of `2e-5` (commonly used for BERT fine-tuning).
  * Train for 3 epochs, computing cross-entropy loss.

* **Testing**:

  * Predict sentiment for the example text “This movie is fantastic!” and output the sentiment and probability.

* **Attention Visualization**:

  * Load the model with `output_attentions=True` to obtain attention weights.
  * Plot the heatmap of the first attention head in the last layer, showing the attention distribution between tokens.

* **Output**:

  * Predicted sentiment (positive/negative).
  * Attention heatmap showing attention weights of `[CLS]` and other tokens.



## 📖 Notes

1. **Environment**:

   * Install dependencies: `pip install transformers datasets torch matplotlib seaborn`.
   * GPU acceleration: move the model and data to GPU (`model.to(device)`).

2. **Dataset**:

   * The IMDb dataset needs to be downloaded (the `datasets` library handles this automatically).
   * Can be replaced with other datasets (e.g., SST-2, GLUE).

3. **Fine-tuning**:

   * In real applications, it is recommended to use the full dataset (e.g., 25,000 IMDb training samples).
   * You can freeze some layers to speed up training: `model.bert.encoder.layer[:8].requires_grad = False`.

4. **Attention Visualization**:

   * The heatmap shows attention weights between tokens. The `[CLS]` token usually aggregates global information.
   * Can be extended to multi-head or multi-layer attention analysis.

5. **Computational Resources**:

   * BERT is computationally intensive, GPU execution is recommended.
   * Batch size and sequence length (`max_length`) should be adjusted based on hardware.



## 📖 Extensions

1. **Multi-task Learning**:

   * Question Answering: use `BertForQuestionAnswering`, input question and context, output answer span.
   * NER: use `BertForTokenClassification`, predict labels for each token.

2. **More Advanced Visualization**:

   * Plot multi-head attention: loop over `attentions[-1][0, i]` (where `i` is the attention head).
   * Analyze attention of a specific token: extract `attn[:, 0, :]` (attention of `[CLS]`).

3. **Inverse Problems**:

   * Estimate BERT parameters (e.g., attention weights) using observed data.



## 📖 Summary

BERT models semantics through a bidirectional Transformer. After pre-training, it can be fine-tuned for specific tasks. The code above demonstrates text classification on the IMDb dataset, including dataset loading, fine-tuning, and attention visualization.



