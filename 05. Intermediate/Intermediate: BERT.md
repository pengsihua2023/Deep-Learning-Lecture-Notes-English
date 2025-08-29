BERT: Bidirectional Encoder Representations from Transformers  
image  
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model proposed by Google in 2018, widely used in natural language processing (NLP) tasks such as text classification, question answering, named entity recognition, etc. The core of BERT is the use of a bidirectional Transformer Encoder, which captures deep semantic information through large-scale unsupervised pre-training, and then fine-tunes it for specific tasks.  

## Mathematical Description of the BERT Model  

### 1. Input Representation  
For an input sequence  

$$
x = x_1, x_2, …, x_n,
$$

the input vector of BERT consists of word embeddings, position embeddings, and segment embeddings:  

$$
h_i^{(0)} = E(x_i) + P(i) + S(s_i),
$$

where:  

- \(E(x_i)\): Word embedding vector, dimension \(d\).  
- \(P(i)\): Position embedding.  
- \(S(s_i)\): Sentence segment embedding (used to distinguish between sentence A/B).  

---

### 2. Transformer Encoder Layer  
BERT is composed of \(L\) stacked Transformer Encoders. The input to the \(l\)-th layer is  

$$
h_1^{(l-1)}, …, h_n^{(l-1)},
$$

and the output is  

$$
h_1^{(l)}, …, h_n^{(l)}.
$$

#### (a) Multi-Head Self-Attention  
First, compute the Query, Key, and Value vectors for each token:  

$$
Q = H^{(l-1)} W_Q, \quad K = H^{(l-1)} W_K, \quad V = H^{(l-1)} W_V,
$$

where \(H^{(l-1)} \in \mathbb{R}^{n \times d}\), projection matrices  
\(W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}\).  

Single-head attention:  

$$
Attention(Q, K, V) = softmax \left( \frac{QK^\top}{\sqrt{d_k}} \right) V.
$$

Multi-head attention:  

$$
MultiHead(Q, K, V) = Concat(head_1, …, head_h) W_O,
$$

$$
head_i = Attention(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)}).
$$

#### (b) Feed Forward Network (FFN)  
Each position independently passes through a two-layer feedforward network:  

$$
FFN(h) = GELU(hW_1 + b_1) W_2 + b_2.
$$

---

### 3. Residual Connection and Layer Normalization  
Each sub-layer has residual connection and normalization:  

$$
\tilde{H}^{(l)} = LayerNorm \big( H^{(l-1)} + MultiHead(Q, K, V) \big),
$$

$$
H^{(l)} = LayerNorm \big( \tilde{H}^{(l)} + FFN(\tilde{H}^{(l)}) \big).
$$

---

### 4. Pre-training Objectives  
BERT has two main pre-training tasks:  

#### (a) Masked Language Model (MLM)  
Randomly mask 15% of the tokens in the input, predict the masked words:  

$$
L_{MLM} = - \sum_{i \in M} \log P(x_i \mid x \setminus M),
$$

where \(M\) is the set of masked positions.  

#### (b) Next Sentence Prediction (NSP)  
Determine whether two input sentences are consecutive:  

$$
L_{NSP} = - \big[ y \log P(IsNext) + (1-y) \log P(NotNext) \big].
$$

#### (c) Total Loss  
$$
L = L_{MLM} + L_{NSP}.
$$

---

### Application Scenarios  
Text classification, question answering, NER, translation, etc.  

---

## Concrete Example Implementation: Text Classification + Dataset Loading + Attention Visualization  

Below is a complete PyTorch code example using Hugging Face’s `transformers` library to implement the BERT model for a text classification task (sentiment analysis on the IMDb dataset), including:  

- Dataset loading (IMDb dataset, simplified version).  
- Fine-tuning the BERT model.  
- Attention weight visualization (showing the attention distribution of the [CLS] token).  

---

### Code Example
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# 1. Custom Dataset
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

# 2. Load Dataset and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('imdb', split='train[:1000]')  # Use the first 1000 IMDb samples
train_dataset = IMDbDataset(dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 3. Load BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification: positive/negative sentiment
model.train()

# 4. Train (Fine-tune) the Model
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

# 5. Testing and Attention Visualization
model.eval()
test_text = "This movie is fantastic!"
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get Attention Weights
model_with_attentions = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=True)
model_with_attentions.load_state_dict(model.state_dict())
model_with_attentions.to(device)
model_with_attentions.eval()

with torch.no_grad():
    outputs = model_with_attentions(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    attentions = outputs.attentions  # Attention weights (num_layers, batch_size, num_heads, seq_len, seq_len)

# Prediction Result
labels = ['Negative', 'Positive']
pred_label = labels[torch.argmax(probs, dim=1).item()]
print(f"Text: {test_text}")
print(f"Predicted sentiment: {pred_label}")
print(f"Probabilities: {probs.tolist()}")

# 6. Visualize Attention Weights (Last Layer, Head 1)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attn = attentions[-1][0, 0].detach().cpu().numpy()  # Last layer, head 1
plt.figure(figsize=(10, 8))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title('Attention Weights (Last Layer, Head 1)')
plt.xlabel('Tokens')
plt.ylabel('Tokens')
plt.show()

