# Intermediate: Attention Mechanism
## 📖 Introduction
The attention mechanism in deep learning is a method that mimics human visual and cognitive systems, allowing neural networks to focus on relevant parts of input data during processing. By introducing the attention mechanism, neural networks can automatically learn to selectively focus on important information in the input, improving model performance and generalization.  
 <div align="center">
<img width="469" height="220" alt="image" src="https://github.com/user-attachments/assets/d5516323-dabc-4ba8-8c32-d1db9bc8e396" />
</div> 

<div align="center">
(This figure is cited from the Internet.)
</div> 

## 📖 Importance:  
The attention mechanism in Transformers is a cornerstone of modern deep learning, giving rise to models like BERT and GPT, driving advancements in NLP and multimodal tasks.  
Advanced courses can delve into variants of attention mechanisms (e.g., multi-head attention, self-attention).  
## 📖 Core Concepts:  
The attention mechanism allows models to focus on the most important parts of the input (e.g., key words in a sentence) by calculating weights through a "query-key-value" mechanism.  
## 📖 Applications: 
Chatbots (e.g., Grok), machine translation, text summarization.    
  
## 📖 Mathematical Description of Attention Mechanism

The core idea of the Attention mechanism is: **assign different weights to different elements in an information sequence, thereby highlighting "important" information and suppressing "irrelevant" information**.

### 1. Input Representation

Given an input vector sequence:

$$
X = [x_1, x_2, \dots, x_n], \quad x_i \in \mathbb{R}^d
$$

Map them into **Query, Key, and Value** vectors through linear transformations:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

Where:

* $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ are learnable parameters;
* $Q, K, V \in \mathbb{R}^{n \times d_k}$;
* Here **$d_k$** denotes the **dimension of the Key vector** (usually also equal to the dimension of Query);

* $d_k = \frac{d_{\text{model}}}{h}, \quad \text{scaling factor} = \sqrt{d_k}$ .


### 2. Attention Scoring Function

Compute **similarity score** to measure the relevance between Query and Key:

$$
\text{score}(q_i, k_j) = \frac{q_i \cdot k_j^\top}{\sqrt{d_k}}
$$

Where $\sqrt{d_k}$ is the scaling factor to prevent values from becoming too large.

### 3. Weight Distribution (Softmax)

Convert all scores into a probability distribution:

$$
\alpha_{ij} = \frac{\exp(\text{score}(q_i, k_j))}{\sum_{l=1}^n \exp(\text{score}(q_i, k_l))}
$$

Where $\alpha_{ij}$ denotes the attention weight of the $i$-th Query on the $j$-th Key.

### 4. Context Vector (Weighted Sum)

Weight the Value vectors according to the attention weights:

$$
z_i = \sum_{j=1}^n \alpha_{ij} v_j
$$

Obtaining the final context representation $z_i$.

### 5. Matrix Form (Scaled Dot-Product Attention)

The above steps can be written in a compact matrix form:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

### 6. Self-Attention
<div align="center">
<img width="469" height="220" alt="image" src="https://github.com/user-attachments/assets/a1dae221-067b-4f1c-a57f-caf7af22fbb5" />
</div>

Self-Attention is a special case of the attention mechanism where **Query (Q), Key (K), and Value (V) all come from the same sequence** $X$.

Formally:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

The attention output is:

$$
\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$


### Intuition

- **Purpose**:  
  Self-Attention enables each position in the sequence to attend to all other positions, thus capturing contextual dependencies.  

- **Example**:  
  In a sentence like *“The cat sat on the mat”*, the word *“cat”* can attend to *“sat”* and *“mat”* to better understand the context.  

- **Benefit**:  
  Unlike recurrent networks, Self-Attention processes all tokens in parallel, making it highly efficient and effective at capturing long-range dependencies.

### 7. Multi-Head Attention
<div align="center">
<img width="560" height="380" alt="image" src="https://github.com/user-attachments/assets/f2c01db3-28ea-4724-9a03-538cda1ebeb3" />
</div>
Instead of computing a single attention function, Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.

Formally:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

where each attention head is defined as:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

with $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ being learnable projection matrices.


### Intuition

- **Why multiple heads?**  
  A single attention function may be limited in its ability to capture different types of relationships.  
  Multiple heads allow the model to focus on different parts of the sequence (or different kinds of dependencies) simultaneously.

- **How it works?**  
  Each head projects the input into a lower-dimensional subspace, applies scaled dot-product attention, and outputs a context vector.  
  The results of all heads are concatenated and linearly projected to form the final representation.

- **Benefit**:  
  Multi-Head Attention enriches the model’s representational power and helps capture diverse relationships in the data.

### 8. Summary

* **Query–Key**: decides what to attend to;
* **Softmax weights**: distribute attention;
* **Value**: carries the information;
* **Self-Attention**: uses the same sequence as Q, K, V to capture dependencies within it;
* **Multi-Head Attention**: applies multiple attention heads in parallel to capture diverse patterns;
* **Final output**: weighted representation of the input.

The core formula is:

$$
\boxed{  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V  }
$$

---

# Code  
Add visualization of attention weights using a heatmap to display the attention weight matrix for the first sample, aiding in intuitively understanding how the attention mechanism focuses on relationships between different words. The code is based on the IMDb dataset and implements a simple Scaled Dot-Product Attention using PyTorch. Since you requested visualization, a heatmap will be generated to show the attention weights.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()
        self.dim = dim
    
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

def yield_tokens(dataset):
    for example in dataset:
        yield example['text'].lower().split()

def plot_attention_weights(attention_weights, tokens, title="Attention Weights Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png")
    plt.close()
    print("Attention heatmap saved as 'attention_heatmap.png'")

def main():
    # Load IMDb dataset
    dataset = load_dataset("imdb", split="train[:1000]")  # Use the first 1000 reviews
    batch_size = 32
    max_length = 20  # Shorten sequence length for visualization
    embed_dim = 64

    # Build vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # Create word embedding layer
    embedding = nn.Embedding(len(vocab), embed_dim)

    # Convert text to indices
    def text_pipeline(text):
        tokens = text.lower().split()[:max_length]
        tokens += ['<pad>'] * (max_length - len(tokens))
        return [vocab[token] for token in tokens]

    input_ids = torch.tensor([text_pipeline(example['text']) for example in dataset], dtype=torch.long)
    
    # Get word embeddings
    embedded = embedding(input_ids)  # [num_samples, max_length, embed_dim]
    
    # Initialize Attention model
    model = SimpleAttention(embed_dim)
    
    # Process in batches
    outputs = []
    attention_weights_list = []
    
    for i in range(0, len(dataset), batch_size):
        batch = embedded[i:i+batch_size]
        output, attention_weights = model(batch, batch, batch)
        outputs.append(output)
        attention_weights_list.append(attention_weights)
    
    outputs = torch.cat(outputs, dim=0)
    attention_weights = torch.cat(attention_weights_list, dim=0)
    
    # Print basic information
    print("Dataset size:", len(dataset))
    print("Sample text:", dataset[0]['text'][:100] + "...")
    print("Output shape:", outputs.shape)
    print("Attention weights shape:", attention_weights.shape)
    
    # Visualize attention weights for the first sample
    first_attention = attention_weights[0].detach().numpy()  # [max_length, max_length]
    first_tokens = dataset[0]['text'].lower().split()[:max_length]
    first_tokens += ['<pad>'] * (max_length - len(first_tokens))
    plot_attention_weights(first_attention, first_tokens)

if __name__ == "__main__":
    main()
```

## 📖 Summary:
1. **Dataset**: Continues using the first 1000 reviews from the IMDb dataset, with sequence length reduced to 20 for clearer heatmap visualization.
2. **Visualization**: Adds a `plot_attention_weights` function, using `seaborn` to draw a heatmap of attention weights for the first sample, saved as `attention_heatmap.png`.
3. **Heatmap Content**:
   - X-axis and Y-axis display the words in the input sentence (or `<pad>`).
   - Color intensity represents the magnitude of attention weights (using the `viridis` color map).
   - The heatmap intuitively shows which words the Attention mechanism focuses on more in relation to others.
4. **Dependencies**: Requires installation of `datasets`, `torchtext`, `matplotlib`, and `seaborn` (`pip install datasets torchtext matplotlib seaborn`).

## 📖 Execution Results:
- The program processes 1000 IMDb reviews, outputting dataset information, tensor shapes, and attention weight shapes.
- Generates a heatmap file `attention_heatmap.png`, displaying the attention weight matrix for the first review.
- Each cell in the heatmap represents the attention weight of a query word for a key word, with brighter colors indicating larger weights.

## 📖 Notes:
- The heatmap file is saved in the working directory and can be opened with an image viewer.
- With the sequence length limited to 20, the heatmap shows attention relationships for the first 20 words, making it suitable for intuitive analysis.
