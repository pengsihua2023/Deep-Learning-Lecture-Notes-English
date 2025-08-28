## Transformer
<div align="center">
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/8d064b02-6166-47ec-bfc6-fb031f94192c" />  
</div>

- Importance: Transformer is the core of modern Natural Language Processing (NLP), powering large models such as ChatGPT, representing the frontier of deep learning.
- Core concepts:  
Transformer uses the "attention mechanism" (Attention), focusing on the most important parts of the input (e.g., key words in a sentence).  
More efficient than RNNs, suitable for handling long sequences.  
- Applications: Chatbots (e.g., Grok), machine translation, text generation.  
 Why teach: Transformer represents the latest progress in AI.


## Mathematical description of Transformer
The Transformer architecture is a core model in NLP and deep learning, originally proposed by Vaswani et al. in the 2017 paper *"Attention is All You Need"*. Below is its mathematical description, covering the main components, including input representation, attention mechanism, positional encoding, feed-forward network, and layer normalization.    

### 1. Overall architecture  
The Transformer consists of an Encoder and a Decoder, each containing multiple stacked layers (usually $N$ layers). The encoder processes the input sequence, and the decoder generates the output sequence. The core innovation is the Self-Attention mechanism, replacing the sequential processing of traditional Recurrent Neural Networks (RNNs).  

Input representation  
The input sequence (e.g., words or tokens) is first converted into vector representations:
* **Word Embedding**: Each word is mapped into a fixed-dimensional vector $x_i \in \mathbb{R}^d$, usually implemented via an embedding matrix  
  $E \in \mathbb{R}^{|V|\times d}$, where $|V|$ is the vocabulary size and $d$ is the embedding dimension.

* **Positional Encoding**: Since Transformer does not have inherent sequential order information, positional encoding (Positional Encoding) is added to capture the position of words.  
  Positional encoding $PE$ can be generated using fixed formulas:  

$$
PE{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right), 
\quad
PE{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
$$

Where $pos$ is the position of the word in the sequence, and $i$ is the dimension index. The final input is:  

$$
x_i = E_i + PE(pos_i)
$$

Where:  

* $E_i$ represents the embedding vector of word $w_i$ (i.e., the result retrieved from embedding matrix $E$);  
* $PE(pos_i)$ represents the positional encoding vector;  
* The two are added element-wise as the input to the Transformer.  

- Encoder  
Each encoder layer contains two main sub-modules:  

Multi-Head Self-Attention  
Feed-Forward Neural Network (FFN)  

Each sub-module is followed by a Residual Connection and Layer Normalization.  

- Decoder  
The decoder is similar to the encoder but includes Masked Self-Attention (to prevent future information leakage) and Encoder-Decoder Attention.  



### 2. Multi-Head Self-Attention Mechanism 
Self-attention is the core of the Transformer, allowing the model to focus on other words in the sequence when processing each word.  

Scaled Dot-Product Attention  

For an input sequence $X \in \mathbb{R}^{n \times d}$ ($n$ is sequence length, $d$ is embedding dimension), the steps for calculating attention scores are as follows:  

**1. Generate Query, Key, and Value:**  

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

Where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ are learnable projection matrices, and $d_k$ is the dimension of the attention head.  

**2. Calculate Attention Weights:**  

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

* $QK^T \in \mathbb{R}^{n \times n}$ represents the dot product between query and key, measuring the correlation between words.  
* $\sqrt{d_k}$ is a scaling factor to prevent large dot products from causing softmax saturation.  
* The softmax operation normalizes each row, yielding attention weights, applied to the value vector $V$.  



### Multi-Head Mechanism  

Multi-head attention splits $Q, K, V$ into $h$ heads, each independently computing attention:  

$$
MultiHead(Q, K, V) = Concat(head_1, \ldots, head_h)W^O
$$

Where:  

$$
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}, \quad W^O \in \mathbb{R}^{h \cdot d_k \times d}, \quad h \text{ is the number of heads}, d_k = \frac{d}{h}
$$



### Masked Self-Attention (Decoder Specific)  

In the decoder, to prevent the current word from attending to subsequent words, a mask matrix $M$ is introduced, making attention weights at future positions $-\infty$:  

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$



### 3. Feed-Forward Neural Network (FFN)  
Each encoder and decoder layer contains a position-wise feed-forward network, applied to each word vector:  

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2
$$  

Where $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$, and $d_{ff}$ is usually much larger than $d$ (e.g., $d_{ff} = 4d$).  



### 4. Residual Connection and Layer Normalization  
Each sub-module (Self-Attention or FFN) is followed by a residual connection and layer normalization:  

$$
y = \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$  

Where Sublayer is either Attention or FFN, and LayerNorm is defined as:  

$$
\mathrm{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$  

$\mu$ and $\sigma^2$ are the mean and variance of the input vector, and $\gamma, \beta$ are learnable parameters.  



### 5. Encoder-Decoder Attention  
The additional attention layer in the decoder uses the encoder’s output $K, V$ and the decoder’s $Q$:  

$\mathrm{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$  

This allows the decoder to attend to the context of the input sequence.  



### 6. Output Layer  

The final decoder layer generates output probabilities via linear transformation and softmax:  

$$
P(y_i) = \mathrm{softmax}(z W_{\text{out}} + b_{\text{out}})
$$  

Where $z$ is the output of the final decoder layer, $W_{\text{out}} \in \mathbb{R}^{d \times |V|}$.  



### 7. Output Layer  
The final decoder layer generates output probabilities via linear transformation and softmax:  

$P(y_i) = \mathrm{softmax}(z W_{\text{out}} + b_{\text{out}})$  

Where $z$ is the output of the final decoder layer, $W_{\text{out}} \in \mathbb{R}^{d \times |V|}$.  



### 8. Loss Function  
Training usually uses cross-entropy loss, with the objective of maximizing the probability of the correct output sequence:  

$\mathcal{L} = -\sum_{i=1}^{T} \log P(y_i \mid y_{<i}, X)$  

Where $T$ is the output sequence length, and $y_{<i}$ is the already generated words.  



### 9. Summary  
The mathematical core of Transformer lies in:  

Self-Attention: Capturing relationships within the sequence via Q, K, V.  
Multi-Head Mechanism: Capturing multiple semantic relationships in parallel.  
Positional Encoding: Compensating for the lack of sequential order information.  
Residuals and Normalization: Stabilizing training and accelerating convergence.  

---

**Full Transformer vs Encoder Transformer**  

### Comparison of Two Transformers  

| Component | Full Transformer | Encoder Transformer |
| ---- | ------------- | -------------- |
| Encoder  | ✅             | ✅              |
| Decoder  | ✅             | ❌              |
| Suitable tasks | Translation, Summarization         | Classification, Sentiment Analysis        |
| Input/Output | Sequence→Sequence         | Sequence→Class          |



## Transformer with Only Encoder  
```python

```

### Summary

Data: Synthetic data generated and classified by statistical features
Model: Transformer with only encoder, for sequence classification
Decoder: Not included, as sequence generation is not needed
Flow: Data generation → Preprocessing → Embedding → Positional Encoding → Self-Attention → Pooling → Classification
This model is suitable for learning Transformer basics, especially Self-Attention and Positional Encoding!
