## BERT：Bidirectional Encoder Representations from Transformers
<div align="center"> 
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/313fa320-c931-4fb3-8fcb-5d81de615a21" />
</div>


BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model proposed by Google in 2018. It is widely used in natural language processing (NLP) tasks such as text classification, question answering, and named entity recognition. The core of BERT lies in using a **bidirectional Transformer Encoder**, capturing deep semantic information through large-scale unsupervised pre-training, and then fine-tuning to adapt to specific tasks.

## Mathematical Description of BERT Model

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

---

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





### Applicable Scenarios:

Text classification, question answering, NER, translation, etc.

---

#### **Specific Implementation: Text Classification + Dataset Loading + Attention Visualization**

Below is a complete PyTorch code example using the `transformers` library from Hugging Face to implement the BERT model for a **text classification task** (sentiment analysis example with the IMDb dataset), including:

* Dataset loading (IMDb dataset, simplified version).
* BERT model fine-tuning.
* Attention weight visualization (showing the attention distribution of the `[CLS]` token).

---


