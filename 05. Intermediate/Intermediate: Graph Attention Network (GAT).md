
# Graph Attention Network (GAT)

## I. Definition

Graph Attention Network (GAT) is a type of Graph Neural Network (GNN). It computes the contribution of neighboring nodes to the target node representation through the **attention mechanism** on graph structures, instead of using fixed normalized weights like GCN. This allows the model to automatically learn which neighbors are more important.

---

## II. Mathematical Description

Let the graph be \$G=(V,E)\$, with node set \$V\$ and edge set \$E\$. The input feature of node \$i\$ is \$h\_i \in \mathbb{R}^F\$.

1. **Linear Mapping**

$$
z_i = W h_i, \quad W \in \mathbb{R}^{F' \times F}
$$

2. **Attention Scoring** (for neighbors \$j\$ of \$i\$, including self-loop):

$$
e_{ij} = \text{LeakyReLU}\big(a^\top [z_i \, \| \, z_j]\big), \quad a \in \mathbb{R}^{2F'}
$$

3. **Normalization (neighbor softmax)**

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)\cup\{i\}} \exp(e_{ik})}
$$

4. **Aggregation Update**

$$
h'_i = \sigma\!\left(\sum_{j \in \mathcal{N}(i)\cup\{i\}} \alpha_{ij}\, z_j \right)
$$

Multi-head attention: repeat the above computation \$H\$ times, then concatenate or average.

---

## III. Minimal Code (PyTorch)

> Suitable for small graphs, teaching demonstrations; for large graphs please use sparse implementations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.6, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat

        # Linear transformation
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        # Attention parameter a: one vector per head
        self.a = nn.Parameter(torch.empty(heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        N = x.size(0)
        z = self.W(x).view(N, self.heads, self.out_dim)   # (N, H, F')

        # Construct pair (i,j)
        z_i = z.unsqueeze(1).expand(N, N, self.heads, self.out_dim)
        z_j = z.unsqueeze(0).expand(N, N, self.heads, self.out_dim)
        e = self.leakyrelu(((torch.cat([z_i, z_j], dim=-1)) * self.a).sum(-1))  # (N,N,H)

        # Adjacency matrix with self-loop
        adj_hat = adj + torch.eye(N, device=adj.device)
        mask = adj_hat.unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask == 0, float('-inf'))

        # softmax
        alpha = torch.softmax(e, dim=1)  # (N,N,H)
        alpha = self.dropout(alpha)

        # Aggregation
        out = (alpha.unsqueeze(-1) * z_j).sum(1)  # (N,H,F')

        if self.concat:
            return out.reshape(N, self.heads * self.out_dim)
        else:
            return out.mean(1)

# Small test: two-layer GAT
class TinyGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATLayer(hidden_dim*heads, out_dim, heads=1, concat=False)

    def forward(self, x, adj):
        x = F.elu(self.gat1(x, adj))
        return self.gat2(x, adj)

if __name__ == "__main__":
    # Example: 4-node small graph
    adj = torch.tensor([
        [0,1,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [0,1,0,0]
    ], dtype=torch.float32)

    x = torch.randn(4, 3)   # Node features
    model = TinyGAT(3, 8, 2, heads=2)
    out = model(x, adj)
    print(out)
```

The output is a 2-dimensional representation of each node, which can be directly fed into softmax for classification.

---



