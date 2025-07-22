## Intermediate: Attention-based Model
## Attention-based Models (Transformer Expansion)
- **Importance**:  
The attention mechanism in Transformers is a cornerstone of modern deep learning, giving rise to models like BERT and GPT, driving advancements in NLP and multimodal tasks.  
Advanced courses can delve into variants of attention mechanisms (e.g., multi-head attention, self-attention).  
- **Core Concepts**:  
The attention mechanism allows models to focus on the most important parts of the input (e.g., key words in a sentence) by calculating weights through a "query-key-value" mechanism.  
- **Applications**: Chatbots (e.g., Grok), machine translation, text summarization.  
<img width="998" height="641" alt="image" src="https://github.com/user-attachments/assets/a78ff1d6-3d30-43e6-b8e2-40acad211a7f" />  
The attention mechanism in deep learning is a method that mimics human visual and cognitive systems, allowing neural networks to focus on relevant parts of input data during processing. By introducing the attention mechanism, neural networks can automatically learn to selectively focus on important information in the input, improving model performance and generalization.  
The image above effectively illustrates the attention mechanism, showing how humans efficiently allocate limited attention resources when viewing an image. The red areas indicate targets that the visual system prioritizes, demonstrating that people tend to focus more attention on human faces.  

## Code  
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
    # 加载IMDb数据集
    dataset = load_dataset("imdb", split="train[:1000]")  # 使用前1000条评论
    batch_size = 32
    max_length = 20  # 缩短序列长度以便可视化
    embed_dim = 64

    # 构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # 创建词嵌入层
    embedding = nn.Embedding(len(vocab), embed_dim)

    # 将文本转换为索引
    def text_pipeline(text):
        tokens = text.lower().split()[:max_length]
        tokens += ['<pad>'] * (max_length - len(tokens))
        return [vocab[token] for token in tokens]

    input_ids = torch.tensor([text_pipeline(example['text']) for example in dataset], dtype=torch.long)
    
    # 获取词嵌入
    embedded = embedding(input_ids)  # [num_samples, max_length, embed_dim]
    
    # 初始化Attention模型
    model = SimpleAttention(embed_dim)
    
    # 分批处理
    outputs = []
    attention_weights_list = []
    
    for i in range(0, len(dataset), batch_size):
        batch = embedded[i:i+batch_size]
        output, attention_weights = model(batch, batch, batch)
        outputs.append(output)
        attention_weights_list.append(attention_weights)
    
    outputs = torch.cat(outputs, dim=0)
    attention_weights = torch.cat(attention_weights_list, dim=0)
    
    # 打印基本信息
    print("Dataset size:", len(dataset))
    print("Sample text:", dataset[0]['text'][:100] + "...")
    print("Output shape:", outputs.shape)
    print("Attention weights shape:", attention_weights.shape)
    
    # 可视化第一个样本的注意力权重
    first_attention = attention_weights[0].detach().numpy()  # [max_length, max_length]
    first_tokens = dataset[0]['text'].lower().split()[:max_length]
    first_tokens += ['<pad>'] * (max_length - len(first_tokens))
    plot_attention_weights(first_attention, first_tokens)

if __name__ == "__main__":
    main()
```

### 修改内容：
1. **数据集**：继续使用IMDb数据集的前1000条评论，序列长度缩短至20，以便热图更易读。
2. **可视化**：添加`plot_attention_weights`函数，使用`seaborn`绘制第一个样本的注意力权重热图，保存为`attention_heatmap.png`。
3. **热图内容**：
   - X轴和Y轴显示输入句子的词（或`<pad>`）。
   - 颜色深浅表示注意力权重大小（通过`viridis`颜色映射）。
   - 热图直观展示哪些词在Attention机制中对其他词的关注程度更高。
4. **依赖**：需安装`datasets`、`torchtext`、`matplotlib`和`seaborn`（`pip install datasets torchtext matplotlib seaborn`）。

### 运行结果：
- 程序将处理1000条IMDb评论，输出数据集信息、输出张量形状和注意力权重形状。
- 生成一个热图文件`attention_heatmap.png`，展示第一个评论的注意力权重矩阵。
- 热图中的每个单元格表示query词对key词的注意力权重，颜色越亮表示权重越大。

### 注意：
- 热图文件保存在运行目录下，可用图像查看器打开。
- 由于序列长度限制为20，热图显示前20个词的注意力关系，适合直观分析。


