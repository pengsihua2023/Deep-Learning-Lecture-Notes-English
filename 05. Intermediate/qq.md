好的，我已经把你代码里的中文完整翻译成英文，同时保持了原有的 Markdown 格式和排版不变：

---

## Contrastive Learning

**Contrastive Learning** is a type of **Self-Supervised Learning** method. It constructs contrastive tasks between samples, enabling the model to learn more discriminative feature representations.

The core idea is:

* Pull “similar” samples closer (smaller embedding distance),
* Push “dissimilar” samples apart (larger embedding distance).

This learning approach usually does not require manual labels. Instead, it automatically generates **positive pairs** and **negative pairs** through data augmentation or contextual information.

### Formal Definition

Given a sample \$x\$, two views \$x\_i, x\_j\$ are obtained via data augmentation (such as image rotation, cropping, noise injection, etc.). They are regarded as **positive pairs**; views from other samples (such as \$x\_k\$) are **negative samples**.

The objective is to learn an encoder \$f(\cdot)\$ that maps samples to the feature space, such that:

* **Positive pairs** are as close as possible in the feature space:

  \$\text{sim}(f(x\_i), f(x\_j)) \ \text{maximize}\$

* **Negative pairs** are as far apart as possible in the feature space:

  \$\text{sim}(f(x\_i), f(x\_k)) \ \text{minimize}\$

Here, \$\text{sim}(\cdot,\cdot)\$ is usually **cosine similarity** or **inner product**.

### Typical Methods

* **SimCLR**: Constructs positive and negative pairs through large-scale data augmentation and trains with the InfoNCE loss.
* **MoCo** (Momentum Contrast): Introduces a momentum update mechanism to maintain a large dynamic negative sample queue.
* **BYOL** (Bootstrap Your Own Latent): Does not explicitly use negative samples, but learns through the interaction of two networks (online network & target network).

### Summary

The essence of contrastive learning is:

* It does not rely on large amounts of manual labels,
* It learns feature representations by “bringing similar samples closer, separating dissimilar samples,”
* It has wide applications in computer vision, natural language processing, speech, etc.

<div align="center">
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/5d389da9-c6c7-46d5-a1c5-096422a5328b" />
</div>

* Importance:
  Contrastive learning is a self-supervised learning method that extracts high-quality feature representations by teaching models to distinguish between “similar” and “dissimilar” data pairs.
  It is the core of modern unsupervised learning, driving successes such as SimCLR, MoCo (computer vision), and CLIP (multimodal learning).
  In scenarios with scarce labeled data (e.g., medical imaging, low-resource languages), contrastive learning can significantly reduce reliance on manual labeling.

* Core Concept:
  The goal of contrastive learning is to bring similar data pairs (positive pairs) closer together in feature space, while pushing dissimilar data pairs (negative pairs) farther apart.
  It optimizes feature representations using contrastive loss functions (e.g., InfoNCE loss).

* Analogy: Like a “find your friends game,” the model learns to cluster “friends” (similar images/texts) together while separating “strangers” (dissimilar data).

* Applications:
  Image classification (SimCLR, MoCo): Achieve high-accuracy classification with limited labeled data.
  Multimodal learning (CLIP): Image-text retrieval, image generation (e.g., DALL·E).

---

要不要我帮你把中英双语对照的版本也排出来，方便你复习和对比？
