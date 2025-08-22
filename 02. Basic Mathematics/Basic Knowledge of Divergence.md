
# Basics of Divergence
## Table of Contents

1. [What is Divergence?](#1-what-is-divergence)

   * (1) Kullback–Leibler Divergence
   * (2) Jensen–Shannon Divergence
   * (3) Wasserstein Distance
2. [Applications in Deep Learning](#2-applications-in-deep-learning)

   * (1) Loss Functions
   * (2) Generative Models
   * (3) Distribution Matching
   * (4) Reinforcement Learning
3. [Divergence comparison](#3-Comparison-of-Divergences)

---

## 1. What is Divergence?

In mathematics and information theory, **divergence** usually refers to a measure of the difference between two probability distributions.

### (1) Kullback–Leibler Divergence (KL Divergence)

Discrete case:

$D_{\mathrm{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$

Continuous case:

$D_{\mathrm{KL}}(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx$

* **$P(x)$**: True distribution (target / data distribution), representing the true probability of event $x$.
* **$Q(x)$**: Approximate distribution or model distribution (approximation / model distribution), representing the model's estimated probability of event $x$.
* **Explanation**: $D_{\mathrm{KL}}(P\|Q)$ measures the amount of information loss when using $Q$ to approximate $P$.


### (2) Jensen–Shannon Divergence (JS Divergence)

$D_{\mathrm{JS}}(P \Vert Q) = \tfrac{1}{2} D_{\mathrm{KL}}(P \Vert M) + \tfrac{1}{2} D_{\mathrm{KL}}(Q \Vert M), \quad M = \tfrac{1}{2}(P+Q)$

**Properties:** symmetric, bounded.



### (3) Wasserstein Distance (Earth Mover’s Distance)

Described as “the minimum cost of transporting one distribution to another.” Commonly used in Generative Adversarial Networks (WGAN).

$W(P, Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y) \sim \gamma} \big[ \lVert x - y \rVert \big]$

---

## 2. Applications in Deep Learning

### (1) Loss Functions

Cross-entropy loss:

$H(P,Q) = H(P) + D_{\mathrm{KL}}(P \parallel Q)$



### (2) Generative Models

**Variational Autoencoder (VAE):**

$\mathcal{L_VAE} \= \mathcal{E_{q_\phi(z \mid x)}} \left[ \log p_\theta(x \mid z) \right] - D_{\mathrm{KL}}\left( q_\phi(z \mid x) \Vert p(z) \right) $

**Generative Adversarial Networks (GAN):**

* Original GAN: minimizes JS divergence
* WGAN: minimizes Wasserstein distance


### (3) Distribution Matching

Knowledge Distillation:

$\min_\theta D_{\mathrm{KL}}(P \parallel Q)$

where \$P\$ is the teacher distribution and \$Q\$ is the student distribution.



### (4) Reinforcement Learning

In policy optimization methods (e.g., TRPO, PPO), KL divergence is often used to constrain the difference between the old and new policies:

$D_{\mathrm{KL}}(\pi_{\text{old}} \parallel \pi_{\text{new}}) \leq \delta$


---

## 3. Comparison of Divergences

| Divergence                | Formula                                                                                                                                            | Properties                                                   | Advantages                                          | Limitations                                                | Applications                                                     |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| **Kullback–Leibler (KL)** | $D_{\mathrm{KL}}(P \Vert Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$                                                                                | Non-negative, asymmetric; =0 iff $P=Q$                       | Intuitive, easy to compute; linked to cross-entropy | Infinite if $Q(x)=0$ while $P(x)>0$; not symmetric         | Classification (cross-entropy loss), VAE, knowledge distillation |
| **Jensen–Shannon (JS)**   | $D_{\mathrm{JS}}(P \Vert Q) = \tfrac{1}{2}D_{\mathrm{KL}}(P \Vert M) + \tfrac{1}{2}D_{\mathrm{KL}}(Q \Vert M) \; M=\tfrac{1}{2}(P+Q)$ | Symmetric, bounded (between 0 and $\log 2$)                  | Symmetry, stability                                 | Gradient vanishing when distributions have no overlap      | Original GAN                                                     |
| **Wasserstein Distance**  | $W(P,Q)=\inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}\left[\lVert x-y\rVert\right]$                                                      | Metric; captures geometric differences between distributions | Smooth gradients even with disjoint distributions   | Computationally more expensive (optimal transport problem) | WGAN, distribution alignment                                     |

---



