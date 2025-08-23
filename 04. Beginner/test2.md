
---

# 📘 Variational Autoencoder (VAE) 数学公式

## **1. 生成模型 (Latent Variable Model)**

$$
p_\theta(x, z) = p_\theta(x \mid z)\, p(z), \quad p(z) = \mathcal{N}(z; 0, I).
$$

---

## **2. 变分近似 (Variational Inference)**

用编码器近似后验分布：

$$
q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \, \mathrm{diag}(\sigma_\phi^2(x))).
$$

---

## **3. 证据下界 (Evidence Lower Bound, ELBO)**

VAE 的核心是最大化下界：

$$
\log p_\theta(x) \;\geq\;
\mathbb{E}_{q_\phi(z \mid x)} \big[ \log p_\theta(x \mid z) \big]
\;-\; D_{\mathrm{KL}}\!\left( q_\phi(z \mid x) \;\|\; p(z) \right).
$$

> ✅ 注意：这里是 **最大化 ELBO**，而不是最小化。

---

## **4. 重参数化技巧 (Reparameterization Trick)**

为了能反向传播：

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, 
\quad \epsilon \sim \mathcal{N}(0, I).
$$

---

## **5. 训练目标 (Loss Function)**

在实现时，我们最小化负的 ELBO：

$$
\mathcal{L}(\theta, \phi; x) 
= - \mathbb{E}_{q_\phi(z \mid x)} \big[ \log p_\theta(x \mid z) \big]
\;+\; D_{\mathrm{KL}}\!\left( q_\phi(z \mid x) \;\|\; p(z) \right).
$$

---


