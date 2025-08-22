

---

# 📘 Variational Autoencoder (VAE) — Mathematical Formulation

## **1. Latent Variable Model**

We assume observed data $x \in \mathbb{R}^D$ is generated from latent variables $z \in \mathbb{R}^L$ via:

$$
p_\theta(x, z) = p_\theta(x \mid z) \, p(z)
$$

with prior

$$
p(z) = \mathcal{N}(z; 0, I).
$$

---

## **2. Variational Inference**

Since the true posterior $p_\theta(z \mid x)$ is intractable, we approximate it with an encoder network:

$$
q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \, \mathrm{diag}(\sigma_\phi^2(x))).
$$

---

## **3. Evidence Lower Bound (ELBO)**

The log-likelihood has the variational lower bound:

$$
\log p_\theta(x) \;\geq\;
\mathbb{E}_{q_\phi(z \mid x)} \big[ \log p_\theta(x \mid z) \big]
- D_{\mathrm{KL}}\!\left( q_\phi(z \mid x) \;\|\; p(z) \right)
$$

This is the **VAE objective**.

---

## **4. Reparameterization Trick**

To enable backpropagation through stochastic $z$:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, 
\quad \epsilon \sim \mathcal{N}(0, I).
$$

---

## **5. Training Objective**

The loss function (to minimize) is:

$$
\mathcal{L}(\theta, \phi; x) =
- \mathbb{E}_{q_\phi(z \mid x)} \big[ \log p_\theta(x \mid z) \big]
+ D_{\mathrm{KL}}\!\left( q_\phi(z \mid x) \;\|\; p(z) \right)
$$
---

👉 With this, you have the **encoder** ($q_\phi(z \mid x)$), **decoder** ($p_\theta(x \mid z)$), and the **training loss** clearly described.

---

Would you like me to also prepare a **full GitHub-ready `README.md` template** (with MathJax script included), so that the formulas render properly on GitHub Pages as well?








