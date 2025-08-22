
# Basics of Probability Distribution

## 1. Mathematical Definition

* **Discrete random variable** (e.g., dice, coin toss):
  Probability Mass Function (PMF):


  $$P(X = x_i) = p_i, \quad \sum_i p_i = 1$$


  Example: fair dice:


  $$P(X = k) = \tfrac{1}{6}, \quad k=1,2,\dots,6$$


* **Continuous random variable** (e.g., height, temperature):
  Probability Density Function (PDF):


  $$P(a \leq X \leq b) = \int_a^b f(x)\,dx, \quad \int_{-\infty}^{\infty} f(x)\,dx = 1$$


  Example: standard normal distribution:


  $$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\tfrac{x^2}{2}}$$



## 2. Common Probability Distributions

* Bernoulli distribution:


  $$P(X=1)=p, \quad P(X=0)=1-p$$


* Binomial distribution:


  $$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$


* Poisson distribution:


  $$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$


* Exponential distribution:


  $$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$


* Uniform distribution:


  $$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$



