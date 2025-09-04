
# LIME Model Interpretation Method (Local Interpretable Model-agnostic Explanations)

## 📖 1. Definition

**LIME** is a **model-agnostic** local interpretability method used to explain the predictions of any black-box model.
Core idea:

* Given an input sample \$x\$ and model \$f\$, we want to know which input features have the greatest influence on the prediction \$f(x)\$.
* LIME does this by:

  1. Generating many perturbed samples \$x'\$ around \$x\$;
  2. Using the black-box model \$f\$ to predict these perturbed samples;
  3. Weighting these samples based on their “proximity” to the original sample;
  4. Training a simple **interpretable model (e.g., linear regression, decision tree)** on these samples;
  5. Using the coefficients of that model as the local explanation.

---

## 📖 2. Mathematical Description

Let:

* Black-box model: \$f: \mathbb{R}^d \to \mathbb{R}\$
* Target input: \$x \in \mathbb{R}^d\$
* Neighborhood sampling: generated samples \${z\_i}\_{i=1}^N\$
* Proximity function: \$\pi\_x(z)\$, measures similarity between \$z\$ and \$x\$ (often an RBF kernel)

LIME trains a simple explanation model \$g \in G\$ (e.g., linear model), with the optimization objective:

$$
\underset{g \in G}{\arg\min} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)
$$

where:

* \$\mathcal{L}(f, g, \pi\_x) = \sum\_i \pi\_x(z\_i),(f(z\_i) - g(z\_i))^2\$, i.e., weighted fitting error;
* \$\Omega(g)\$ is a complexity penalty for the explanation model (e.g., limiting the number of features).

Ultimately, the parameters of \$g\$ represent the local explanation around input \$x\$.

---

## 📖 3. Minimal Code Example

We use the `lime` library to explain the prediction of an sklearn logistic regression classifier.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Train a black-box model (logistic regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode="classification"
)

# 4. Select a sample to explain
i = 0
exp = explainer.explain_instance(
    data_row=X_test[i],
    predict_fn=model.predict_proba,
    num_features=2  # Explain only the 2 most important features
)

# 5. Print explanation results
print("True class:", iris.target_names[y_test[i]])
print("Predicted class:", iris.target_names[model.predict(X_test[i].reshape(1, -1))[0]])
print("LIME explanation:")
print(exp.as_list())
```

Sample output:

```
True class: versicolor
Predicted class: versicolor
LIME explanation:
[('petal width (cm) <= 1.75', -0.25), ('petal length (cm) > 4.8', +0.35)]
```

Explanation: Around this sample, the model’s prediction is mainly influenced by petal width and petal length.

---

## 📖 Summary

* **LIME definition**: explains black-box predictions through local perturbations + fitting a simple model.
* **Formula**: \$\arg\min\_g \sum\_i \pi\_x(z\_i),(f(z\_i)-g(z\_i))^2 + \Omega(g)\$.
* **Code**: easily implemented in a few lines using the `lime` library.

---

Below is a **complete example of LIME for text classification model interpretation**. We use a simple sentiment analysis model (positive/negative review classification), then use **LIME** to explain the prediction.

## LIME Explaining Text Classification Model

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import lime
import lime.lime_text

# 1. Prepare data (here we use 20 newsgroups dataset, simplified into binary classification)
categories = ['rec.autos', 'sci.med']  # Two topics: autos vs medicine
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# 2. Build text classification model (TF-IDF + logistic regression)
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(train.data, train.target)

# 3. Initialize LIME explainer
class_names = train.target_names
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

# 4. Select a test sample
idx = 10
text_sample = test.data[idx]
true_label = class_names[test.target[idx]]
pred_label = class_names[model.predict([text_sample])[0]]

# 5. LIME explanation
exp = explainer.explain_instance(
    text_instance=text_sample,
    classifier_fn=model.predict_proba,
    num_features=5   # Show the top 5 important words
)

# 6. Print results
print("Text sample:", text_sample[:200], "...")
print(f"True class: {true_label}")
print(f"Predicted class: {pred_label}")
print("\nLIME explanation (word importance):")
print(exp.as_list())
```

### Example Output (may look like this)

```
Text sample: I recently bought a new car and I really love driving it ...  
True class: rec.autos  
Predicted class: rec.autos  

LIME explanation (word importance):
[('car', +0.42), ('driving', +0.25), ('engine', +0.15), ('doctor', -0.18), ('medicine', -0.22)]
```

Explanation:

* The model sees **"car"**, **"driving"**, **"engine"** as strongly pushing prediction toward "autos";
* Words like **"doctor"**, **"medicine"** push prediction toward "medicine".

---

## 📖 Summary

* **LIME for text**: perturbs text (deleting/replacing words) and observes prediction changes to identify the most influential words.
* **Output**: lists positive/negative important words, helping interpret model decisions.

---

Below is a **complete example of LIME for image classification model interpretation**. We use the **MNIST handwritten digit classifier** from Keras, and then use **LIME** to explain why the model predicted a certain image as “8”.

## LIME Explaining Image Classification Model

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# 1. Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train = np.expand_dims(x_train, -1)  # [N, 28, 28, 1]
x_test = np.expand_dims(x_test, -1)

# 2. Simple CNN model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1)  # Train only 1 epoch for demo

# 3. Select an image
idx = 0
image = x_test[idx]
label = y_test[idx]
pred = np.argmax(model.predict(image[np.newaxis]))
print(f"True label: {label}, Predicted label: {pred}")

# 4. LIME explanation
explainer = lime_image.LimeImageExplainer()

def predict_fn(images):
    return model.predict(images)

explanation = explainer.explain_instance(
    image=image.astype('double'),
    classifier_fn=predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

# 5. Visualization
from skimage.color import gray2rgb
temp, mask = explanation.get_image_and_mask(
    label=pred, 
    positive_only=True, 
    hide_rest=False, 
    num_features=10, 
    min_weight=0.0
)

plt.subplot(1,2,1)
plt.title(f"Original Image (label={label}, pred={pred})")
plt.imshow(image.squeeze(), cmap="gray")

plt.subplot(1,2,2)
plt.title("LIME Explanation")
plt.imshow(mark_boundaries(gray2rgb(temp), mask))
plt.show()
```

### Results

* **Left**: the original handwritten digit (e.g., “8”).
* **Right**: LIME highlights the most important local regions (green/bounded areas).

For example:

* If predicted as **8**, LIME may highlight the “middle loop” and the “top/bottom arcs”;
* If predicted as **3**, LIME may highlight the “upper arc” and the “lower arc”.

---

## 📖 Summary

* **LIME for images**: perturbs different regions (e.g., masking superpixels) and observes prediction changes, identifying regions most relied upon by the model.
* **Benefit**: does not depend on specific model structures (works with CNNs, Transformers, etc.), truly **model-agnostic**.

---

## 📖 LIME Comparison Across Text / Images / Tabular

| Data Type        | Perturbation Method                                               | Interpretable Model               | Explanation Output                                             | Application Scenarios                                              |
| ---------------- | ----------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Tabular Data** | Randomly sample and replace feature values (local neighborhood)   | Linear regression / Decision tree | Feature importance list (weights, positive/negative influence) | Structured data modeling (credit scoring, medical risk prediction) |
| **Text Data**    | Randomly delete/mask words or n-grams, observe prediction changes | Linear model (bag-of-words)       | Word importance ranking (which words drive prediction)         | Sentiment analysis, topic classification, text classification      |
| **Image Data**   | Split image into superpixels, then randomly mask regions          | Linear model (on superpixels)     | Heatmap / highlighted regions (key areas)                      | Image classification, medical imaging, object detection            |

---

## 📖 Summary

* **Tabular data** → LIME tells you which features (age, income, blood pressure, etc.) influence the prediction most around a sample.
* **Text data** → LIME tells you which words/phrases push the prediction (e.g., “great” or “terrible” in sentiment analysis).
* **Image data** → LIME tells you which image regions (eyes, edges, contours, etc.) drive the prediction.

---

## 📖  LIME vs SHAP Comparison

| Feature              | **LIME**                                                            | **SHAP**                                                                                          |   |      |   |                                  |
| -------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | - | ---- | - | -------------------------------- |
| **Full name**        | Local Interpretable Model-agnostic Explanations                     | SHapley Additive exPlanations                                                                     |   |      |   |                                  |
| **Core Idea**        | Local sampling → fit a simple model to approximate black-box        | Based on game theory **Shapley values** → measure marginal contribution of each feature           |   |      |   |                                  |
| **Formula**          | \$\arg\min\_g \sum\_i \pi\_x(z\_i)(f(z\_i)-g(z\_i))^2 + \Omega(g)\$ | \$\phi\_i = \sum\_{S \subseteq F\setminus{i}} \frac{                                              | S | !(M- | S | -1)!}{M!},\[f(S\cup {i})-f(S)]\$ |
| **Model Dependence** | Completely model-agnostic                                           | Also model-agnostic, but with optimized versions (TreeSHAP, DeepSHAP)                             |   |      |   |                                  |
| **Result Type**      | Local explanation: feature influence around a sample                | Global + local explanations: each feature’s marginal contribution, with game-theoretic guarantees |   |      |   |                                  |
| **Stability**        | Unstable (different samples may give different explanations)        | Stable (Shapley values are unique, satisfy fairness axioms)                                       |   |      |   |                                  |
| **Complexity**       | Low (relies on sampling + fitting linear model)                     | High (original Shapley is exponential, but approximations exist)                                  |   |      |   |                                  |
| **Interpretability** | Simple, intuitive, quick local explanation                          | More rigorous, theoretically sound, explanations more trustworthy                                 |   |      |   |                                  |
| **Use Cases**        | Quick approximate explanations, focusing on individual samples      | Strict, stable explanations, especially in high-risk fields (medical, finance)                    |   |      |   |                                  |

---

## 📖 Final Summary

* **LIME**: intuitive, fast, model-agnostic, but explanations can be unstable. Best for quick exploration and debugging.
* **SHAP**: Shapley-based, mathematically guaranteed, more stable and trustworthy, but computationally expensive. Best for high-reliability applications.



