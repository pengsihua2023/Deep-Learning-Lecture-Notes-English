## Summary of Deep Learning Model Evaluation Methods

Deep learning model evaluation techniques are mainly used to measure model performance, generalization ability, and effectiveness on specific tasks. Below are some common evaluation techniques covering classification, regression, and generative models, divided into quantitative and qualitative methods:

### I. **Quantitative Evaluation Techniques**

These techniques evaluate model performance using numerical metrics, commonly applied in classification and regression tasks.

#### 1. **Evaluation Metrics for Classification Tasks**

* **Accuracy**: The proportion of correctly predicted samples over the total number of samples. Suitable for balanced datasets.

- Formula: \$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}\$

  * Limitation: Performs poorly on imbalanced datasets.

* **Precision**: The proportion of true positives among all predicted positives.

- Formula: \$\text{Precision} = \frac{TP}{TP + FP}\$

  * Suitable for scenarios where the cost of false positives (FP) is high.

* **Recall**: The proportion of true positives among all actual positives.

- Formula: \$\text{Recall} = \frac{TP}{TP + FN}\$

  * Suitable for scenarios where the cost of false negatives (FN) is high.

* **F1-Score**: The harmonic mean of precision and recall, balancing the trade-off between them.

- Formula: \$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}\$

  * Suitable for imbalanced datasets.

* **ROC Curve and AUC**:

  * ROC (Receiver Operating Characteristic) curve plots True Positive Rate (TPR) against False Positive Rate (FPR).
  * AUC (Area Under Curve) measures the model's ability to distinguish between classes, with values closer to 1 being better.
* **Confusion Matrix**: Displays the distribution of predictions for each class, showing TP, TN, FP, FN directly.
* **Multi-class Metrics**:

  * Macro-Average: Average of metrics (e.g., precision, recall) across classes, treating all classes equally.
  * Micro-Average: Aggregates TP, FP, FN across all classes before calculating metrics, favoring larger classes.
  * Weighted-Average: Weighted average of metrics according to class sample sizes.

#### 2. **Evaluation Metrics for Regression Tasks**

* **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values.

- Formula: \$\text{MSE} = \frac{1}{n} \sum\_{i=1}^n (y\_i - \hat{y}\_i)^2\$

  * Sensitive to outliers.

* **Root Mean Squared Error (RMSE)**: The square root of MSE, with the same units as the target variable, making it easier to interpret.

- Formula: \$\text{RMSE} = \sqrt{\text{MSE}}\$

* **Mean Absolute Error (MAE)**: The average of the absolute differences between predicted and actual values.

- Formula: \$\text{MAE} = \frac{1}{n} \sum\_{i=1}^n \left| y\_i - \hat{y}\_i \right|\$

  * Less sensitive to outliers.

* **Coefficient of Determination (R²)**: Measures the proportion of variance explained by the model, closer to 1 is better.

- Formula: \$R^2 = 1 - \frac{\sum (y\_i - \hat{y}\_i)^2}{\sum (y\_i - \bar{y})^2}\$

Where:

* \$y\_i\$: Actual value
* \$\hat{y}\_i\$: Predicted value
* \$\bar{y}\$: Mean of actual values

- **Mean Absolute Percentage Error (MAPE)**: Measures relative error, suitable for regression tasks with a clear scale.

  <img width="195" height="52" alt="image" src="https://github.com/user-attachments/assets/b01eb65b-8401-4b57-aa0f-6f1715dd2e33" />  

#### 3. **Evaluation Metrics for Generative Models**

* **Generative Adversarial Networks (GANs)**:

  * **Fréchet Inception Distance (FID)**: Measures similarity between distributions of generated and real images; lower is better.
  * **Inception Score (IS)**: Evaluates quality and diversity of generated images based on pretrained Inception model predictions.
* **Natural Language Generation**:

  * **BLEU (Bilingual Evaluation Understudy)**: Measures n-gram overlap between generated and reference texts, common in machine translation.
  * **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures recall against reference texts, often used in text summarization.
  * **Perplexity**: Evaluates how well a language model predicts the next word; lower indicates better performance.
  * **METEOR**: Improves on BLEU by considering synonyms, stemming, etc.

#### 4. **Cross-Validation**

* **K-Fold Cross-Validation**: Splits dataset into K folds, trains on K-1 and validates on 1, repeating K times and averaging performance.
* **Leave-One-Out Cross-Validation (LOOCV)**: Each sample is used once as validation; suitable for small datasets but computationally expensive.
* **Stratified K-Fold**: Ensures class distribution is consistent across folds, suitable for imbalanced datasets.

#### 5. **Other Quantitative Methods**

* **Learning Curves**: Plot training and validation performance versus dataset size to evaluate underfitting or overfitting.
* **Validation Set Performance**: Uses a separate validation set to evaluate generalization, commonly for hyperparameter tuning.
* **Test Set Performance**: Final evaluation on unseen data to ensure unbiased results.

### II. **Qualitative Evaluation Techniques**

Qualitative evaluation relies on subjective analysis or visualization to assess model outputs, suitable for generative tasks or those requiring human judgment.

* **Visualization Analysis**:

  * **Feature Map Visualization**: Displays activations of intermediate CNN layers to understand model focus.
  * **t-SNE/PCA**: Reduces high-dimensional data to 2D/3D to visualize distributions or learned representations.
  * **Generated Sample Inspection**: Manually check outputs (images, text, etc.) from generative models like GANs or diffusion models.
* **Human Evaluation**:

  * **Subjective Scoring**: Experts or users rate generated outputs for realism, coherence, or aesthetics.
  * **A/B Testing**: Compare outputs from two models to see which is preferred by humans.
* **Error Analysis**:

  * Inspect errors on specific samples to identify failure modes (e.g., bias, edge cases).
  * Analyze confusion matrices to find commonly confused classes.

### III. **Other Advanced Techniques**

* **Adversarial Evaluation**: Test robustness with adversarial examples to see if models are easily fooled.
* **Transfer Learning Evaluation**: Assess pretrained models on downstream tasks, common in large language or vision models.
* **Uncertainty Estimation**:

  * Use Bayesian methods or Monte Carlo dropout to estimate prediction confidence.
  * Quantify uncertainty on unseen data, critical in high-risk domains (e.g., medical diagnosis).
* **Fairness and Bias Evaluation**:

  * Measure performance differences across groups (e.g., gender, race).
  * Metrics include Equal Opportunity Difference or Demographic Parity.

### IV. **Considerations**

* **Choose Appropriate Metrics**: Select based on task type (classification, regression, generative) and application. For example, medical diagnosis may prioritize recall, while recommender systems may emphasize precision.
* **Dataset Splitting**: Ensure independence of training, validation, and test sets to avoid data leakage.
* **Overfitting Detection**: Compare training and validation performance to identify overfitting.
* **Task-Specificity**: Some domains (e.g., medical imaging, autonomous driving) may require customized metrics or evaluation pipelines.


