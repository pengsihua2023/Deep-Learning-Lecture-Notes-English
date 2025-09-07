# DeepChem Library for Molecular Toxicity Prediction
## 📖 Introduction
The following is a real-world example using the DeepChem library. This example is based on the Tox21 dataset (Toxicology in the 21st Century), a public database containing toxicity measurements for approximately 12,000 compounds across 12 biological targets, sourced from the 2014 Tox21 Data Challenge. The dataset uses SMILES to represent molecular structures and is commonly employed for multitask classification problems in molecular toxicity prediction.
  
<img width="410" height="219" alt="image" src="https://github.com/user-attachments/assets/6a0b26c8-ef80-4fc5-8612-3f12dc693c23" />
  
This example demonstrates how to load the Tox21 dataset and train and evaluate two models: GraphConvModel (a graph convolutional model) and RobustMultitaskClassifier (a robust multitask classifier). The code is executable (assuming DeepChem, TensorFlow, and other dependencies are installed) and is derived from DeepChem’s tutorials and documentation. I have combined the code into a complete script and added comments.

## 📖 Complete Code Example
```python
import deepchem as dc
from deepchem.models import GraphConvModel, RobustMultitaskClassifier
from deepchem.metrics import roc_auc_score  # Use ROC-AUC as the evaluation metric

# Part 1: Load Tox21 dataset with GraphConv featurizer (graph representation)
tasks_graph, datasets_graph, transformers_graph = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset_graph, valid_dataset_graph, test_dataset_graph = datasets_graph

# Define and train GraphConvModel (graph convolutional model for handling molecular graph structures)
gcn_model = GraphConvModel(n_tasks=len(tasks_graph), mode='classification', dropout=0.2)
gcn_model.fit(train_dataset_graph, nb_epoch=50)  # Train for 50 epochs

# Evaluate GraphConvModel
train_score_gcn = gcn_model.evaluate(train_dataset_graph, [roc_auc_score], transformers_graph)
test_score_gcn = gcn_model.evaluate(test_dataset_graph, [roc_auc_score], transformers_graph)
print("GraphConvModel AUC-ROC metrics:")
print(f"Training set score: {train_score_gcn}")
print(f"Test set score: {test_score_gcn}")

# Part 2: Load Tox21 dataset with ECFP featurizer (circular fingerprint representation, non-graph model)
tasks_ecfp, datasets_ecfp, transformers_ecfp = dc.molnet.load_tox21(featurizer='ECFP')
train_dataset_ecfp, valid_dataset_ecfp, test_dataset_ecfp = datasets_ecfp

# Define and train RobustMultitaskClassifier (robust multitask classifier)
mtc_model = RobustMultitaskClassifier(n_tasks=len(tasks_ecfp), n_features=1024, layer_sizes=[1000], dropout=0.5)
mtc_model.fit(train_dataset_ecfp, nb_epoch=50)  # Train for 50 epochs

# Evaluate RobustMultitaskClassifier
train_score_mtc = mtc_model.evaluate(train_dataset_ecfp, [roc_auc_score], transformers_ecfp)
test_score_mtc = mtc_model.evaluate(test_dataset_ecfp, [roc_auc_score], transformers_ecfp)
print("RobustMultitaskClassifier AUC-ROC metrics:")
print(f"Training set score: {train_score_mtc}")
print(f"Test set score: {test_score_mtc}")
```

## 📖 Code Explanation
- **Data Loading**: The `dc.molnet.load_tox21()` function loads the Tox21 dataset directly from MoleculeNet. The `featurizer` parameter specifies the molecular featurization method: 'GraphConv' for graph convolutional models and 'ECFP' (Extended-Connectivity Fingerprints) for non-graph models. The dataset is automatically split into training, validation, and test sets.
- **Model Training**: GraphConvModel is suitable for handling molecular graph structures; RobustMultitaskClassifier is used for multitask learning (12 toxicity targets). Training uses 50 epochs, adjustable as needed.
- **Evaluation**: ROC-AUC scores are used to assess model performance, suitable for binary classification problems (toxic/non-toxic).
- **Running Requirements**: Requires installation of DeepChem (pip install deepchem). The Tox21 data will be downloaded automatically if not cached. Expected output includes AUC-ROC scores for training and test sets, e.g., training set scores may exceed 0.9, while test set scores typically range from 0.7 to 0.8 (depending on the random seed).
- **Applications**: This example can be used for molecular toxicity prediction in quantum chemistry or drug discovery, aiding in screening potentially harmful compounds.
