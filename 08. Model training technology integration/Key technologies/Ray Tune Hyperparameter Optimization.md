# Ray Tune Hyperparameter Optimization Method
## 📖 What is the Ray Tune Hyperparameter Optimization Method?
**Ray Tune** is an open-source Python library for hyperparameter tuning and experiment management, part of the Ray ecosystem for distributed computing. It enables efficient searching of hyperparameter combinations to optimize machine learning models, supporting scalable execution from single machines to large clusters. Ray Tune integrates with various search algorithms (e.g., grid search, random search, Bayesian optimization via libraries like HyperOpt or Optuna) and is particularly suited for distributed hyperparameter tuning in deep learning and ML tasks.

## 📖 Core Features:
1. **Automated Search**: Define hyperparameter spaces (e.g., ranges for learning rate or layer sizes), and Ray Tune automatically explores them using specified algorithms.
2. **Distributed Execution**: Runs trials in parallel across CPUs/GPUs or clusters, making it ideal for large-scale tuning.
3. **Integration with Algorithms**: Supports advanced optimizers like Population Based Training (PBT), ASHA (Asynchronous Successive Halving Algorithm) for early stopping, and integration with tools like Optuna or Ax.
4. **Fault Tolerance and Logging**: Handles failures gracefully, logs results (e.g., via TensorBoard), and supports checkpoints.
5. **Easy Framework Integration**: Works seamlessly with PyTorch, TensorFlow, XGBoost, and more.

## 📖 Advantages:
- Scales to distributed environments, speeding up tuning for complex models.
- More efficient than manual or basic searches, with early stopping to save resources.
- Flexible for experiment tracking and reproducibility.

## 📖 Challenges:
- Requires installing Ray, which adds overhead for simple, non-distributed tasks.
- Learning curve for distributed setup; may be overkill for small experiments.


## 📖 Principles of Ray Tune
1. **Define Trainable Function**:
   - Create a function that takes a configuration (hyperparameters) and performs training, reporting metrics (e.g., accuracy) via `tune.report()`.

2. **Hyperparameter Space**:
   - Specify search spaces using `tune` utilities like `tune.uniform()`, `tune.loguniform()`, or `tune.choice()`.

3. **Search Algorithms and Schedulers**:
   - Use algorithms for sampling (e.g., random, grid) or optimization (e.g., Bayesian).
   - Schedulers like ASHA prune poor-performing trials early based on intermediate metrics.

4. **Tuning Process**:
   - A `Tuner` object runs multiple trials, potentially in parallel, and returns results with the best configuration.

---

## 📖 Simple Code Example: Hyperparameter Optimization with PyTorch and Ray Tune
Below is a simple example using Ray Tune to optimize hyperparameters (learning rate and hidden layer size) for a PyTorch model on the MNIST dataset. This demonstrates distributed tuning basics (assumes Ray is installed via `pip install "ray[tune]"`).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# 1. Define the Model
class SimpleNet(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. Define the Trainable Function (Ray Tune calls this for each trial)
def train_mnist(config):
    # Data loading
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST('.', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Model, optimizer, and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(config["hidden_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Training loop (2 epochs as example)
    for epoch in range(2):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        tune.report(accuracy=accuracy)  # Report metric for tuning and pruning

# 3. Initialize Ray and Set Up Tuning
ray.init(ignore_reinit_error=True)  # Initialize Ray (for local use)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-1),  # Learning rate search space
    "hidden_size": tune.choice([64, 128, 256])  # Hidden size options
}

scheduler = ASHAScheduler(metric="accuracy", mode="max")  # Early stopping scheduler

tuner = tune.Tuner(
    train_mnist,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=6,  # Number of trials (combinations to try)
        scheduler=scheduler
    )
)

results = tuner.fit()  # Run the tuning

# 4. Get Best Results
best_result = results.get_best_result(metric="accuracy", mode="max")
print("Best config:", best_result.config)
print(f"Best accuracy: {best_result.metrics['accuracy']:.4f}")
```



## 📖 Code Explanation
1. **Trainable Function (`train_mnist`)**:
   - Takes a `config` dict with hyperparameters (e.g., `lr`, `hidden_size`).
   - Trains a simple network and reports validation accuracy using `tune.report()` for optimization and pruning.

2. **Search Space**:
   - Defines ranges: logarithmic for learning rate, discrete choices for hidden size.

3. **Tuner Setup**:
   - Uses `Tuner` to run trials, with ASHA scheduler for early termination of poor trials.
   - `num_samples=6` runs 6 hyperparameter combinations (adjust for more exploration).

4. **Results**:
   - After fitting, retrieve the best configuration and its metric.



## 📖 Key Points
1. **Scalability**: By default local, but add `resources_per_trial={"cpu": 1, "gpu": 1}` to `TuneConfig` for distributed runs on clusters.
2. **Pruning**: ASHA scheduler stops underperforming trials based on reported metrics.
3. **Extensibility**: Integrate with Optuna by using `OptunaSearch` in Tune, or combine with Curriculum Learning/AMP by modifying the trainable function (e.g., add `torch.cuda.amp` for mixed precision).



### 📖 Practical Effects
- **Efficiency**: Ray Tune can parallelize trials, reducing tuning time (e.g., from hours to minutes on multi-GPU setups).
- **Flexibility**: Handles complex spaces and integrates with advanced algorithms for better results than random search.
- **Results**: In this example, it might find an optimal learning rate and hidden size in 6 trials, improving accuracy over default values.
