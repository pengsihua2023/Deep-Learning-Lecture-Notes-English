## Famous network: Diffusion Models(2020s)
## Diffusion Models
- **Proposed by**: Multiple researchers (e.g., DDPM)  
- **Features**: Generates high-quality images through a noise-adding and denoising process, surpassing GANs in performance.  
- **Key Points to Master**: Denoising process, probabilistic modeling.  
- **Importance**:  
Diffusion models are the latest technology for generating high-quality images, powering generative AI systems like DALL·E 2 and Stable Diffusion.  
They have surpassed GANs in fields such as image generation and text-to-image, becoming the new benchmark for generative models.  
- **Core Concept**:  
Diffusion models learn data distributions through a "noise-adding and denoising" process, first corrupting data into random noise and then progressively reconstructing it.  
- **Applications**: Image generation (art, game design), video generation, scientific simulations.  
<img width="2060" height="920" alt="image" src="https://github.com/user-attachments/assets/427d35b9-10d1-4bca-b74c-b5e166d7613d" />

## Code description
This code implements a simple **Denoising Diffusion Probabilistic Model (DDPM)** using the PyTorch library, with the following main functionalities:  

1. **Data Generation**: Generates 2D normal distribution data (mean [2, 2], standard deviation 0.5) as training and visualization samples.  
2. **Forward Diffusion**: Gradually adds Gaussian noise (over 1000 steps with linear variance scheduling) to transform original data into pure noise.  
3. **Denoising Model**: Uses a simple multilayer perceptron (MLP) to predict noise at each step, taking noisy data with timestep information as input and outputting predicted noise.  
4. **Training**: Trains the model by minimizing the mean squared error between predicted and actual noise (1000 epochs, Adam optimizer).  
5. **Sampling**: Starts from pure noise and generates samples resembling the original data distribution through a reverse denoising process.  
6. **Visualization**: Uses Matplotlib to plot 2D scatter plots of original and generated data, comparing their distributions.  

The code supports CPU/GPU execution, ensures device consistency, and outputs training loss and the shape of generated samples. The final scatter plot visually demonstrates whether the model successfully learned the data distribution.
## Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
num_steps = 1000  # Number of diffusion steps
beta_start = 0.0001  # Starting value for variance schedule
beta_end = 0.02  # Ending value for variance schedule
data_dim = 2  # Data dimension (2D normal distribution)
batch_size = 128
epochs = 1000
lr = 0.001
n_samples = 1000  # Number of samples for visualization

# Linear variance schedule
betas = torch.linspace(beta_start, beta_end, num_steps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Forward diffusion process
def forward_diffusion(x_0, t, device):
    betas_t = betas.to(device)
    alphas_cumprod_t = alphas_cumprod.to(device)
    noise = torch.randn_like(x_0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_t[t]).view(-1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod_t[t]).view(-1, 1)
    return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise, noise

# Simple MLP model
class SimpleDenoiser(nn.Module):
    def __init__(self):
        super(SimpleDenoiser, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim)
        )
    
    def forward(self, x, t):
        t = t.view(-1, 1).float() / num_steps
        x_t = torch.cat([x, t], dim=1)
        return self.model(x_t)

# Generate simple dataset (2D normal distribution)
def generate_data(n_samples):
    return torch.randn(n_samples, data_dim) * 0.5 + torch.tensor([2.0, 2.0])

# Train model
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move global tensors to device
    global betas, alphas, alphas_cumprod
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    
    model = SimpleDenoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        data = generate_data(batch_size).to(device)
        t = torch.randint(0, num_steps, (batch_size,), device=device)
        
        x_t, noise = forward_diffusion(data, t, device)
        predicted_noise = model(x_t, t)
        loss = nn.MSELoss()(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    return model

# Sampling process
def sample(model, n_samples, device):
    betas_t = betas.to(device)
    alphas_t = alphas.to(device)
    alphas_cumprod_t = alphas_cumprod.to(device)
    
    x = torch.randn(n_samples, data_dim).to(device)
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)
        predicted_noise = model(x, t_tensor)
        alpha = alphas_t[t]
        alpha_cumprod = alphas_cumprod_t[t]
        beta = betas_t[t]
        
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
        if t > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)
    return x

# Visualization function
def visualize_samples(original_data, generated_samples):
    plt.figure(figsize=(10, 5))
    
    # Scatter plot of original data
    plt.subplot(1, 2, 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, label="Original Data")
    plt.title("Original Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    # Scatter plot of generated samples
    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, color="orange", label="Generated Samples")
    plt.title("Generated Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    model = train()
    
    # Generate original data and generated samples for visualization
    original_data = generate_data(n_samples).cpu().numpy()
    generated_samples = sample(model, n_samples, device).cpu().detach().numpy()
    
    # Visualize results
    visualize_samples(original_data, generated_samples)
    print("Generated samples shape:", generated_samples.shape)
```


## Training Results

Epoch 700, Loss: 0.1903  
Epoch 800, Loss: 0.1887  
Epoch 900, Loss: 0.2179  
Epoch 1000, Loss: 0.3247  
Generated samples shape: (1000, 2)  

<img width="984" height="493" alt="image" src="https://github.com/user-attachments/assets/b57caf73-74d1-41a4-b547-ff59cd9670a8" />

Figure 2: Comparison of Original and Generated Samples  

