## Beginner: VAE
## VAE (Variational Autoencoder)  
- **Importance**:  
VAE is a generative extension of the Autoencoder, incorporating probabilistic modeling to generate new data (e.g., images, text).  
It is as renowned as GANs in the field of generative models, suitable for data generation and distribution learning.  
- **Core Concept**:  
VAE maps the encoder's output to a probability distribution (typically a normal distribution), and the decoder samples from this distribution to generate data.  
- **Applications**: Generating art, data augmentation, anomaly detection.

<center><img width="617" height="376" alt="image" src="https://github.com/user-attachments/assets/d8b5e82e-5b83-41d9-8b3c-521a3aeeb38e" /></center>  

## Code （Pytorch）
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

# Set random seed
torch.manual_seed(42)

# Hyperparameters
input_dim = 784  # 28x28 MNIST images
hidden_dim = 400
latent_dim = 20
batch_size = 128
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saving visualization results
if not os.path.exists('results'):
    os.makedirs('results')

# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    # Denormalize target values from [-1, 1] to [0, 1]
    x = (x.view(-1, input_dim) + 1) / 2  # Denormalization
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualization function
def visualize_results(model, test_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)
        
        # Denormalize for display
        data = (data + 1) / 2
        recon_batch = (recon_batch + 1) / 2
        
        # Compare original and reconstructed images
        comparison = torch.cat([data[:8], recon_batch.view(batch_size, 1, 28, 28)[:8]])
        vutils.save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)
        
        # Generate new samples
        sample = torch.randn(64, latent_dim).to(device)
        sample = model.decode(sample).cpu()
        sample = (sample + 1) / 2
        vutils.save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png', nrow=8)

# Initialize model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')
    return avg_train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_test_loss:.4f}')
    return avg_test_loss

# Main training loop
if __name__ == "__main__":
    train_losses = []
    test_losses = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Visualize results
        visualize_results(model, test_loader, epoch, device)
        
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/loss_curve_{epoch}.png')
        plt.close()
```


### 训练结果

  <img width="934" height="478" alt="image" src="https://github.com/user-attachments/assets/ecda78c7-6330-4a20-9a9c-e62bdbd035d7" />    

图一 训练和验证损失曲线    
<img width="286" height="109" alt="image" src="https://github.com/user-attachments/assets/344ba28c-ea33-492c-afd4-7b352fecc93e" />    

图2 原始图像（上）和生成图像（下）的比较   
