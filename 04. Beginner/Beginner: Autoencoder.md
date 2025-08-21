Autoencoder
Autoencoder

Importance:An Autoencoder is an unsupervised learning model used for data compression, denoising, or feature learning.It is a precursor to generative models (e.g., GANs) and is widely used in data preprocessing and anomaly detection.
Core Concept:An Autoencoder consists of an encoder (compresses data) and a decoder (reconstructs data), with the goal of making the output as close as possible to the input.Analogy: Like a "data compressor," it compresses a large file and then decompresses it, striving to preserve the original.
Applications: Image denoising, data compression, anomaly detection (e.g., credit card fraud detection).
<img width="1400" height="797" alt="image" src="https://github.com/user-attachments/assets/44c3b2df-579f-418e-ad1c-8d77ba3a7621" />  
  

Figure 1: First Representation  
  
<img width="700" height="220" alt="image" src="https://github.com/user-attachments/assets/1a227a27-c26d-4f11-965d-44f95cbfea61" />  
  
Figure 2: Second Representation  
<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/f6e08480-a20a-488d-af52-be84c7c6a392" />  

Figure 3: Third Representation

Mathematical Description of Autoencoder

Basic StructureAn Autoencoder consists of an Encoder and a Decoder:  
<img width="1072" height="144" alt="image" src="https://github.com/user-attachments/assets/18448061-c597-4236-a8f5-979e693e55ae" />  
- **Condition 1**: Let \( x \in \mathbb{R}^d \) be the projection of the latent representation \( z \in \mathbb{R}^m \) (where \( m < d \)).
- **Condition 2**: Let \( z \) be the conditional mean \( \hat{z} \in \mathbb{R}^d \), and \( \hat{z} \sim x \).


2. Mathematical Expression  
<img width="1022" height="465" alt="image" src="https://github.com/user-attachments/assets/fcb07a67-3615-413f-8aa9-1a04e72193b3" />

3. Parameterization  
<img width="1030" height="380" alt="image" src="https://github.com/user-attachments/assets/9a72c301-261b-4b41-92e6-93dc805d2661" />

4. Regularization Variants  
<img width="1275" height="515" alt="image" src="https://github.com/user-attachments/assets/b84a7c3e-8710-49ed-a661-89dcc7b2622c" />

5. Optimization  
Optimized via backpropagation:  
<img width="633" height="150" alt="image" src="https://github.com/user-attachments/assets/58822023-69b6-4c43-a835-dd494d2c7b23" />

6. Applications  
<img width="693" height="234" alt="image" src="https://github.com/user-attachments/assets/d11696c3-35b2-4290-a5ff-784f808f67fe" />  
  

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

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Loss function (using only reconstruction loss)
def loss_function(recon_x, x):
    # Denormalize target values from [-1, 1] to [0, 1]
    x = (x.view(-1, input_dim) + 1) / 2
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE

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
        recon_batch = model(data)
        
        # Denormalize for display
        data = (data + 1) / 2
        recon_batch = (recon_batch + 1) / 2
        
        # Compare original and reconstructed images
        comparison = torch.cat([data[:8], recon_batch.view(batch_size, 1, 28, 28)[:8]])
        vutils.save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)

# Initialize model and optimizer
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
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
            recon_batch = model(data)
            test_loss += loss_function(recon_batch, data).item()
    
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
        plt.plot(test_losses, --label='Test Loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/loss_curve_{epoch}.png')
        plt.close()
```
## Results
====> Epoch: 9 Average training loss: 69.1569====> Test set loss: 69.0569Train Epoch: 10 [0/60000 (0%)] Loss: 71.628830Train Epoch: 10 [12800/60000 (21%)] Loss: 65.910645Train Epoch: 10 [25600/60000 (43%)] Loss: 68.564079Train Epoch: 10 [38400/60000 (64%)] Loss: 70.579895Train Epoch: 10 [51200/60000 (85%)] Loss: 69.532722====> Epoch: 10 Average training loss: 68.6832====> Test set loss: 68.4474Figure 4: Loss CurveFigure 5: Comparison of Input and 
  
<img width="960" height="490" alt="image" src="https://github.com/user-attachments/assets/2eac166e-59fe-4528-bb38-4eb4c7cf8c87" />  

