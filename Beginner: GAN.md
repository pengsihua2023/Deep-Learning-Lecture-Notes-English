## Beginner: GAN
Generative Adversarial Network (GAN)  
- Importance: GAN is a flagship generative model, showcasing the creativity of deep learning and ideal for sparking middle school students' interest in AI.  
- Core Concept:  
GAN consists of two networks: a generator (creates fake images) and a discriminator (distinguishes real from fake), which "compete" to learn.  
The generator ultimately produces realistic data (e.g., images, music).  
- Applications: Generating artwork, restoring old photos, designing game characters.  
 <img width="1213" height="529" alt="image" src="https://github.com/user-attachments/assets/8cca8f1f-73ad-4df2-9b66-55353dd7b7c8" />

This image appears to illustrate the concept of a Generative Adversarial Network (GAN), a machine learning model. It consists of two main components:  
1. **Generator**: Represented in cyan on the left, the generator is responsible for creating data samples (e.g., the blocks in the image). Its role is to produce fake data that mimics real data.  
2. **Discriminator**: Represented in red on the right, the discriminator evaluates the received data, determining whether it is "real" (authentic data) or "fake" (generated data). The green labels "real" and "fake" indicate the discriminator's classification results.  
In a GAN, the generator and discriminator are trained simultaneously through a competitive process: the generator improves by attempting to deceive the discriminator, while the discriminator improves by better distinguishing real data from fake data. This iterative process continues until the generator can produce highly realistic data.  

## Code(Pytorch)

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)

# 1. Parameter settings
z_dim = 100  # Noise input dimension
image_dim = 28 * 28  # MNIST image size (28x28)
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5  # Beta1 parameter for Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. Define the generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.ReLU(True),
            nn.Linear(1024, image_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

# 4. Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 5. Initialize model, loss function, and optimizer
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# 6. Train the GAN
def train_gan():
    real_label = 1.0
    fake_label = 0.0
    
    for epoch in range(num_epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        num_batches = 0
        
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train the discriminator
            discriminator.zero_grad()
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            d_real_output = discriminator(real_images)
            d_real_loss = criterion(d_real_output, real_labels)
            
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)
            d_fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train the generator
            generator.zero_grad()
            g_output = discriminator(fake_images)
            g_loss = criterion(g_output, real_labels)  # Hope the generator's output is classified as real
            g_loss.backward()
            g_optimizer.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            num_batches += 1
        
        # Print loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'D Loss: {d_loss_total/num_batches:.4f}, '
                  f'G Loss: {g_loss_total/num_batches:.4f}')
        
        # Save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = generator(torch.randn(16, z_dim).to(device)).cpu()
                save_images(fake_images, epoch + 1)

# 7. Save generated images
def save_images(images, epoch):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    images = images * 0.5 + 0.5  # Denormalize to [0, 1]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.close()

# 8. Visualize results
def plot_final_results():
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_images = generator(z).cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.title('Generated MNIST Digits')
    plt.show()

# 9. Execute training and visualization
if __name__ == "__main__":
    print("Training started...")
    train_gan()
    print("\nGenerating final visualization...")
    plot_final_results()
```

## Training results
Epoch [45/50], D Loss: 0.3414, G Loss: 2.7712   
Epoch [50/50], D Loss: 0.3680, G Loss: 2.7854   

Generating final visualization...   
<img width="741" height="708" alt="image" src="https://github.com/user-attachments/assets/1787c031-5bf5-4ef9-b734-ef7e6ef65b3b" />
