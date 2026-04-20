
# ✅ 一、CNN version cVAE


## Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# =========================
# Hyperparameters
# =========================
latent_dim = 20
num_classes = 10
batch_size = 128
epochs = 10
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data
# =========================
transform = transforms.ToTensor()

train_loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# =========================
# cVAE Model
# =========================
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + num_classes, 32, 4, 2, 1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),               # 14 -> 7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 14 -> 28
            nn.Sigmoid()
        )

    def encode(self, x, y):
        y = self.label_emb(y).unsqueeze(2).unsqueeze(3)
        y = y.expand(-1, -1, 28, 28)

        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y = self.label_emb(y)
        zy = torch.cat([z, y], dim=1)

        h = self.fc_decode(zy)
        h = h.view(-1, 64, 7, 7)

        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


# =========================
# Loss
# =========================
def loss_fn(recon_x, x, mu, logvar):
    recon = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld


# =========================
# Train
# =========================
model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x, y)
        loss = loss_fn(recon, x, mu, logvar)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

# =========================
# Conditional Generation
# =========================
model.eval()
with torch.no_grad():
    n = 10
    z = torch.randn(n, latent_dim).to(device)
    labels = torch.arange(0, 10).to(device)

    samples = model.decode(z, labels)
    save_image(samples, "cvae_digits.png", nrow=10)
```

---

