import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Simulating a dataset with normal and fraudulent transactions
np.random.seed(42)
n_samples = 10000

# Normal transactions (mean: 50, std: 20)
legit_data = np.random.normal(loc=50, scale=20, size=(n_samples, 2))

# Fraudulent transactions (mean: 150, std: 50)
fraud_data = np.random.normal(loc=150, scale=50, size=(int(n_samples * 0.02), 2))

# Combine data and create labels
transactions = np.vstack([legit_data, fraud_data])
labels = np.array([0] * n_samples + [1] * int(n_samples * 0.02))  # 0 = Legit, 1 = Fraud

# Normalize the transactions
scaler = StandardScaler()
transactions_scaled = scaler.fit_transform(transactions)

# Convert to PyTorch tensors
transactions_tensor = torch.tensor(transactions_scaled, dtype=torch.float32)

# Define the Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Model parameters
input_dim = transactions_tensor.shape[1]
hidden_dim = 16
latent_dim = 4

# Initialize the model, optimizer, and loss function
vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Training the VAE on normal transactions only (Unsupervised Anomaly Detection)
epochs = 50
normal_data = transactions_tensor[labels == 0]  # Train only on non-fraudulent transactions

for epoch in range(epochs):
    optimizer.zero_grad()
    reconstructed = vae(normal_data)
    loss = loss_function(reconstructed, normal_data)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Fraud detection using reconstruction error
with torch.no_grad():
    reconstructed = vae(transactions_tensor)
    reconstruction_errors = torch.mean((transactions_tensor - reconstructed) ** 2, dim=1).numpy()

# Set a threshold for fraud detection
threshold = np.percentile(reconstruction_errors[labels == 0], 95)  # 95th percentile of normal transactions
predictions = reconstruction_errors > threshold  # Flag transactions with high reconstruction error as fraud

# Evaluate the model
auc_score = roc_auc_score(labels, predictions)
print(f'Fraud Detection AUC Score: {auc_score:.4f}')
