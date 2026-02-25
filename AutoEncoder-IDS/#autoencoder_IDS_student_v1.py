import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------------
# 1. Set Random Seeds (Reproducibility)
# --------------------------------
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------
# 2. Generate Synthetic Dataset
# ---------------------------------

# Normal data (1000 samples, 10 features)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10))

# Anomalies (50 samples far from normal cluster)
anomalies = np.random.normal(loc=5, scale=1.5, size=(50, 10))

# Combine dataset
X = np.vstack([normal_data, anomalies])

# Labels (for evaluation only, not used in training)
y_true = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalies))])

# -------------------------------------
# 3. Scale the Data
# -------------------------------------
scaler = StandardScaler()
normal_data_scaled = scaler.fit_transform(normal_data)
X_scaled = scaler.transform(X)

# Convert to PyTorch tensors
normal_tensor = torch.tensor(normal_data_scaled, dtype=torch.float32)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ---------------------------------------
# 4. Train/Validation Split (Normal Data Only)
# ---------------------------------------
X_train, X_val = train_test_split(normal_tensor, test_size=0.2)

# ---------------------------------------
# 5. Define Autoencoder Model
# ---------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = normal_tensor.shape[1]
model = Autoencoder(input_dim)

# -------------------------------------
# 6. Define Loss and Optimizer
# -------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------
# 7. Training Loop
# -------------------------------------
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for index in range(0, X_train.size(0), batch_size):  # start at 0
        indices = permutation[index:index + batch_size]
        batch = X_train[indices]

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch.size(0)  # sum over batch

    epoch_loss /= X_train.size(0)  # average over all samples
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

# ------------------------------------
# 8. Compute Reconstruction Error
# ------------------------------------
model.eval()

with torch.no_grad():
    reconstructed_train = model(normal_tensor)
    mse_train = torch.mean((normal_tensor - reconstructed_train) ** 2, dim=1)

mse_train = mse_train.numpy()

threshold = np.percentile(mse_train, 95)
print("\nAnomaly Threshold:", threshold)

# -------------------------------------
# 10. Detect Anomalies
# -------------------------------------
predictions = mse_train > threshold

print("Total anomalies detected:", np.sum(predictions))
print("Actual anomalies:", np.sum(y_true))

# -------------------------------------
# 11. Visulization
# -------------------------------------
plt.hist(mse_train, bins=50)
plt.axvline(threshold, color='red', linestyle='--')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# ================
# 12. Vis 2
# ================
# -------------------------------------
# Compute Reconstruction Error (All Data)
# -------------------------------------
with torch.no_grad():
    reconstructed_all = model(X_tensor)
    mse_all = torch.mean((X_tensor - reconstructed_all) ** 2, dim=1)

mse_all = mse_all.numpy()

# Predictions
predictions = mse_all > threshold

# -------------------------------------
# Dot Plot of Reconstruction Errors
# -------------------------------------
plt.clf()

# Small vertical jitter so points don’t overlap perfectly
jitter = np.random.normal(0, 0.01, size=len(mse_all))

plt.scatter(
    mse_all[y_true == 0],
    jitter[y_true == 0],
    c='blue',
    alpha=0.6,
    label='True Normal'
)

plt.scatter(
    mse_all[y_true == 1],
    jitter[y_true == 1],
    c='red',
    alpha=0.8,
    label='True Anomaly'
)

# Threshold line
plt.axvline(threshold, color='black', linestyle='--', label='Threshold')

plt.yticks([])  # hide meaningless y-axis
plt.xlabel("Reconstruction Error (MSE)")
plt.title("Reconstruction Errors with Anomaly Classification")
plt.legend()
plt.show()
