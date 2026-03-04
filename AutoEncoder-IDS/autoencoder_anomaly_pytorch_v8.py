# autoencoder_anomaly_pytorch_v8.py

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --------------------------------
# Note:
# If you want to see more information about any of the classes and functions used,
# just hover over the function/class and PyCharm should show you a description of it.
# The documentation says what it is/does, and what its parameters are. It's very useful.

# Also, remember that you can add print statements to see what any object really is.
# E.g. use `print(normal_data)` to see what the data actually looks like.
# --------------------------------


# =====================================
# 1. Set Random Seeds (Reproducibility)
# =====================================
np.random.seed(42)
torch.manual_seed(42)


# =====================================
# 2. Generate Synthetic Dataset
# =====================================

# Normal data (1000 samples, 10 features)
# This data is randomly generated following a normal distribution (bell curve).
# The mean of the data will be 0 and standard deviation will be 1.
# Note: "normal_data" means the data is not anomalous, not that it comes from a normal distribution.
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10))

# Anomalies (50 samples far from normal cluster)
# This data is randomly generated following a normal distribution (bell curve).
# The mean of the data will be 5 and standard deviation will be 1.5.
anomalous_data = np.random.normal(loc=5, scale=1.5, size=(50, 10))

# Combine dataset
all_data = np.vstack([normal_data, anomalous_data])

# Create labels
# These are for EVALUATION ONLY, NOT TO BE USED DURING TRAINING!
# y_true[i] is the label for X[i]
all_data_labels = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalous_data))])


# =====================================
# 3. Scale the Data
# =====================================

# The Scikit-Learn StandardScaler scales data to a more focused range.
# Ensuring that our model sees inputs in a consistent range will improve accuracy.
# The output is how many standard deviations away from the mean the input is (y = (x - mean) / stddev).
scaler = StandardScaler()

# The scaler should be fit only on the training data (so just the normal data in our case).
# "Fitting" is where we set the mean and std dev to the mean and std dev from the dataset passed in.
scaler.fit(normal_data)

# Scale the input data with our fitted scaler
normal_data_scaled = scaler.transform(normal_data)
all_data_scaled = scaler.transform(all_data)


# =====================================
# 4. Train/Validation Split (Normal Data Only)
# =====================================

# We split our model into a training and testing/validation set:
#     The training set is what we show the model during training
#     The testing set is used to evaluate what the model learned
# We need a separate set for evaluation, because we don't want to test it on the questions we gave it the answers to.
# Just like how your professor isn't going to show you the exam questions before you take it.
train_data, test_data = train_test_split(normal_data_scaled, test_size=0.2)


# =====================================
# 5. Create Autoencoder Model
# =====================================

# We inherit from nn.Module, which is the base PyTorch class for all neural networks
class Autoencoder(nn.Module):

    # The __init__() function is called automatically by Python when we create a new object of this class.
    # Here is where we define the layers of our model, along with any other properties you may want to add.
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Note for the encoder and decoder below:
        # nn.Sequential just groups multiple layers together, so we don't need to save each separately.
        # It will run the input through the defined layers sequentially and output the last layer's output.
        # This isn't strictly necessary-- we could just save each individual layer if we wanted.

        # The encoder architecture is:
        #   input layer (as many neurons as there are in our input data)
        #   intermediate encoder layer (16 neurons)
        #   encoding layer (8 neurons)
        # Each layer has a ReLU activation function.
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU()
        )

        # The decoder architecture is:
        #   input layer (as many neurons as there are in our encoding-- 8)
        #   intermediate decoder layer (16 neurons)
        #   decoding layer (as many neurons as there are in our output (which is the same as our input))
        # We don't use ReLU on the last layer, because that would prevent negative values from being output...
        #   ...but our inputs could contain negative values.
        self.decoder = nn.Sequential(
            nn.Linear(in_features=8, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=input_dim)
        )

    # forward() is the function that actually runs the model. Every PyTorch Module must define this.
    # We just have to pass our input through the encoder, and then pass the encoding through the decoder.
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  # our model output is the decoded reconstruction of the input

input_dim = 10  # number of features input to the model-- the data we generated has 10 features

# Create our model using the class we created above, with our data's input dimension
model = Autoencoder(input_dim)


# =====================================
# 6. Train the Model
# =====================================
model.train()  # set model to train mode (affects some models; best practice is to always use it)

# Training loop parameters
epochs = 50
batch_size = 4  # how many samples will be processed at once-- speeds up training, but can reduce accuracy

# -------------------------------------
# 6.1. Define Loss and Optimizer
# -------------------------------------

# We will use the mean-squared error to evaluate our model.
# MSE is the difference between the prediction and outputs, squared (to make all the values positive).
# MSE = (prediction - target)^2
# For our autoencoder, our inputs/outputs have 10 features. The loss will be calculated separately for each feature.
criterion = nn.MSELoss()

# The optimizer will handle the updating of the model weights (parameters) for us.
# Adam optimizer is the most widely used optimizer.
# The learning rate determines how much the weights will be adjusted each update:
#   A high learning rate will make the model unstable; it could overshoot the best values
#   A low learning rate means that the model will learn very slowly
# Adam will actively update the learning rate during training to learn faster or slower depending on what's best.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------
# 6.2. Prepare Data for Training
# -------------------------------------

# Convert to PyTorch tensor (multidimensional matrices used for processing batches of data)
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)

# Convert to PyTorch dataset for training
# Normally, this dataset would contain the inputs and labels (e.g. TensorDataset(X, y_labels)).
# But since our autoencoder training doesn't use labels, the dataset just contains the inputs.
# We need this for a DataLoader below, which will handle the batching and data shuffling for us.
training_dataset = TensorDataset(train_data_tensor)

# The "shuffle" parameter will shuffle the data between epochs, so every epoch uses a different set of batches.
# This makes training more stable-- otherwise, one unlucky batch could interfere with the training process.
training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=batch_size,
    shuffle=True
)

# -------------------------------------
# 6.3. Training Loop
# -------------------------------------

for epoch in range(epochs):               # for each epoch, we will show the model every item in the dataset
    epoch_loss = 0

    for batch in training_dataloader:     # for each batch of inputs in the training dataset
        batch = batch[0]                  # usually the dataloader returns a list of (x, y), so we need to extract our batch
        optimizer.zero_grad()             # reset gradients from last batch
        outputs = model(batch)            # get model predictions for each item in the batch
        loss = criterion(outputs, batch)  # calculate loss between model's predictions and target outputs (for the autoencoder model, the target output is the input)
        loss.backward()                   # calculate the gradient for each weight (how each weight affects the loss)
        optimizer.step()                  # adjust the weights to improve model accuracy

        epoch_loss += loss.item() * batch.size(0)  # sum over batch

    epoch_loss /= len(training_dataset)  # average over all samples
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")  # print loss so we can track training progress


# =====================================
# 7. Select Reconstruction Error Threshold for Anomaly Detection
# =====================================
model.eval()  # set model to eval mode (affects some models; best practice is to always use it)

# Convert test data to tensor to be passed to the model
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

# ------------------------------------
# 7.1. Calculate Reconstruction Errors for Testing Data
# ------------------------------------

# The reconstruction error threshold should be set on normal data that was NOT seen during training.
# We use the testing set because we want to set the threshold based on how normal, unseen data will be reconstructed.
with torch.no_grad():  # disable unnecessary and costly PyTorch gradient calculations since we only need those for training
    reconstructed_train = model(test_data_tensor)
    mse_test = torch.mean((test_data_tensor - reconstructed_train) ** 2, dim=1)

# Convert tensor to numpy array for functions below
mse_test = mse_test.numpy()

# ------------------------------------
# 7.2. Select Threshold from Testing Data
# ------------------------------------

# The threshold indicates what percentage of normal data will be correctly identified as normal.
# Data with an error in the 95th percentile will be classified as an anomaly.
# In other words, 5% of normal data will be false positives.
threshold = np.percentile(mse_test, 95)
print("\nAnomaly Threshold:", threshold)

# -------------------------------------
# 7.3. Visualize Threshold
# -------------------------------------

# Show the distribution of reconstruction errors for our testing data
plt.hist(mse_test, bins=50)
plt.axvline(threshold, color='red', linestyle='--')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()


# =====================================
# 8. Evaluate Model
# =====================================

# Convert to tensor
all_data_tensor = torch.tensor(all_data_scaled, dtype=torch.float32)

# -------------------------------------
# 8.1. Get Model Predictions For All Data
# -------------------------------------

# Calculate reconstruction errors for all data (normal and anomalous)
with torch.no_grad():
    reconstructed_all = model(all_data_tensor)
    mse_all = torch.mean((all_data_tensor - reconstructed_all) ** 2, dim=1)

# Convert tensor to numpy array
mse_all = mse_all.numpy()

# Determine if each datapoint is an anomaly using our threshold
predictions = mse_all > threshold

print(f"Data points correctly predicted: {(predictions == all_data_labels).sum()}/{len(predictions)}")

# -------------------------------------
# 8.2. Plot Reconstruction Errors
# -------------------------------------
plt.clf()  # clear the previous matplotlib plot

# Small vertical jitter so points don’t overlap perfectly.
# This just makes it easier to see where our points are.
jitter = np.random.normal(0, 0.01, size=len(mse_all))

# Plot the true normal data with blue dots
plt.scatter(
    mse_all[all_data_labels == 0],
    jitter[all_data_labels == 0],
    c='blue',
    alpha=0.6,
    label='True Normal'
)

# Plot the true anomalous data with red dots
plt.scatter(
    mse_all[all_data_labels == 1],
    jitter[all_data_labels == 1],
    c='red',
    alpha=0.8,
    label='True Anomaly'
)

# Plot threshold line
plt.axvline(threshold, color='black', linestyle='--', label='Threshold')

plt.yticks([])  # hide meaningless y-axis
plt.xlabel("Reconstruction Error (MSE)")
plt.title("Reconstruction Errors with Anomaly Classification")
plt.legend()
plt.show()