""" Configuration """
TRAINING_DATA_DIR = "../data/training"
VALIDATION_DATA_DIR = "../data/validation"
BATCH_SIZE = 16
NUM_WORKERS = 2
SHUFFLE_DATASET = True
LEARNING_RATE = 0.005
N_EPOCHS = 2
MODEL_DIR = "../models"

import os
import torch

from math import inf
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks import UNet
from dataset import VtkDataset

training_data = VtkDataset(base_dir=TRAINING_DATA_DIR)
validation_data = VtkDataset(base_dir=VALIDATION_DATA_DIR)

training_data_loader = DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_DATASET,
    num_workers=NUM_WORKERS,
)

validation_data_loader = DataLoader(
    validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a model and optimization method
model = UNet()
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_loss = inf

for epoch in range(N_EPOCHS):

    # Monitor training and validation loss
    train_loss = 0
    validation_loss = 0

    for idx_batch, batch in tqdm(enumerate(training_data_loader)):

        # Reading the source and target data
        geometries, flows = batch["geometry"], batch["flow"]
        geometries = geometries.to(device)
        flows = flows.to(device)

        # Forward and backward pass for optimization
        optimizer.zero_grad()
        outputs = model(geometries)
        loss = criterion(outputs, flows)
        loss.backward()
        optimizer.step()

        # Compute loss
        train_loss += loss.item() * flows.size(0)

    with torch.no_grad():
        for batch in validation_data_loader:
            # Reading the source and target data
            geometries, flows = batch["geometry"], batch["flow"]
            geometries = geometries.to(device)
            flows = flows.to(device)

            outputs = model(geometries)
            loss = criterion(outputs, flows)

            # Compute loss
            validation_loss += loss.item() * flows.size(0)

    # Training statistics
    train_loss = train_loss / len(training_data)
    print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f}")
    validation_loss = validation_loss / len(validation_data)
    print(f"Epoch: {epoch} \tValidation Loss: {validation_loss:.6f}")

    # We always store the best model up to the current epoch
    if validation_loss < best_loss:
        best_loss = validation_loss
        print("Saving the model!")
        path_to_save = os.path.join(MODEL_DIR, f"model_checkpoint.pt")
        torch.save(model.state_dict(), path_to_save)
