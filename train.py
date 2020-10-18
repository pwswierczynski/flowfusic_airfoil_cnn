""" Configuration """
TRAINING_DATA_DIR = "../data/train"
VALIDATION_DATA_DIR = "../data/validation"
BATCH_SIZE = 16
NUM_WORKERS = 2
SHUFFLE_DATASET = True
LEARNING_RATE = 0.01
N_EPOCHS = 100
MODEL_DIR = "../models"

import os
import torch

from math import inf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

from networks import UNet
from dataset import SimulationDataset


def compute_loss(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Function evaluating the forward pass and computing the loss

    :param batch: dictionary containing information about domain geometry
        and the target flow.
    :param model: neural network with a forward method.
    :param criterion: pytorch loss function.
    :param device: CPU or GPU depending on availability.

    :return:
    loss_functional: evaluation of the loss functional.
        Contains also information about the functional's gradient.
    """
    model = model.to(device)

    # Reading the source and target data
    geometries, flows = batch["geometry"], batch["flow"]
    geometries = geometries.to(device)
    flows = flows.to(device)

    # Forward and backward pass for optimization
    outputs = model(geometries)
    loss_functional = criterion(outputs, flows)

    return loss_functional


if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define dataset
    training_data = SimulationDataset(base_dir=TRAINING_DATA_DIR)
    validation_data = SimulationDataset(base_dir=VALIDATION_DATA_DIR)

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
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Save checkpoints for inspection with Tensorboard
    writer = SummaryWriter()

    best_loss = inf

    # Start training
    for epoch in range(1, N_EPOCHS + 1):

        # Monitor training and validation loss
        train_loss = 0.0
        validation_loss = 0.0

        print(f"Training epoch: {epoch} out of {N_EPOCHS}")

        for idx_batch, batch in tqdm(enumerate(training_data_loader)):
            # Clear the gradient
            optimizer.zero_grad()

            loss_functional = compute_loss(
                batch=batch, model=model, criterion=criterion, device=device
            )

            loss_value = loss_functional.item() * BATCH_SIZE

            loss_functional.backward()
            optimizer.step()

            # Compute loss
            train_loss += loss_value

        with torch.no_grad():
            print("Measuring validation error")
            for batch in tqdm(validation_data_loader):
                loss_functional = compute_loss(
                    batch=batch, model=model, criterion=criterion, device=device
                )

                loss_value = loss_functional.item() * BATCH_SIZE

                # Compute loss
                validation_loss += loss_value

        # Reduce the learning rate once the training plateaus
        lr_scheduler.step(validation_loss)

        # Training statistics
        train_loss = train_loss / len(training_data)
        validation_loss = validation_loss / len(validation_data)
        writer.add_scalar("training_loss", train_loss, epoch)
        writer.add_scalar("validation_loss", validation_loss, epoch)
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f}")
        print(f"Epoch: {epoch} \tValidation Loss: {validation_loss:.6f}")

        # We always store the best model up to the current epoch
        if validation_loss < best_loss:
            model = model.to("cpu")
            best_loss = validation_loss
            print("Saving the model!")
            path_to_save = os.path.join(MODEL_DIR, f"model_checkpoint.pt")
            torch.save(model.state_dict(), path_to_save)

    writer.add_graph(model.to("cpu"), batch["geometry"].to("cpu"))
    writer.flush()
    writer.close()
