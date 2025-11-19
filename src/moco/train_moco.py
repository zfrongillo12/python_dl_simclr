import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import time

# Writing a function to train the MoCo model for one epoch
def train_one_epoch_moco(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    positive_logit_values = []
    negative_logit_values = []

    for batch in data_loader:
        images_query, images_key, _ = batch
        images_query = images_query.to(device)
        images_key = images_key.to(device)

        logits, labels = model(images_query, images_key)

        # Computing the loss using cross entropy
        loss = loss_fn(logits, labels)

        # Performing the backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        positives = logits[:, 0].detach().cpu()
        negatives = logits[:, 1:].mean(dim=1).detach().cpu()

        positive_logit_values.extend(positives.tolist())
        negative_logit_values.extend(negatives.tolist())

    avg_pos = sum(positive_logit_values) / len(positive_logit_values)
    avg_neg = sum(negative_logit_values) / len(negative_logit_values)

    return total_loss / len(data_loader), avg_pos, avg_neg


# Writing a full MoCo training loop with epochs
def run_moco_training(model, data_loader, optimizer, device, epochs, save_path=None, csv_log_path=None):
    loss_history = []

    if csv_log_path is not None:
        with open(csv_log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "epoch_loss",
                "learning_rate",
                "queue_mean",
                "queue_std",
                "avg_positive_logit",
                "avg_negative_logit",
                "epoch_seconds"
            ])

    for epoch in range(epochs):
        start_time = time.time()

        epoch_loss, avg_pos, avg_neg = train_one_epoch_moco(model, data_loader, optimizer, device)
        loss_history.append(epoch_loss)

        # Getting the learning rate for this epoch
        lr = optimizer.param_groups[0]["lr"]

        # Getting the queue statistics
        queue = model.queue.detach().cpu()
        queue_mean = queue.mean().item()
        queue_std = queue.std().item()

        epoch_seconds = time.time() - start_time

        # Time calculations
        elapsed = time.time() - start_time
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = remaining_epochs * elapsed

        # ETA formatting
        eta_min = eta_seconds / 60
        eta_hr  = eta_min / 60

        if eta_hr >= 1:
            eta_str = f"{eta_hr:.1f}h"
        else:
            eta_str = f"{eta_min:.1f}m"

        # Printing out Epoch Statistics in Notebook
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Pos: {avg_pos:.3f} | "
            f"Neg: {avg_neg:.3f} | "
            f"Queue μ: {queue_mean:.3f} | "
            f"Queue σ: {queue_std:.3f} | "
            f"Time: {elapsed:.1f}s | ETA: {eta_str}"
        )

        if save_path is not None:
            checkpoint_path = f"{save_path}/moco_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)

        if csv_log_path is not None:
            with open(csv_log_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch+1,
                    epoch_loss,
                    lr,
                    queue_mean,
                    queue_std,
                    avg_pos,
                    avg_neg,
                    epoch_seconds
                ])

    return loss_history
