"""Reconstruction-score ROC evaluation."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


def calculate_reconstruction_roc_auc(vae_model, normal_data, anomaly_data, batch_size=None):
    """
    Calculate ROC-AUC using EXACTLY the same reconstruction loss as the VAE.
    """

    device = next(vae_model.parameters()).device

    if batch_size is None:
        batch_size = vae_model.batch_size

    # Combine data + labels
    mixed_data = np.concatenate([normal_data, anomaly_data], axis=0)
    labels = np.concatenate([
        np.zeros(len(normal_data)),
        np.ones(len(anomaly_data))
    ])

    # DataLoader
    tensor = torch.FloatTensor(mixed_data).to(device)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    anomaly_scores = []

    # ---- SAME RECONSTRUCTION LOSS LOGIC AS YOUR VAE ----
    def reconstruction_loss_exact(X, X_recons):
        # main SSE component
        err = (X - X_recons) ** 2
        reconst_loss = torch.sum(err, dim=(1, 2))

        # penalty along dim=2 (time axis)
        x_r = torch.mean(X, dim=2)
        x_c_r = torch.mean(X_recons, dim=2)
        axis_err = (x_r - x_c_r) ** 2
        axis_loss = torch.sum(axis_err, dim=1)
        
        return reconst_loss + axis_loss

    # -----------------------------------------------------

    vae_model.eval()
    with torch.no_grad():
        for batch in loader:
            X = batch[0]

            # Forward pass
            z_mean, z_log_var, z = vae_model.encoder(X)
            recon = vae_model.decoder(z_mean)

            # Per-sample anomaly score (your EXACT recon loss)
            loss_per_sample = reconstruction_loss_exact(X, recon)

            anomaly_scores.extend(loss_per_sample.cpu().numpy())

    anomaly_scores = np.array(anomaly_scores)

    roc_auc = roc_auc_score(labels, anomaly_scores)
    return roc_auc, anomaly_scores, labels


def save_roc_curve(anomaly_scores, labels, save_dir, scenario_name="mixed_dataset"):
        """
        Plot the ROC curve for anomaly detection and save the figure.

        Args:
            anomaly_scores (np.ndarray): Per-sample anomaly scores.
            labels (np.ndarray): Ground truth labels (0=normal, 1=anomaly).
            save_dir (str): Directory where the ROC plot will be saved.
            scenario_name (str): Prefix for the saved file.
        """

        # ---- Compute ROC curve and AUC ----
        fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)

        # ---- Create plot ----
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({scenario_name})')
        plt.legend(loc='lower right')

        # ---- Annotate best threshold (Youden's J statistic) ----
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', label=f'Best Threshold = {best_threshold:.4f}')
        plt.legend(loc='lower right')

        # ---- Save the figure ----
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{scenario_name}_roc_curve.png")
        plt.savefig(save_path)
        plt.close()

        print(f"ROC curve saved at: {save_path}")
        return roc_auc, best_threshold
