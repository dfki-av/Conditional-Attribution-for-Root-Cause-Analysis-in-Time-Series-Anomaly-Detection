import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np

class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class BaseVariationalAutoencoder(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()

    def fit_on_data(self, train_data, val_data=None, max_epochs=1000, verbose=0, device=None):
        device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(device)
        
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if val_data is not None:
            val_tensor = torch.FloatTensor(val_data).to(device)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        optimizer = optim.Adam(self.parameters())
        
        for epoch in range(max_epochs):
            self.train()
            total_loss = 0
            reconstruction_loss = 0
            kl_loss = 0
            
            for batch in train_loader:
                X = batch[0]
                optimizer.zero_grad()
                
                z_mean, z_log_var, z = self.encoder(X)
                reconstruction = self.decoder(z)
                
                loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                
                # Normalize the loss by the batch size
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                kl_loss += kl.item()

            if verbose:
                print(f"Epoch {epoch + 1}/{max_epochs} | Total loss: {total_loss / len(train_loader):.4f} | "
                    f"Recon loss: {reconstruction_loss / len(train_loader):.4f} | "
                    f"KL loss: {kl_loss / len(train_loader):.4f}")

            if val_loader is not None:
                self.eval()
                val_total_loss = 0
                val_reconstruction_loss = 0
                val_kl_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        X = batch[0]
                        z_mean, z_log_var, z = self.encoder(X)
                        reconstruction = self.decoder(z)
                        loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                        
                        # Normalize the loss by the batch size
                        loss = loss / X.size(0)
                        recon_loss = recon_loss / X.size(0)
                        kl = kl / X.size(0)

                        val_total_loss += loss.item()
                        val_reconstruction_loss += recon_loss.item()
                        val_kl_loss += kl.item()

            else:
                val_total_loss += None
                val_reconstruction_loss += None
                val_kl_loss += None

            if verbose:
                if val_total_loss is not None:
                    print(f"\t Val Total loss: {val_total_loss / len(val_loader):.4f} | "
                        f"Val Recon loss: {val_reconstruction_loss / len(val_loader):.4f} | "
                        f"Val KL loss: {val_kl_loss / len(val_loader):.4f}")

    def evaluate_dataset(self, data, batch_size=None):
        """
        Evaluate the model over the full dataset without training.
        
        Args:
            data (numpy array): Full dataset to evaluate.
            batch_size (int or None): Batch size for evaluation. 
                                    If None, uses model.batch_size.
                                    
        Returns:
            dict: Average total_loss, recon_loss, kl_loss over dataset.
        """
        self.eval()
        device = next(self.parameters()).device

        if batch_size is None:
            batch_size = self.batch_size

        # Create loader
        data_tensor = torch.FloatTensor(data).to(device)
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                X = batch[0]
                n = X.size(0)

                z_mean, z_log_var, z = self.encoder(X)
                recon = self.decoder(z_mean)

                loss, recon_loss, kl_loss = self.loss_function(X, recon, z_mean, z_log_var)

                # Sum the batch losses (will normalize later)
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                total_samples += n

        # Normalize by number of samples
        return {
            "total_loss": total_loss / total_samples,
            "recon_loss": total_recon_loss / total_samples,
            "kl_loss": total_kl_loss / total_samples
        }

    def evaluate_dataset_windows(self, data, batch_size=None):
        """
        Evaluate the model over the full dataset without training.

        Args:
            data (numpy array): Dataset with shape (N, T, D).
            batch_size (int or None): Batch size for evaluation.

        Returns:
            dict containing:
                total_loss: Average total loss
                recon_loss: Average reconstruction loss
                kl_loss: Average KL loss
                scores: Reconstruction score for each window
        """
        self.eval()
        device = next(self.parameters()).device

        if batch_size is None:
            batch_size = self.batch_size

        data_tensor = torch.FloatTensor(data)
        dataset = TensorDataset(data_tensor)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0

        all_scores = []

        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(device)
                n = X.size(0)

                z_mean, z_log_var, z = self.encoder(X)

                # Deterministic reconstruction
                recon = self.decoder(z_mean)

                loss, recon_loss, kl_loss = self.loss_function(
                    X,
                    recon,
                    z_mean,
                    z_log_var
                )

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_samples += n

                # One reconstruction-error score per window
                window_scores = torch.sum(
                    (X - recon) ** 2,
                    dim=tuple(range(1, X.ndim))
                )

                all_scores.append(window_scores.cpu())

        all_scores = torch.cat(all_scores).numpy()

        return {
            "total_loss": total_loss / total_samples,
            "recon_loss": total_recon_loss / total_samples,
            "kl_loss": total_kl_loss / total_samples,
            "scores": all_scores
        }

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        return x_decoded
    
    def predict(self, X, return_cpu=False):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if torch.is_tensor(X):
                X = X.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                X = torch.as_tensor(X, dtype=torch.float32, device=device)
            z_mean, z_log_var, z = self.encoder(X)
            x_decoded = self.decoder(z_mean)
        if return_cpu:
            return x_decoded.cpu().detach().numpy()
        return x_decoded.detach()

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    def get_latent_mean(self, X):
        """
        Encode a batch into latent means (z_mean).
        Returns a tensor on the same device as the model.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=device)
            z_mean, _, _ = self.encoder(X)
        return z_mean

    def encode_dataset_means(self, data, batch_size=None):
        """
        Encode an entire dataset into latent space using z_mean.
        Returns a single tensor (N, latent_dim) on the same device as the model.
        """
        self.eval()
        device = next(self.parameters()).device
        if batch_size is None:
            batch_size = self.batch_size
        data_tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        latents = []
        with torch.no_grad():
            for (X,) in loader:
                z_mean, _, _ = self.encoder(X)
                latents.append(z_mean)
        return torch.cat(latents, dim=0)


    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        err = torch.pow(X - X_recons, 2)
        reconst_loss = torch.sum(err)
        
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis

        return reconst_loss

    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_name = getattr(self, "checkpoint_name", self.model_name)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, f"{checkpoint_name}_encoder_wts.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(model_dir, f"{checkpoint_name}_decoder_wts.pth"))

    def load_weights(self, model_dir):
        checkpoint_name = getattr(self, "checkpoint_name", self.model_name)
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, f"{checkpoint_name}_encoder_wts.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{checkpoint_name}_decoder_wts.pth")))

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes) if hasattr(self, 'hidden_layer_sizes') else None,
        }
        checkpoint_name = getattr(self, "checkpoint_name", self.model_name)
        params_file = os.path.join(model_dir, f"{checkpoint_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)



class VAEWrapperTorch:
    """
    Wrapper for PyTorch VAE that outputs smooth probabilities using a sigmoid
    centered at the threshold.
    """

    def __init__(self, vae_model, threshold, sharpness=5.0):
        self.vae = vae_model
        self.threshold = torch.tensor(threshold, dtype=torch.float32)
        self.sharpness = torch.tensor(sharpness, dtype=torch.float32)    # controls how steep the sigmoid is
        self.device = next(vae_model.parameters()).device

    def _reconstruction_loss_per_sample(self, X):
        self.vae.eval()
        with torch.no_grad():
            if X.dim() == 4:
                X = X.squeeze(0)
            X = X.to(self.device)

            z_mean, z_log_var, z = self.vae.encoder(X)
            recon = self.vae.decoder(z_mean)

            # main loss
            err = torch.pow(X - recon, 2)
            main_loss = torch.sum(err, dim=[1, 2])

            # time-axis loss
            x_r = torch.mean(X, dim=2)
            x_c_r = torch.mean(recon, dim=2)
            err_t = torch.pow(x_r - x_c_r, 2)
            time_loss = torch.sum(err_t, dim=1)

            total_loss = main_loss + time_loss

        return total_loss.cpu()


    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x_t = torch.tensor(x, dtype=torch.float32)
        else:
            x_t = x

        losses = self._reconstruction_loss_per_sample(x_t.unsqueeze(0))
        
        # compute smooth probability using sigmoid
        logits = self.sharpness * (losses - self.threshold)
        logits = logits.to(dtype=torch.float32, device=self.device)
        p_anomaly = 1.0 / (1.0 + torch.exp(-logits))
        p_normal = 1.0 - p_anomaly
 
        # shape: [batch, 2]
        probs = torch.stack([p_normal, p_anomaly], axis=1)

        return probs
