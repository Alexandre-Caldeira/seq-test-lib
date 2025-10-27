# seqtestlib/models/dl.py
"""
Implements the Domain-Adversarial Neural Network (DANN) for ORD detection.
Uses PyTorch for the underlying deep learning framework.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .base import BaseDetector
from .. import config

# --- Core DANN Components (from unified.ipynb) ---

class GradientReversalFunc(torch.autograd.Function):
    """
    Custom autograd function for implementing the Gradient Reversal Layer (GRL).
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient by multiplying with -lambda
        return (grad_output.neg() * ctx.lambda_val), None

class DANN(nn.Module):
    """
    The PyTorch neural network architecture for the DANN.
    """
    def __init__(self, input_dim: int, num_domains: int):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, num_domains)
        )

    def forward(self, x, lambda_val: float = 1.0):
        features = self.feature_extractor(x)
        label_output = self.label_predictor(features)
        rev_features = GradientReversalFunc.apply(features, lambda_val)
        domain_output = self.domain_classifier(rev_features)
        return label_output.squeeze(), domain_output

# --- Library Wrapper for DANN ---

class DANNModel(BaseDetector):
    """
    A detector that wraps the DANN PyTorch model to conform to the BaseDetector API.
    """

    def __init__(self, input_dim: int, num_domains: int, **kwargs):
        """
        Initializes the DANN wrapper.

        Args:
            input_dim (int): The dimensionality of the input features (i.e., m_val).
            num_domains (int): The number of distinct domains (e.g., SNR levels).
        """
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DANN(input_dim=input_dim, num_domains=num_domains).to(self.device)
        self.params.update({'input_dim': input_dim, 'num_domains': num_domains})

    def fit(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray, **kwargs):
        """
        Trains the DANN model.

        Args:
            X (np.ndarray): Training features, shape (n_samples, m_val).
            y (np.ndarray): Class labels (0 or 1).
            domains (np.ndarray): Domain labels for each sample.
            **kwargs: Additional arguments (e.g., epochs, batch_size, lr).
        """
        epochs = kwargs.get('epochs', config.DANN_EPOCHS)
        batch_size = kwargs.get('batch_size', 512)
        lr = kwargs.get('lr', 0.001)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(domains, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_class_fn = nn.BCEWithLogitsLoss()
        loss_domain_fn = nn.CrossEntropyLoss()

        print(f"  Training DANN model on {len(X)} samples (Device: {self.device})...")
        pbar = tqdm(range(epochs), desc=f"    Epoch 1/{epochs}", leave=False)
        for epoch in pbar:
            self.model.train()
            total_loss = 0
            for x_b, y_c_b, y_d_b in loader:
                x_b, y_c_b, y_d_b = x_b.to(self.device), y_c_b.to(self.device), y_d_b.to(self.device)
                
                # Dynamic lambda for GRL
                lambda_val = 2. / (1. + np.exp(-10 * float(epoch) / epochs)) - 1
                
                label_out, domain_out = self.model(x_b, lambda_val)
                
                loss_class = loss_class_fn(label_out, y_c_b)
                loss_domain = loss_domain_fn(domain_out, y_d_b)
                loss = loss_class + loss_domain

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            pbar.set_description(f"    Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(loader):.4f}")
        pbar.close()

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the raw logit score for the positive class (signal present).

        Args:
            X (np.ndarray): An array of feature windows to score.

        Returns:
            np.ndarray: An array of raw logit scores.
        """
        if not self.model:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predicting.")

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
            # The model returns (label_output, domain_output), we only need the label logit
            scores = self.model(x_tensor)[0].detach().cpu().numpy()
        return scores