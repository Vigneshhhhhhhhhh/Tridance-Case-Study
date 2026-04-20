"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
Tredence AI Engineering Internship - Case Study Solution

This script implements a feed-forward neural network that learns to prune
itself during training using learnable gate parameters and L1 sparsity
regularization. The network is trained and evaluated on CIFAR-10.

Architecture:
    - PrunableLinear layers with sigmoid-gated weights
    - L1 sparsity loss to encourage gates to collapse to zero
    - Training loop with configurable lambda (sparsity trade-off)
    - Full evaluation with sparsity metrics and result comparison

Author  : [Candidate]
Date    : 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ===========================================================================
# Part 1 — PrunableLinear
# ===========================================================================

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns which weights to prune.

    Each weight w_{ij} is paired with a learnable scalar gate score g_{ij}.
    During the forward pass:

        gates        = sigmoid(gate_scores)     ∈ (0, 1)
        pruned_weights = weight  ⊙  gates        element-wise product
        output       = x @ pruned_weights.T + bias

    Because sigmoid is smooth and differentiable, gradients flow through
    gate_scores normally. The sparsity loss (applied outside this class)
    pushes gate values toward zero, effectively "removing" weights.

    Parameters
    ----------
    in_features  : int  — number of input features
    out_features : int  — number of output features
    bias         : bool — whether to use a bias term (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight parameter — shape (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Gate score parameter — same shape as weight.
        # Initialized near zero so initial gates ≈ sigmoid(0) = 0.5,
        # giving the network a "full" starting point before pruning begins.
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Weight initialisation — Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Gradient flow analysis
        ----------------------
        dL/d(gate_scores) = dL/d(output)  ·  x  ·  sigmoid'(gate_scores) ⊙ weight
        dL/d(weight)      = dL/d(output)  ·  x  ·  gates

        Both are non-zero (in general), so the optimizer can update both
        parameters independently via standard back-propagation.
        """
        # Step 1: turn raw scores into gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)         # shape: (out, in)

        # Step 2: element-wise product — prune low-gate weights
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3: standard affine transform using pruned weights
        # F.linear computes  x @ weight.T + bias
        return F.linear(x, pruned_weights, self.bias)

    # -----------------------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (after sigmoid) as a flat tensor."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    # -----------------------------------------------------------------------
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 penalty for this layer: sum of all sigmoid(gate_scores) values.

        Since sigmoid output is strictly positive, |gates| = gates, so the
        L1 norm equals the plain sum. This term penalises active gates and,
        combined with the classification loss, pushes gates toward zero.
        """
        return torch.sigmoid(self.gate_scores).sum()

    # -----------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ===========================================================================
# Network definition
# ===========================================================================

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 using PrunableLinear layers.

    CIFAR-10 images are 32×32 RGB → 3072 input features.
    Architecture:  3072 → 1024 → 512 → 256 → 10

    ReLU activations are used between layers.
    Batch normalisation is intentionally omitted to keep the gating
    mechanism the sole form of implicit regularisation studied here.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512,  256)
        self.fc4 = PrunableLinear(256,   10)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)       # flatten: (B, 3, 32, 32) → (B, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)                 # logits — no softmax here (handled by CE loss)
        return x

    # -----------------------------------------------------------------------
    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum L1 gate penalty across all PrunableLinear layers."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss = loss + module.sparsity_loss()
        return loss

    # -----------------------------------------------------------------------
    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Percentage of weights whose gate value is below `threshold`.

        A gate < threshold means the corresponding weight contributes
        negligibly to the output and is considered pruned.
        """
        total   = 0
        pruned  = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates  = module.get_gates()
                total  += gates.numel()
                pruned += (gates < threshold).sum().item()
        return 100.0 * pruned / total if total > 0 else 0.0

    # -----------------------------------------------------------------------
    def get_all_gates(self) -> torch.Tensor:
        """Concatenate all gate values from all layers into one flat tensor."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().flatten())
        return torch.cat(all_gates)


# ===========================================================================
# Data loading
# ===========================================================================

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
    """
    Download (if needed) and return DataLoaders for CIFAR-10.

    Normalisation uses CIFAR-10 channel means and std deviations
    computed from the training set:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_set,  batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ===========================================================================
# Training
# ===========================================================================

def train_one_epoch(
    model:        SelfPruningNet,
    loader:       DataLoader,
    optimizer:    optim.Optimizer,
    lambda_sparse: float,
    device:       torch.device,
) -> tuple[float, float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_total_loss   : float — mean (CE + λ·sparsity) over all batches
    avg_cls_loss     : float — mean CE loss
    avg_sparse_loss  : float — mean sparsity loss
    """
    model.train()
    total_loss_sum  = 0.0
    cls_loss_sum    = 0.0
    sparse_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward
        logits        = model(images)
        cls_loss      = F.cross_entropy(logits, labels)
        sparse_loss   = model.total_sparsity_loss()
        total_loss    = cls_loss + lambda_sparse * sparse_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        total_loss_sum  += total_loss.item()
        cls_loss_sum    += cls_loss.item()
        sparse_loss_sum += sparse_loss.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n, sparse_loss_sum / n


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(
    model:  SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return top-1 test accuracy (%)."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits  = model(images)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total


# ===========================================================================
# Single experiment
# ===========================================================================

def run_experiment(
    lambda_sparse: float,
    train_loader:  DataLoader,
    test_loader:   DataLoader,
    epochs:        int   = 30,
    lr:            float = 1e-3,
    device:        torch.device = DEVICE,
) -> dict:
    """
    Train one model with a specific lambda and return a results dictionary.

    Returns
    -------
    dict with keys: lambda, test_accuracy, sparsity_level, gates, history
    """
    print(f"\n{'='*60}")
    print(f"  Experiment  λ = {lambda_sparse}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"total": [], "cls": [], "sparse": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total_l, cls_l, sparse_l = train_one_epoch(
            model, train_loader, optimizer, lambda_sparse, device
        )
        val_acc = evaluate(model, test_loader, device)
        scheduler.step()

        history["total"].append(total_l)
        history["cls"].append(cls_l)
        history["sparse"].append(sparse_l)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            sparsity = model.compute_sparsity()
            elapsed  = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Total {total_l:.4f} | CE {cls_l:.4f} | "
                f"Sparse {sparse_l:.1f} | Acc {val_acc:.2f}% | "
                f"Sparsity {sparsity:.1f}% | {elapsed:.0f}s"
            )

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.compute_sparsity()
    all_gates      = model.get_all_gates().numpy()

    print(f"\n  FINAL — Accuracy: {final_acc:.2f}%  |  Sparsity: {final_sparsity:.1f}%")

    return {
        "lambda":          lambda_sparse,
        "test_accuracy":   final_acc,
        "sparsity_level":  final_sparsity,
        "gates":           all_gates,
        "history":         history,
    }


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def plot_gate_distribution(results: list[dict], save_path: str = "gate_distribution.png"):
    """
    Plot the gate-value distribution for every lambda in a single figure.
    A successful run shows a spike near 0 and a second cluster away from 0.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        gates   = res["gates"]
        lam     = res["lambda"]
        acc     = res["test_accuracy"]
        sparse  = res["sparsity_level"]

        ax.hist(gates, bins=80, color="black", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Gate Value", fontsize=12)
        ax.set_ylabel("Count",      fontsize=12)
        ax.set_title(
            f"λ = {lam}\nAcc = {acc:.2f}%  |  Sparsity = {sparse:.1f}%",
            fontsize=11
        )
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Gate Value Distributions — Self-Pruning Network",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gate distribution plot saved → {save_path}")


def plot_training_curves(results: list[dict], save_path: str = "training_curves.png"):
    """Plot validation accuracy and CE loss curves for all lambdas."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = ["-", "--", ":"]
    for res, style in zip(results, styles):
        lam     = res["lambda"]
        history = res["history"]
        epochs  = range(1, len(history["val_acc"]) + 1)

        ax1.plot(epochs, history["val_acc"], style, color="black",
                 label=f"λ={lam}", linewidth=1.8)
        ax2.plot(epochs, history["cls"],     style, color="black",
                 label=f"λ={lam}", linewidth=1.8)

    for ax, ylabel, title in [
        (ax1, "Test Accuracy (%)", "Validation Accuracy"),
        (ax2, "CE Loss",           "Classification Loss"),
    ]:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


# ===========================================================================
# Results table (console)
# ===========================================================================

def print_results_table(results: list[dict]):
    header = f"{'Lambda':>10}  {'Test Accuracy (%)':>20}  {'Sparsity Level (%)':>20}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for res in results:
        print(
            f"{res['lambda']:>10.0e}  "
            f"{res['test_accuracy']:>20.2f}  "
            f"{res['sparsity_level']:>20.1f}"
        )
    print(sep)


# ===========================================================================
# Main
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    BATCH_SIZE = 128
    EPOCHS     = 30
    LR         = 1e-3

    # Three lambda values: low / medium / high
    LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("[INFO] Loading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE, num_workers=2
    )

    # -----------------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------------
    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(
            lambda_sparse=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
        )
        all_results.append(result)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print_results_table(all_results)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_gate_distribution(all_results, save_path="gate_distribution.png")
    plot_training_curves(all_results,   save_path="training_curves.png")

    print("\n[INFO] All experiments complete.")
    print("       Outputs: gate_distribution.png  |  training_curves.png")


if __name__ == "__main__":
    main()
