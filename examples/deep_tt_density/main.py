from __future__ import annotations

"""
Preset runners for TT density training on multiple SDE scenarios.

Each case follows the same steps:
1) define constants / hyperparameter ranges
2) create data generators via TrainConfig
3) initialize and train the model
4) save the checkpoint
5) report validation loss
"""

from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from .trainer import build_model, build_streaming_loaders, default_ranks, generate_trajectories, split_datasets, train


def _common_loaders(
    process: str,
    dim: int,
    sample_size: int,
    *,
    t_start: float = 0.0,
    t_end: float = 2.0,
    dt: float = 0.01,
    window: int = 5,
    val_split: float = 0.1,
    batch_size: int = 64,
    mix_ou_prob: float = 0.5,
    x0_std_range=(0.5, 1.5),
    double_well_alpha_range=(0.5, 1.5),
    double_well_sigma_range=(0.3, 0.8),
    ou_theta_range=(0.5, 1.5),
    ou_mu_range=(-0.5, 0.5),
    ou_sigma_range=(0.3, 0.8),
    device: torch.device | None = None,
    streaming: bool = False,
    num_workers: int = 0,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if streaming:
        return build_streaming_loaders(
            process=process,
            dim=dim,
            sample_size=sample_size,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            window=window,
            val_split=val_split,
            batch_size=batch_size,
            mix_ou_prob=mix_ou_prob,
            x0_std_range=x0_std_range,
            double_well_alpha_range=double_well_alpha_range,
            double_well_sigma_range=double_well_sigma_range,
            ou_theta_range=ou_theta_range,
            ou_mu_range=ou_mu_range,
            ou_sigma_range=ou_sigma_range,
            seed=0,
            num_workers=num_workers,
            device=device,
        )
    trajectories = generate_trajectories(
        process,
        dim,
        sample_size,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        double_well_alpha_range=double_well_alpha_range,
        double_well_sigma_range=double_well_sigma_range,
        ou_theta_range=ou_theta_range,
        ou_mu_range=ou_mu_range,
        ou_sigma_range=ou_sigma_range,
        x0_std_range=x0_std_range,
        mix_ou_prob=mix_ou_prob,
        device=device,
    )
    train_ds, val_ds = split_datasets(trajectories, t_start, dt, window, val_split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, device


def run_ou_2d():
    name = "ou_2d"
    dim = 2
    batch_size = 64
    sample_size = 256
    num_epochs = 12
    ranks = default_ranks(dim, rank=4)
    n_per_dim: List[int] = [10] * dim
    train_loader, val_loader, device = _common_loaders(
        "ornstein_uhlenbeck",
        dim,
        sample_size,
        batch_size=batch_size,
        streaming=True,
        num_workers=0,
    )
    model = build_model(
        dim=dim,
        n_per_dim=n_per_dim,
        ranks=ranks,
        linear_transform=False,
        grid_min=-3.0,
        grid_max=3.0,
        d_model=128,
        nhead=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        device=device,
    )
    save_path = Path("checkpoints/ou_2d.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running case: {name} ===")
    best_val = train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=1e-3,
        grad_clip=1.0,
        save_path=str(save_path),
        device=device,
    )
    print(f"Finished {name} | best val NLL: {best_val:.4f}")


def run_double_well_4d():
    name = "double_well_4d"
    dim = 4
    batch_size = 48
    sample_size = 192
    num_epochs = 18
    ranks = default_ranks(dim, rank=4)
    n_per_dim: List[int] = [8] * dim
    train_loader, val_loader, device = _common_loaders(
        "double_well",
        dim,
        sample_size,
        batch_size=batch_size,
        double_well_alpha_range=(0.7, 1.6),
        double_well_sigma_range=(0.3, 0.9),
        streaming=True,
        num_workers=0,
    )
    model = build_model(
        dim=dim,
        n_per_dim=n_per_dim,
        ranks=ranks,
        linear_transform=False,
        grid_min=-3.0,
        grid_max=3.0,
        d_model=160,
        nhead=4,
        num_layers=3,
        ff_dim=320,
        dropout=0.1,
        device=device,
    )
    save_path = Path("checkpoints/double_well_4d.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running case: {name} ===")
    best_val = train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=8e-4,
        grad_clip=1.0,
        save_path=str(save_path),
        device=device,
    )
    print(f"Finished {name} | best val NLL: {best_val:.4f}")


def run_mixed_balanced():
    name = "mixed_balanced"
    dim = 2
    batch_size = 64
    sample_size = 320
    num_epochs = 15
    ranks = default_ranks(dim, rank=4)
    n_per_dim: List[int] = [10] * dim
    train_loader, val_loader, device = _common_loaders(
        "mixed",
        dim,
        sample_size,
        batch_size=batch_size,
        mix_ou_prob=0.5,
        ou_mu_range=(-0.6, 0.6),
        x0_std_range=(0.6, 1.4),
        streaming=True,
        num_workers=0,
    )
    model = build_model(
        dim=dim,
        n_per_dim=n_per_dim,
        ranks=ranks,
        linear_transform=False,
        grid_min=-3.0,
        grid_max=3.0,
        d_model=128,
        nhead=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        device=device,
    )
    save_path = Path("checkpoints/mixed_balanced.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running case: {name} ===")
    best_val = train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=1e-3,
        grad_clip=1.0,
        save_path=str(save_path),
        device=device,
    )
    print(f"Finished {name} | best val NLL: {best_val:.4f}")


def main():
    run_ou_2d()
    run_double_well_4d()
    run_mixed_balanced()


if __name__ == "__main__":
    main()

