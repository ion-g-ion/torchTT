"""
Transformer-based trainer for TT density models.

This script builds a conditioner (Transformer) that maps a sequence of
conditioning variables (time + history of states) to flattened TT parameters,
then evaluates a `torchtt.nn.TTDensityLayer` on the target state and maximizes
log-likelihood.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

import torchtt
import torchtt.functional as ttfunc

from .generate_data import generate_sample


# ----------------------------
# Utilities
# ----------------------------


def default_ranks(dim: int, rank: int) -> List[int]:
    """Construct TT ranks list with boundary ranks 1."""
    return [1] + [rank] * (dim - 1) + [1]


def _sample_uniform(a: float, b: float) -> float:
    """Sample a float uniformly in [a, b]."""
    return float(torch.empty(1).uniform_(a, b).item())


def _sample_initial_states(sample_size: int, dim: int, std_range: Tuple[float, float], device: torch.device) -> torch.Tensor:
    """Sample random initial states with a random standard deviation in a range."""
    std = _sample_uniform(*std_range)
    return torch.randn(sample_size, dim, device=device) * std


def _sample_double_well_params(dim: int, alpha_range: Tuple[float, float], sigma_range: Tuple[float, float]) -> dict:
    return {
        "dim": dim,
        "alpha": _sample_uniform(*alpha_range),
        "sigma": _sample_uniform(*sigma_range),
    }


def _sample_ou_params(
    dim: int, theta_range: Tuple[float, float], mu_range: Tuple[float, float], sigma_range: Tuple[float, float]
) -> dict:
    return {
        "dim": dim,
        "theta": _sample_uniform(*theta_range),
        "mu": _sample_uniform(*mu_range),
        "sigma": _sample_uniform(*sigma_range),
    }


def _sample_process_params(
    process: str,
    dim: int,
    *,
    double_well_alpha_range: Tuple[float, float],
    double_well_sigma_range: Tuple[float, float],
    ou_theta_range: Tuple[float, float],
    ou_mu_range: Tuple[float, float],
    ou_sigma_range: Tuple[float, float],
):
    if process == "double_well":
        return _sample_double_well_params(dim, double_well_alpha_range, double_well_sigma_range)
    if process == "ornstein_uhlenbeck":
        return _sample_ou_params(dim, ou_theta_range, ou_mu_range, ou_sigma_range)
    raise ValueError(f"Unknown process {process}")


def generate_trajectories(
    process: str,
    dim: int,
    sample_size: int,
    *,
    t_start: float,
    t_end: float,
    dt: float,
    double_well_alpha_range: Tuple[float, float] = (0.5, 1.5),
    double_well_sigma_range: Tuple[float, float] = (0.3, 0.8),
    ou_theta_range: Tuple[float, float] = (0.5, 1.5),
    ou_mu_range: Tuple[float, float] = (-0.5, 0.5),
    ou_sigma_range: Tuple[float, float] = (0.3, 0.8),
    x0_std_range: Tuple[float, float] = (0.5, 1.5),
    mix_ou_prob: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate trajectories for a given configuration.
    Supports single-process or mixed (OU + double well) batches.
    """
    device = device or torch.device("cpu")

    def _one(process_name: str, batch_size: int) -> torch.Tensor:
        hyper = _sample_process_params(
            process_name,
            dim,
            double_well_alpha_range=double_well_alpha_range,
            double_well_sigma_range=double_well_sigma_range,
            ou_theta_range=ou_theta_range,
            ou_mu_range=ou_mu_range,
            ou_sigma_range=ou_sigma_range,
        )
        x0 = _sample_initial_states(batch_size, dim, x0_std_range, device)
        return generate_sample(
            process_name,
            hyper,
            x_start=x0,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            sample_size=batch_size,
        ).to(device)

    if process != "mixed":
        return _one(process, sample_size)

    ou_count = max(1, int(round(sample_size * mix_ou_prob)))
    dw_count = max(1, sample_size - ou_count)
    parts: List[torch.Tensor] = []
    if ou_count > 0:
        parts.append(_one("ornstein_uhlenbeck", ou_count))
    if dw_count > 0:
        parts.append(_one("double_well", dw_count))
    return torch.cat(parts, dim=1)


# ----------------------------
# Data pipeline
# ----------------------------

class TrajectoryWindowDataset(Dataset):
    """
    Creates sliding-window samples from trajectories.

    For each trajectory b and time index i (window <= i < steps-1):
        cond_seq = [(t_{i-window}, x_{i-window}), ..., (t_{i-1}, x_{i-1})]
        target = x_{i}
    """

    def __init__(self, trajectories: torch.Tensor, t_start: float, dt: float, window: int):
        """
        Args:
            trajectories: Tensor [steps, batch, dim]
            t_start: initial time
            dt: time step (equidistant)
            window: conditioning window length
        """
        super().__init__()
        assert trajectories.ndim == 3, "trajectories must be [steps, batch, dim]"
        self.trajectories = trajectories
        self.window = window
        self.times = torch.arange(trajectories.shape[0], device=trajectories.device, dtype=trajectories.dtype) * dt + t_start
        self.indices: List[Tuple[int, int]] = []
        steps, batch, _ = trajectories.shape
        for b in range(batch):
            for i in range(window, steps - 1):
                self.indices.append((b, i))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, i = self.indices[idx]
        sl = slice(i - self.window, i)
        cond_times = self.times[sl].unsqueeze(-1)  # [window, 1]
        cond_states = self.trajectories[sl, b, :]  # [window, dim]
        cond_seq = torch.cat([cond_times, cond_states], dim=1)  # [window, 1+dim]
        target = self.trajectories[i, b, :]  # [dim]
        target_time = self.times[i].unsqueeze(0)  # [1]
        return cond_seq, target, target_time


class StreamingWindowIterable(IterableDataset):
    """
    Streams trajectory windows by repeatedly generating fresh trajectories.
    Avoids holding full datasets in memory.
    """

    def __init__(
        self,
        *,
        process: str,
        dim: int,
        sample_size: int,
        t_start: float,
        t_end: float,
        dt: float,
        window: int,
        val_split: float,
        split: str,
        mix_ou_prob: float = 0.5,
        x0_std_range: Tuple[float, float] = (0.5, 1.5),
        double_well_alpha_range: Tuple[float, float] = (0.5, 1.5),
        double_well_sigma_range: Tuple[float, float] = (0.3, 0.8),
        ou_theta_range: Tuple[float, float] = (0.5, 1.5),
        ou_mu_range: Tuple[float, float] = (-0.5, 0.5),
        ou_sigma_range: Tuple[float, float] = (0.3, 0.8),
        seed: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert split in {"train", "val"}
        self.process = process
        self.dim = dim
        self.sample_size = sample_size
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.window = window
        self.val_split = val_split
        self.split = split
        self.mix_ou_prob = mix_ou_prob
        self.x0_std_range = x0_std_range
        self.double_well_alpha_range = double_well_alpha_range
        self.double_well_sigma_range = double_well_sigma_range
        self.ou_theta_range = ou_theta_range
        self.ou_mu_range = ou_mu_range
        self.ou_sigma_range = ou_sigma_range
        self.seed = seed
        self.device = device or torch.device("cpu")

    def __iter__(self):
        worker_seed = torch.initial_seed()
        torch.manual_seed(worker_seed + self.seed)
        while True:
            trajectories = generate_trajectories(
                self.process,
                self.dim,
                self.sample_size,
                t_start=self.t_start,
                t_end=self.t_end,
                dt=self.dt,
                double_well_alpha_range=self.double_well_alpha_range,
                double_well_sigma_range=self.double_well_sigma_range,
                ou_theta_range=self.ou_theta_range,
                ou_mu_range=self.ou_mu_range,
                ou_sigma_range=self.ou_sigma_range,
                x0_std_range=self.x0_std_range,
                mix_ou_prob=self.mix_ou_prob,
                device=self.device,
            )

            steps, batch, _ = trajectories.shape
            val_count = max(1, int(batch * self.val_split))
            train_count = batch - val_count
            if train_count < 1:
                raise ValueError("Train split too small; reduce val_split or increase sample_size.")

            if self.split == "train":
                subset = trajectories[:, :train_count, :]
            else:
                subset = trajectories[:, train_count:, :]

            ds = TrajectoryWindowDataset(subset, self.t_start, self.dt, self.window)
            for sample in ds:
                yield sample



# ----------------------------
# Model
# ----------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerConditioner(nn.Module):
    """Transformer encoder that outputs flattened TT parameters."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            tt_params: [batch, output_dim]
        """
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class TTDensityTransformer(nn.Module):
    """Full model: transformer conditioner + TT density layer."""

    def __init__(self, conditioner: TransformerConditioner, tt_layer: torchtt.nn.TTDensityLayer):
        super().__init__()
        self.conditioner = conditioner
        self.tt_layer = tt_layer

    def forward(self, cond_seq: torch.Tensor, target_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tts_params = self.conditioner(cond_seq)
        pdf_val = self.tt_layer(tts_params, target_x)
        return pdf_val, tts_params


# ----------------------------
# Dataset helpers
# ----------------------------


def build_tt_layer(dim: int, n_per_dim: List[int], ranks: List[int], linear_transform: bool, grid_min: float, grid_max: float):
    basis = [
        ttfunc.GaussianBasis(torch.linspace(grid_min, grid_max, n_per_dim[i]), delta_overlap=1) for i in range(dim)
    ]
    return torchtt.nn.TTDensityLayer(n_per_dim, ranks, basis, linear_transformation=linear_transform)


def split_datasets(trajectories: torch.Tensor, t_start: float, dt: float, window: int, val_split: float) -> Tuple[Dataset, Dataset]:
    steps, batch, _ = trajectories.shape
    val_count = max(1, int(batch * val_split))
    train_count = batch - val_count
    if train_count < 1:
        raise ValueError("Train split too small; reduce val_split or increase sample_size.")

    train_traj = trajectories[:, :train_count, :]
    val_traj = trajectories[:, train_count:, :]

    train_ds = TrajectoryWindowDataset(train_traj, t_start, dt, window)
    val_ds = TrajectoryWindowDataset(val_traj, t_start, dt, window)
    return train_ds, val_ds


def build_streaming_loaders(
    *,
    process: str,
    dim: int,
    sample_size: int,
    t_start: float,
    t_end: float,
    dt: float,
    window: int,
    val_split: float,
    batch_size: int,
    num_workers: int = 0,
    mix_ou_prob: float = 0.5,
    x0_std_range: Tuple[float, float] = (0.5, 1.5),
    double_well_alpha_range: Tuple[float, float] = (0.5, 1.5),
    double_well_sigma_range: Tuple[float, float] = (0.3, 0.8),
    ou_theta_range: Tuple[float, float] = (0.5, 1.5),
    ou_mu_range: Tuple[float, float] = (-0.5, 0.5),
    ou_sigma_range: Tuple[float, float] = (0.3, 0.8),
    seed: int = 0,
    device: torch.device | None = None,
):
    """Create streaming DataLoaders that do not hold all trajectories in memory."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_kwargs = dict(
        process=process,
        dim=dim,
        sample_size=sample_size,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        window=window,
        val_split=val_split,
        mix_ou_prob=mix_ou_prob,
        x0_std_range=x0_std_range,
        double_well_alpha_range=double_well_alpha_range,
        double_well_sigma_range=double_well_sigma_range,
        ou_theta_range=ou_theta_range,
        ou_mu_range=ou_mu_range,
        ou_sigma_range=ou_sigma_range,
        seed=seed,
        device=device,
    )
    train_ds = StreamingWindowIterable(split="train", **common_kwargs)
    val_ds = StreamingWindowIterable(split="val", **common_kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, device


# ----------------------------
# Model builders
# ----------------------------


def build_model(
    dim: int,
    n_per_dim: List[int],
    ranks: List[int],
    linear_transform: bool,
    grid_min: float,
    grid_max: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
    device: torch.device,
) -> TTDensityTransformer:
    if len(ranks) != dim + 1 or ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError(f"Ranks must be length dim+1 with boundary 1s, got {ranks}")

    tt_layer = build_tt_layer(dim, n_per_dim, ranks, linear_transform, grid_min, grid_max).to(device)
    tt_input_size = torchtt.nn.TTDensityLayer.input_requireemnt(n_per_dim, ranks, linear_transform)

    conditioner = TransformerConditioner(
        input_dim=1 + dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff_dim,
        dropout=dropout,
        output_dim=tt_input_size,
    )
    model = TTDensityTransformer(conditioner, tt_layer).to(device)
    return model


# ----------------------------
# Training loop (takes DataLoaders)
# ----------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_epochs: int,
    lr: float,
    eps: float = 1e-9,
    grad_clip: float | None = None,
    train_steps: int | None = None,
    val_steps: int | None = None,
    save_path: str | None = None,
    device: torch.device | None = None,
) -> float:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for step, (cond_seq, target, _) in enumerate(train_loader):
            if train_steps is not None and step >= train_steps:
                break
            cond_seq = cond_seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pdf_val, _ = model(cond_seq, target)
            loss = -(torch.log(pdf_val + eps)).mean()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * cond_seq.size(0)
            total_seen += cond_seq.size(0)

        avg_train = total_loss / max(1, total_seen)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for step, (cond_seq, target, _) in enumerate(val_loader):
                if val_steps is not None and step >= val_steps:
                    break
                cond_seq = cond_seq.to(device)
                target = target.to(device)
                pdf_val, _ = model(cond_seq, target)
                loss = -(torch.log(pdf_val + eps)).mean()
                val_loss += loss.item() * cond_seq.size(0)
                val_seen += cond_seq.size(0)
        avg_val = val_loss / max(1, val_seen)

        print(f"Epoch {epoch+1}/{num_epochs} | train NLL {avg_train:.4f} | val NLL {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            if save_path:
                torch.save({"model_state": model.state_dict()}, save_path)
                print(f"  Saved checkpoint to {save_path}")
    return best_val


