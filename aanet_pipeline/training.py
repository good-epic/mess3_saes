from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gc
import torch
import torch.utils.data as torch_data


def _import_aanet_modules():
    try:
        from AAnet.AAnet_torch import models, utils as aanet_utils  # type: ignore
        return models, aanet_utils
    except ImportError:
        import sys

        module_root = Path(__file__).resolve().parents[1] / "AAnet"
        if module_root.exists() and str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        from AAnet_torch import models, utils as aanet_utils  # type: ignore

        return models, aanet_utils


models, aanet_utils = _import_aanet_modules()


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gamma_reconstruction: float
    gamma_archetypal: float
    gamma_extrema: float
    simplex_scale: float
    noise: float
    layer_widths: Sequence[int]
    min_samples: int
    num_workers: int = 0
    shuffle: bool = True
    val_fraction: float = 0.1
    min_val_size: int = 256
    early_stop_patience: int = 10
    early_stop_delta: float = 1e-6
    lr_patience: int = 5
    lr_factor: float = 0.5
    grad_clip: Optional[float] = None
    restarts_no_extrema: int = 1
    active_threshold: float = 1e-6


@dataclass
class TrainingResult:
    k: int
    status: str
    metrics: Dict[str, float]
    epoch_metrics: List[Dict[str, float]]
    model_state_dict: Dict[str, torch.Tensor]


def _train_one_epoch(
    model,
    loader,
    optimizer,
    *,
    epoch: int,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_arche = 0.0
    total_extrema = 0.0
    num_batches = 0

    num_batches_total = max(len(loader), 1)

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            features = batch[0]
        else:
            features = batch
        features = features.view(-1, model.input_shape).to(device)

        if model.diffusion_extrema is not None:
            batch_features = torch.cat(
                (model.diffusion_extrema.view(-1, model.input_shape), features), dim=0
            )
        else:
            batch_features = features

        optimizer.zero_grad()
        output, original, embedding = model(batch_features.float())

        recon_loss = torch.mean((output - batch_features) ** 2)
        arche_loss = model.calc_archetypal_loss(embedding)
        if model.diffusion_extrema is not None:
            extrema_loss = model.calc_diffusion_extrema_loss(embedding)
        else:
            extrema_loss = torch.tensor(0.0, device=device)

        decay = (
            config.gamma_extrema
            / (epoch * num_batches_total + (batch_idx + 1))
            if config.gamma_extrema != 0
            else 0.0
        )
        loss = (
            config.gamma_reconstruction * recon_loss
            + config.gamma_archetypal * arche_loss
            + decay * extrema_loss
        )
        loss.backward()
        if config.grad_clip is not None and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().item())
        total_recon += float(recon_loss.detach().item())
        total_arche += float(arche_loss.detach().item())
        total_extrema += float(extrema_loss.detach().item()) if model.diffusion_extrema is not None else 0.0
        num_batches += 1
        
        del features, output, original, embedding, recon_loss, arche_loss, extrema_loss, loss
        gc.collect()
        torch.cuda.empty_cache()

    num_batches = max(num_batches, 1)
    return {
        "loss": total_loss / num_batches,
        "reconstruction_loss": total_recon / num_batches,
        "archetypal_loss": total_arche / num_batches,
        "extrema_loss": total_extrema / num_batches if model.diffusion_extrema is not None else 0.0,
    }


def _evaluate_epoch(
    model,
    loader,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_arche = 0.0
    count = 0

    noise_backup = model.noise
    model.noise = 0.0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                features = batch[0]
            else:
                features = batch
            features = features.view(-1, model.input_shape).to(device)
            recon, original, embedding = model(features.float())
            recon_loss = torch.mean((recon - original) ** 2)
            arche_loss = model.calc_archetypal_loss(embedding)
            loss = (
                config.gamma_reconstruction * recon_loss
                + config.gamma_archetypal * arche_loss
            )
            total_loss += float(loss.detach().item())
            total_recon += float(recon_loss.detach().item())
            total_arche += float(arche_loss.detach().item())
            count += 1
            
            del features, recon, original, embedding, recon_loss, arche_loss, loss
            gc.collect()
            torch.cuda.empty_cache()
    model.noise = noise_backup
    count = max(count, 1)
    return {
        "loss": total_loss / count,
        "reconstruction_loss": total_recon / count,
        "archetypal_loss": total_arche / count,
    }


def _evaluate_full_dataset(model, data: torch.Tensor, *, device: torch.device) -> Dict[str, float]:
    model.eval()
    noise_backup = model.noise
    model.noise = 0.0
    with torch.no_grad():
        recon, original, embedding = model(data.float())
        recon_mse = torch.nn.functional.mse_loss(recon, original).item()
        arche_loss = model.calc_archetypal_loss(embedding).item()
        barycentric = model.euclidean_to_barycentric(embedding)
        in_simplex = model.is_in_simplex(barycentric).float().mean().item()
    model.noise = noise_backup
    return {
        "reconstruction_mse_eval": float(recon_mse),
        "archetypal_loss_eval": float(arche_loss),
        "in_simplex_fraction": float(in_simplex),
    }


def _train_single_run(
    data: torch.Tensor,
    *,
    k: int,
    config: TrainingConfig,
    device: torch.device,
    diffusion_extrema: Optional[torch.Tensor],
    seed: int,
    noise_override: Optional[float],
) -> TrainingResult:
    import copy

    torch.manual_seed(seed)
    total_samples = data.shape[0]

    val_fraction = max(0.0, min(config.val_fraction, 0.5))
    val_count = int(total_samples * val_fraction)
    if val_count < config.min_val_size:
        val_count = 0

    perm = torch.randperm(total_samples, device=data.device)
    if val_count > 0:
        val_indices = perm[:val_count]
        train_indices = perm[val_count:]
        val_data = data[val_indices]
    else:
        train_indices = perm
        val_data = None
    train_data = data[train_indices]

    train_dataset = torch_data.TensorDataset(train_data)
    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=False,
    )
    if val_data is not None:
        val_loader = torch_data.DataLoader(
            torch_data.TensorDataset(val_data),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
    else:
        val_loader = None

    extrema_tensor = None
    if diffusion_extrema is not None and diffusion_extrema.shape[0] >= k:
        extrema_tensor = diffusion_extrema[:k].to(device)

    noise_value = noise_override if noise_override is not None else config.noise

    model = models.AAnet_vanilla(
        input_shape=data.shape[1],
        n_archetypes=k,
        layer_widths=list(config.layer_widths),
        simplex_scale=config.simplex_scale,
        noise=noise_value,
        device=device,
        diffusion_extrema=extrema_tensor,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if extrema_tensor is not None:
        model.diffusion_extrema = model.diffusion_extrema.to(model.device)

    scheduler = None
    if val_loader is not None and config.lr_patience > 0 and config.lr_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
        )

    epoch_history: List[Dict[str, float]] = []
    best_snapshot = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_stats = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            config=config,
            device=device,
        )

        if val_loader is not None:
            val_stats = _evaluate_epoch(
                model,
                val_loader,
                config=config,
                device=device,
            )
            val_loss = val_stats["loss"]
        else:
            val_stats = {"loss": train_stats["loss"], "reconstruction_loss": train_stats["reconstruction_loss"], "archetypal_loss": train_stats["archetypal_loss"]}
            val_loss = train_stats["loss"]

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_history.append(
            {
                "epoch": float(epoch),
                "loss": float(train_stats["loss"]),
                "reconstruction_loss": float(train_stats["reconstruction_loss"]),
                "archetypal_loss": float(train_stats["archetypal_loss"]),
                "extrema_loss": float(train_stats["extrema_loss"]),
                "val_loss": float(val_loss),
                "lr": float(current_lr),
            }
        )

        if val_loss + config.early_stop_delta < best_val_loss:
            best_val_loss = float(val_loss)
            best_snapshot = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "lr": current_lr,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if config.early_stop_patience > 0 and patience_counter >= config.early_stop_patience:
                break

    if best_snapshot is None:
        best_snapshot = {
            "state_dict": model.state_dict(),
            "epoch": len(epoch_history),
            "train_stats": epoch_history[-1] if epoch_history else {"loss": float("inf"), "reconstruction_loss": float("inf"), "archetypal_loss": float("inf"), "extrema_loss": float("inf")},
            "val_stats": {
                "loss": epoch_history[-1]["loss"] if epoch_history else float("inf"),
                "reconstruction_loss": epoch_history[-1]["reconstruction_loss"] if epoch_history else float("inf"),
                "archetypal_loss": epoch_history[-1]["archetypal_loss"] if epoch_history else float("inf"),
            },
            "lr": optimizer.param_groups[0]["lr"],
        }

    model.load_state_dict(best_snapshot["state_dict"])

    full_eval = _evaluate_full_dataset(model, data.to(device), device=device)

    metrics = {
        "num_samples": float(data.shape[0]),
        "loss_final": float(best_snapshot["train_stats"]["loss"]),
        "reconstruction_loss_final": float(best_snapshot["train_stats"]["reconstruction_loss"]),
        "archetypal_loss_final": float(best_snapshot["train_stats"]["archetypal_loss"]),
        "val_loss_final": float(best_snapshot["val_stats"]["loss"]),
        "val_reconstruction_loss_final": float(best_snapshot["val_stats"]["reconstruction_loss"]),
        "val_archetypal_loss_final": float(best_snapshot["val_stats"]["archetypal_loss"]),
        "epochs_trained": float(best_snapshot["epoch"]),
        "lr_final": float(best_snapshot["lr"]),
        "noise_used": float(noise_value),
        **full_eval,
    }

    return TrainingResult(
        k=k,
        status="ok",
        metrics=metrics,
        epoch_metrics=epoch_history,
        model_state_dict=best_snapshot["state_dict"],
    )


def train_aanet_model(
    data: torch.Tensor,
    *,
    k: int,
    config: TrainingConfig,
    device: torch.device,
    diffusion_extrema: Optional[torch.Tensor],
    seed: int = 0,
    noise_override: Optional[float] = None,
) -> TrainingResult:
    if data.ndim != 2 or data.shape[0] == 0:
        return TrainingResult(
            k=k,
            status="empty_dataset",
            metrics={
                "num_samples": float(data.shape[0]),
            },
            epoch_metrics=[],
            model_state_dict={},
        )
    if data.shape[0] < max(config.min_samples, k):
        return TrainingResult(
            k=k,
            status="insufficient_samples",
            metrics={
                "num_samples": float(data.shape[0]),
            },
            epoch_metrics=[],
            model_state_dict={},
        )

    restart_count = 1
    if diffusion_extrema is None and config.restarts_no_extrema > 1:
        restart_count = config.restarts_no_extrema

    best_result: Optional[TrainingResult] = None
    best_val = float("inf")

    for restart in range(restart_count):
        result = _train_single_run(
            data,
            k=k,
            config=config,
            device=device,
            diffusion_extrema=diffusion_extrema,
            seed=seed + restart * 9973,
            noise_override=noise_override,
        )
        result.metrics["restart_index"] = float(restart)
        val_loss = result.metrics.get("val_loss_final", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            best_result = result

    assert best_result is not None
    return best_result
