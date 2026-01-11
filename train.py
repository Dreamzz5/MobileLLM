#!/usr/bin/env python3
"""
Spatio-Temporal Traffic Prediction Training Script

This script provides comprehensive training, validation, and testing functionality
for spatio-temporal traffic prediction models including STAEformer, ST-LLM, and MobileLLM.

Features:
- Multiple model support (STAEformer, ST_LLM, MobileLLM, STID)
- Multiple dataset support (MILAN, SHANGHAI, TRENTO)
- GPU training with automatic device selection
- Comprehensive logging and model checkpointing
- Early stopping and learning rate scheduling
- Detailed evaluation metrics (RMSE, MAE, R²)

Usage:
    python train.py -d MILAN -m MobileLLM -g 0

Arguments:
    -d, --dataset: Dataset choice (MILAN, SHANGHAI, TRENTO)
    -g, --gpu_num: GPU device number (default: 0)
    -m, --model: Model choice (STAEformer, ST_LLM, MobileLLM, STID)

Author: TMC-Mobile Research Team
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torchinfo import summary

from mobilellm import (
    MobileLLM,
    MaskedMAELoss,
    RMSE_MAE_R2,
    get_dataloaders,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    valset_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scalers: List[Any],
) -> float:
    """
    Evaluate the model on validation/test set.

    Args:
        model: The neural network model to evaluate
        valset_loader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to run evaluation on
        scalers: List of scalers for inverse transformation

    Returns:
        Average loss across all batches
    """
    model.eval()
    batch_loss_list = []

    for batch in valset_loader:
        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        embeddings_batch = batch.get("embeddings", None)
        embeddings_batch = embeddings_batch.to(device) if embeddings_batch is not None else None

        out_batch = model(x_batch, embeddings_batch)
        out_batch_np = out_batch.cpu().numpy()

        # Inverse transform predictions
        for feature_idx in range(out_batch_np.shape[-1]):
            out_batch_np[..., feature_idx] = scalers[feature_idx].inverse_transform(
                out_batch_np[..., feature_idx]
            )

        out_batch = torch.from_numpy(out_batch_np).to(device)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return float(np.mean(batch_loss_list))


@torch.no_grad()
def predict(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, scalers: List[Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for the entire dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing the data
        device: Device to run inference on
        scalers: List of scalers for inverse transformation

    Returns:
        Tuple of (ground_truth, predictions) as numpy arrays
    """
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        embeddings_batch = batch.get("embeddings", None)
        embeddings_batch = embeddings_batch.to(device) if embeddings_batch is not None else None

        out_batch = model(x_batch, embeddings_batch)

        # Inverse transform predictions
        for feature_idx in range(out_batch.shape[-1]):
            out_batch[..., feature_idx] = scalers[feature_idx].inverse_transform(
                out_batch[..., feature_idx]
            )

        y_pred.append(out_batch.cpu().numpy())
        y_true.append(y_batch.cpu().numpy())

    y_pred = np.vstack(y_pred).squeeze()
    y_true = np.vstack(y_true).squeeze()

    return y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    trainset_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    scalers: List[Any],
    clip_grad: Optional[float] = None,
    log: Optional[List[str]] = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model to train
        trainset_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to run training on
        scalers: List of scalers for inverse transformation
        clip_grad: Gradient clipping value (None for no clipping)
        log: Optional logging list

    Returns:
        Average loss for the epoch
    """
    model.train()
    batch_loss_list = []

    for batch in trainset_loader:
        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        embeddings_batch = batch.get("embeddings", None)
        embeddings_batch = embeddings_batch.to(device) if embeddings_batch is not None else None

        out_batch = model(x_batch, embeddings_batch)

        # Inverse transform predictions for loss computation
        for feature_idx in range(out_batch.shape[-1]):
            out_batch[..., feature_idx] = scalers[feature_idx].inverse_transform(
                out_batch[..., feature_idx]
            )

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = float(np.mean(batch_loss_list))
    scheduler.step()
    return epoch_loss


def train(
    model: nn.Module,
    trainset_loader: torch.utils.data.DataLoader,
    valset_loader: torch.utils.data.DataLoader,
    testset_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    scalers: List[Any],
    clip_grad: Optional[float] = None,
    max_epochs: int = 200,
    early_stop: int = 10,
    verbose: int = 1,
    plot: bool = False,
    log: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> nn.Module:
    """
    Train the model with early stopping and validation monitoring.

    Args:
        model: The neural network model to train
        trainset_loader: DataLoader for training data
        valset_loader: DataLoader for validation data
        testset_loader: DataLoader for test data
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to run training on
        scalers: List of scalers for inverse transformation
        clip_grad: Gradient clipping value
        max_epochs: Maximum number of training epochs
        early_stop: Early stopping patience
        verbose: Logging frequency (epochs)
        plot: Whether to generate training plots
        log: Optional logging list
        save_path: Path to save the best model

    Returns:
        Trained model with best validation performance
    """
    model = model.to(device)
    wait = 0
    min_val_loss = float("inf")
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, device, scalers, clip_grad, log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion, device, scalers)
        val_loss_list.append(val_loss)

        test_loss = eval_model(model, testset_loader, criterion, device, scalers)
        test_loss_list.append(test_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                f"Train Loss = {train_loss:.5f}",
                f"Val Loss = {val_loss:.5f}",
                f"Test Loss = {test_loss:.5f}",
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_state_dict, save_path)
        else:
            wait += 1
            if wait >= early_stop:
                break

    # Load best model and compute final metrics
    model.load_state_dict(best_state_dict)
    train_y_true, train_y_pred = predict(model, trainset_loader, device, scalers)
    val_y_true, val_y_pred = predict(model, valset_loader, device, scalers)

    train_rmse, train_mae, train_r2 = RMSE_MAE_R2(train_y_true, train_y_pred)
    val_rmse, val_mae, val_r2 = RMSE_MAE_R2(val_y_true, val_y_pred)

    out_str = (
        f"Early stopping at epoch: {epoch+1}\n"
        f"Best at epoch {best_epoch+1}:\n"
        f"Train Loss = {train_loss_list[best_epoch]:.5f}\n"
        f"Train RMSE = {train_rmse:.5f}, MAE = {train_mae:.5f}, R2 = {train_r2:.5f}\n"
        f"Val Loss = {val_loss_list[best_epoch]:.5f}\n"
        f"Val RMSE = {val_rmse:.5f}, MAE = {val_mae:.5f}, R2 = {val_r2:.5f}\n"
        f"Test Loss = {test_loss_list[best_epoch]:.5f}"
    )
    print_log(out_str, log=log)

    return model


@torch.no_grad()
def test_model(
    model: nn.Module,
    testset_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    scalers: List[Any],
    log: Optional[List[str]] = None,
) -> None:
    """
    Test the trained model and report comprehensive evaluation metrics.

    Args:
        model: The trained neural network model
        testset_loader: DataLoader for test data
        cfg: Configuration dictionary
        device: Device to run testing on
        scalers: List of scalers for inverse transformation
        log: Optional logging list
    """
    model.eval()
    print_log("--------- Test ---------", log=log)

    start_time = time.time()
    y_true, y_pred = predict(model, testset_loader, device, scalers)
    end_time = time.time()

    # Overall metrics
    rmse_all, mae_all, r2_all = RMSE_MAE_R2(y_true, y_pred)
    out_str = f"All Steps RMSE = {rmse_all:.5f}, MAE = {mae_all:.5f}, R2 = {r2_all:.5f}\n"
    print(out_str)
    print(f"Predictions shape: {y_true.shape}, {y_pred.shape}")

    # Per-feature and per-step metrics
    feature_names = cfg.get("feature_names", [f"Feature_{i}" for i in range(y_true.shape[-1])])

    for j in range(y_true.shape[-1]):
        feature_name = feature_names[j] if j < len(feature_names) else f"Feature_{j}"
        print(f"================{feature_name}====================")
        out_str = ""

        for i in range(y_pred.shape[1]):
            rmse, mae, r2 = RMSE_MAE_R2(y_true[:, i, :, j], y_pred[:, i, :, j])
            out_str += f"Step {i + 1} RMSE = {rmse:.5f}, MAE = {mae:.5f}, R2 = {r2:.5f}\n"

        print_log(out_str, log=log, end="")

    inference_time = end_time - start_time
    print_log(f"Inference time: {inference_time:.2f} s", log=log)


def main() -> None:
    """Main training function for MobileLLM model."""
    parser = argparse.ArgumentParser(
        description="Train MobileLLM spatio-temporal prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="TRENTO",
        choices=["MILAN", "SHANGHAI", "TRENTO"],
        help="Dataset to use for training",
    )
    parser.add_argument("-g", "--gpu_num", type=int, default=0, help="GPU device number to use")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="MobileLLM",
        choices=["MobileLLM"],
        help="Model architecture to use",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = torch.randint(1000, (1,)).item()
    seed_everything(seed)
    set_cpu_num(1)

    # Setup device
    gpu_id = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    dataset = args.dataset.upper()
    model_name = args.model
    data_path = f"src/data/{dataset.lower()}"

    # Load configuration
    config_file = "configs/MobileLLM.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    model_architecture = full_cfg.get("model_architecture", {})
    cfg = full_cfg[dataset]
    cfg["model_architecture"] = model_architecture

    # Create model
    model_args = cfg["model_args"].copy()

    if model_name == "MobileLLM":
        model_args["config"] = cfg
        model_args["device"] = device
        # Create model with filtered arguments
        filtered_args = {k: v for k, v in model_args.items() if k != "config"}
        model = MobileLLM(cfg, **filtered_args)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = "src/logs"
    os.makedirs(log_path, exist_ok=True)

    log_filename = f"{model_name}-{dataset}-{timestamp}.log"
    log_file = open(os.path.join(log_path, log_filename), "w", encoding="utf-8")

    print_log(f"Dataset: {dataset}", log=[log_file])
    print_log(f"Model: {model_name}", log=[log_file])
    print_log(f"Random seed: {seed}", log=[log_file])

    # Load data
    train_loader, val_loader, test_loader, scalers = get_dataloaders(
        data_path,
        data_format="stllm" if model_name == "MobileLLM" else "index",
        batch_size=cfg.get("batch_size", 64),
        log=[log_file],
    )

    print_log("", log=[log_file])  # Empty line

    # Setup model saving
    save_path = "saved_models"
    os.makedirs(save_path, exist_ok=True)

    save_filename = f"{model_name}-{dataset}-{timestamp}.pt"
    model_save_path = os.path.join(save_path, save_filename)

    # Setup loss function, optimizer and scheduler
    criterion = MaskedMAELoss() if dataset in ("MILAN", "SHANGHAI", "TRENTO") else None
    if criterion is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
    )

    # Log configuration and model summary
    print_log(f"--------- {model_name} ---------", log=[log_file])
    print_log(json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=[log_file])

    # Get input shape for model summary
    sample_batch = next(iter(train_loader))
    input_shape = (
        cfg["batch_size"],
        cfg["in_steps"],
        cfg["num_nodes"],
        sample_batch["x"].shape[-1],
    )

    print_log(str(summary(model, input_shape, verbose=0)), log=[log_file])
    print_log("", log=[log_file])  # Empty line

    print_log(f"Loss function: {criterion._get_name()}", log=[log_file])
    print_log("", log=[log_file])  # Empty line

    # Train model
    model = train(
        model=model,
        trainset_loader=train_loader,
        valset_loader=val_loader,
        testset_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        scalers=scalers,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=[log_file],
        save_path=model_save_path,
    )

    print_log(f"Saved model: {model_save_path}", log=[log_file])

    # Final testing
    test_model(model, test_loader, cfg, device, scalers, log=[log_file])

    log_file.close()
    print(f"Training completed. Logs saved to: {os.path.join(log_path, log_filename)}")


if __name__ == "__main__":
    main()
