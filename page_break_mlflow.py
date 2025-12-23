"""
Aim Callback for PyTorch Lightning.

This module provides a custom callback to log training metrics and loss curves
to Aim (an experiment tracking tool).
"""

import os
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from aim import Image, Run


class AimCallback(pl.Callback):
    def __init__(self, repo: str, experiment: str, config_dict: Dict[str, Any]):
        super().__init__()
        self.repo = repo
        self.experiment = experiment
        self.config_dict = config_dict
        self.run: Optional[Run] = None
        self.train_loss = []
        self.val_loss = []
        self._rank = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        # Initialize Aim ONLY on the main process (Rank 0)
        # This happens INSIDE the DDP worker, so no pickling needed!
        self._rank = trainer.global_rank
        if self._rank == 0:
            try:
                self.run = Run(repo=self.repo, experiment=self.experiment)

                if self.run is not None:
                    # Convert config to basic types for logging
                    clean_config = {}
                    for k, v in self.config_dict.items():
                        if isinstance(v, (int, float, str, bool)):
                            clean_config[k] = v
                    self.run["hparams"] = clean_config
            except Exception as e:
                print(f"[WARN] Failed to initialize Aim: {e}")
                self.run = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._rank != 0 or self.run is None:
            return

        metrics = trainer.callback_metrics
        # Log all available scalar metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                self.run.track(v.item(), name=k, context={
                               "subset": "train" if "train" in k else "val"})

        # Track for plotting
        t_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        if t_loss is not None:
            self.train_loss.append(t_loss.item())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._rank != 0 or self.run is None:
            return

        v_loss = trainer.callback_metrics.get("val_loss")
        if v_loss is not None:
            self.val_loss.append(v_loss.item())
            # Explicit tracking for validation to ensure it shows up
            self.run.track(v_loss.item(), name="val_loss",
                           context={"subset": "val"})

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._rank != 0 or self.run is None:
            return

        # --- Generate Loss Plot ---
        if self.train_loss and self.val_loss:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(self.train_loss, label='Train Loss')

                # Interpolate Val loss to match Train x-axis if needed
                if len(self.val_loss) > 0:
                    val_x = [i * (len(self.train_loss) / max(1, len(self.val_loss)))
                             for i in range(len(self.val_loss))]
                    plt.plot(val_x, self.val_loss, label='Val Loss')

                plt.title('Training & Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)

                plot_path = "loss_curve_temp.png"
                plt.savefig(plot_path)
                plt.close()

                self.run.track(
                    Image(plot_path, caption="Loss Curve"), name="loss_curve")
                if os.path.exists(plot_path):
                    os.remove(plot_path)
            except Exception as e:
                print(f"[WARN] Plotting failed: {e}")

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self._rank == 0 and self.run is not None:
            self.run.close()
            self.run = None
