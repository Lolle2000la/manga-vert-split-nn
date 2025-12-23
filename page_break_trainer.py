#!/usr/bin/env python3
"""
Training Script for Page Break Detector.

This script handles the training loop using PyTorch Lightning, including data loading,
augmentation, loss calculation (EMD), and optimization.
"""

import argparse
import atexit
import gc
import json
import multiprocessing
import os
import signal
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import lightning.pytorch as pl
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from optuna.integration import PyTorchLightningPruningCallback
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from page_break_metrics import PageBreakMetrics
from page_break_mlflow import AimCallback
from page_break_model import DeepPageBreakDetector

def cleanup_zombies():
    children = multiprocessing.active_children()
    if not children:
        return
    for p in children:
        try:
            name = p.name.lower()
            if "spawn" in name or "process" in name or "worker" in name:
                p.terminate()
                p.join(timeout=0.2)
                if p.is_alive():
                    p.kill()
        except Exception:
            pass

atexit.register(cleanup_zombies)

# --- GLOBAL OPTIMIZATION ---
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True 
cv2.setNumThreads(0)

# --- SAFE PRUNING CALLBACK ---
class SafePruningCallback(PyTorchLightningPruningCallback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        
        try:
            super().on_validation_epoch_end(trainer, pl_module)
        except optuna.TrialPruned:
            try: 
                with open(".pruned_lock", "w") as f:
                    f.write("pruned")
            except:
                pass
            raise RuntimeError("TrialPruned - Force DDP Stop")
        except Exception as e:
            print(f"[WARN] Optuna Pruning comms failed: {e}")

# --- METRIC WRITER ---
class MetricWriter(pl.Callback):
    def __init__(self, filename: str):
        self.filename = filename
        
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            checkpoint_callback = trainer.checkpoint_callback
            score = getattr(checkpoint_callback, "best_model_score", None)
            
            if score is not None:
                try: 
                    score_val = score.item() if isinstance(score, torch.Tensor) else score
                    with open(self.filename, 'w') as f:
                        json.dump({"best_score": score_val}, f)
                except Exception as e:
                    print(f"[WARN] Failed to write best score: {e}")

# --- LOSS ---
class EMDLoss1D(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mass = torch.sigmoid(pred_logits)
        pred_cdf = torch.cumsum(pred_mass, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)
        return torch.mean(torch.abs(pred_cdf - target_cdf))

# --- DATASET ---
class StripDataset(Dataset):
    def __init__(self, data_dir: Union[str, Path], crop_height: Optional[int] = None, augment: bool = False, target_width: int = 768, sample_ratio: float = 1.0, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.image_files = sorted(list((self.data_dir / "images").glob("*.jpg")))
        self.crop_height = crop_height
        self.augment = augment
        self.target_width = target_width
        
        if sample_ratio < 1.0:
            # Use standard random for list shuffling
            import random
            rng = random.Random(seed)
            rng.shuffle(self.image_files)
            num_samples = max(1, int(len(self.image_files) * sample_ratio))
            self.image_files = self.image_files[:num_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.data_dir / "labels" / (img_path.stem + ".npy")
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                scale = self.target_width / w
                new_h = int(h * scale)
                img = img.resize((self.target_width, new_h), resample=Image.Resampling.BILINEAR)
                img_np = np.array(img)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load {img_path}: {e}")

        target = np.load(str(label_path))
        target_t = torch.from_numpy(target).float().view(1, 1, -1)
        target_t = F.interpolate(target_t, size=new_h, mode='linear', align_corners=False)
        target = target_t.view(-1).numpy()
        h = new_h 
        
        if self.crop_height and h > self.crop_height:
            start_y = np.random.randint(0, h - self.crop_height)
            end_y = start_y + self.crop_height
            img_np = img_np[start_y:end_y, :]
            target = target[start_y:end_y]
            
        if self.augment:
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 15, img_np.shape).astype(np.int16)
                img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            if np.random.rand() > 0.5:
                gamma = np.random.uniform(0.8, 1.2)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                img_np = table[img_np]

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target).float()
        return img_tensor, target_tensor

def collate_variable_height(batch):
    images, targets = zip(*batch)
    if len(images) == 1:
        return images[0].unsqueeze(0), targets[0].unsqueeze(0)
    return torch.stack(images), torch.stack(targets)

# --- LIGHTNING MODULE ---
class PageBreakModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Initialize the Standalone Model
        self.model = DeepPageBreakDetector(
            layers=self.hparams["layers"],
            hidden_dim=self.hparams["hidden_dim"],
            kernel_size=self.hparams["kernel_size"],
            dropout=self.hparams.get("dropout", 0.0),
            activation=self.hparams.get("activation", "ReLU"),
            width_stride=self.hparams.get("width_stride", 4)
        )
        
        self.training_model = self.model
        if self.hparams.get("compile_model", False) and hasattr(torch, "compile"):
            print("[Info] Compiling model for training...")
            self.training_model = torch.compile(self.model)
            
        pos_weight_val = self.hparams.get("pos_weight", 1.0)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
        self.emd_criterion = EMDLoss1D()
        
        self.metrics = PageBreakMetrics(tolerance=30)
        self.test_metrics = PageBreakMetrics(tolerance=30) 
        self.inference_mult = 4 

    def on_train_start(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def forward(self, x):
        return self.model(x)

    def predict_sliding_window(self, x, chunk_size=2048, overlap=512):
        b, c, h, w = x.shape
        x_padded = F.pad(x, (0, 0, overlap, overlap))
        
        slices = []
        for start_y in range(0, h, chunk_size):
            slice_start = start_y
            slice_end = start_y + chunk_size + 2 * overlap
            
            if slice_end > x_padded.shape[2]:
                chunk = x_padded[:, :, slice_start:, :]
                is_full = False
            else:
                chunk = x_padded[:, :, slice_start:slice_end, :]
                is_full = True
            
            slices.append({
                "chunk": chunk, 
                "is_full": is_full, 
                "start_y": start_y, 
                "valid_len": h - start_y if start_y + chunk_size >= h else None
            })

        full_chunks = [s["chunk"] for s in slices if s["is_full"]]
        full_logits = []
        
        if full_chunks:
            while self.inference_mult >= 1:
                ref_batch = self.hparams.get("batch_size", 32)
                current_batch_size = max(1, ref_batch * self.inference_mult)
                
                temp_logits = []
                try:
                    with torch.inference_mode():
                        for i in range(0, len(full_chunks), current_batch_size):
                            batch_group = full_chunks[i : i + current_batch_size]
                            batch = torch.cat(batch_group, dim=0)
                            batch_logits = self.model(batch)
                            temp_logits.extend(list(torch.unbind(batch_logits, dim=0)))
                    full_logits = temp_logits
                    break 
                except torch.cuda.OutOfMemoryError:
                    print(f"[Warn] Validation OOM. Downgrading...")
                    self.inference_mult //= 2
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if self.inference_mult < 1:
                        raise RuntimeError("OOM even at 1x inference batch size.")

        final_logits_list = []
        full_idx = 0
        for s in slices:
            if s["is_full"]:
                chunk_logits = full_logits[full_idx].unsqueeze(0)
                full_idx += 1
            else:
                with torch.inference_mode():
                    chunk_logits = self.model(s["chunk"])
            
            if s["valid_len"] is not None:
                valid_logits = chunk_logits[:, overlap : overlap + s["valid_len"]]
            else:
                valid_logits = chunk_logits[:, overlap : -overlap]
            final_logits_list.append(valid_logits)

        return torch.cat(final_logits_list, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(memory_format=torch.channels_last)
        logits = self.training_model(x)
        
        weight = self.hparams.get("emd_weight", 0.5)
        loss = self.criterion(logits, y) + weight * self.emd_criterion(logits, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_sliding_window(x)
        
        if logits.shape[-1] != y.shape[-1]:
            logits = F.interpolate(logits.unsqueeze(1), size=y.shape[-1], mode='linear').squeeze(1)
        
        weight = self.hparams.get("emd_weight", 0.5)
        loss = self.criterion(logits, y) + weight * self.emd_criterion(logits, y)
        
        self.metrics.update(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_sliding_window(x)
        
        if logits.shape[-1] != y.shape[-1]:
            logits = F.interpolate(logits.unsqueeze(1), size=y.shape[-1], mode='linear').squeeze(1)
        
        self.test_metrics.update(logits, y)

    def on_validation_epoch_end(self):
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(k, v, prog_bar=True, sync_dist=True)
        
        self.log("opt_metric", 1.0 - results["val_soft_f1"], prog_bar=False, sync_dist=True)
        self.metrics.reset()

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        for k, v in results.items():
            self.log(f"test_{k}", v, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Any:
        lr = self.hparams.get("lr", 3.6e-3)
        wd = self.hparams.get("weight_decay", 0.0)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["epochs"], eta_min=1e-6)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

# --- MAIN ---
def train_model(config_dict, trial=None, logger=None, final_mode=False, evaluate_on_test=False):
    pl.seed_everything(config_dict['seed'], workers=True)
    data_root = Path(config_dict['data_dir'])
    
    phys_batch = config_dict['batch_size']
    accum = config_dict.get('accumulate_grad_batches', 1)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        strategy = "ddp_spawn"
        sync_batchnorm = False
        devices = num_gpus
        print(f"[Info] Running on {num_gpus} GPUs with DDP Spawn.")
    else:
        strategy = "auto"
        sync_batchnorm = False
        devices = 1
        print(f"[Info] Running on 1 GPU.")

    if final_mode:
        ds_list = []
        for folder in ["train", "val"]:
            if (data_root / folder).exists():
                ds_list.append(StripDataset(data_root / folder, crop_height=config_dict['crop_height'], augment=True, seed=config_dict['seed']))
        train_ds = ConcatDataset(ds_list)
        
        val_folder_name = "test" if (data_root / "test").exists() else "val"
        val_ds = StripDataset(data_root / val_folder_name, crop_height=None, augment=False, seed=config_dict['seed'])
    else:
        train_ds = StripDataset(data_root / "train", crop_height=config_dict['crop_height'], augment=True, sample_ratio=config_dict.get('sample_ratio', 1.0), seed=config_dict['seed'])
        val_ds = StripDataset(data_root / "val", crop_height=None, augment=False, sample_ratio=config_dict.get('sample_ratio', 1.0), seed=config_dict['seed'])
    
    train_loader = DataLoader(
        train_ds, batch_size=phys_batch, shuffle=True, num_workers=4, 
        collate_fn=collate_variable_height, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, 
        collate_fn=collate_variable_height, pin_memory=True, persistent_workers=False
    )

    model = PageBreakModule(config_dict)
    result_file = f"result_{uuid.uuid4().hex}.json"
    
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor='val_soft_f1', mode='max', save_top_k=1, filename='page-break-{epoch:02d}-{val_soft_f1:.4f}'
    )
    callbacks = [checkpoint_callback, MetricWriter(result_file)]
    
    if 'aim_repo' in config_dict:
        callbacks.append(AimCallback(
            repo=config_dict['aim_repo'], 
            experiment=config_dict.get('experiment_name', 'default'), 
            config_dict=config_dict
        ))

    if config_dict.get('patience', 0) > 0:
        callbacks.append(EarlyStopping(monitor="val_soft_f1", min_delta=0.001, patience=config_dict['patience'], verbose=False, mode="max"))
    
    if trial and not final_mode:
        callbacks.append(SafePruningCallback(trial, monitor="opt_metric"))

    logger = CSVLogger("lightning_logs", name="default")

    gradient_clip_val = config_dict.get('gradient_clip_val', 0.5)
    trainer = pl.Trainer(
        max_epochs=config_dict['epochs'], 
        accelerator="gpu", 
        devices=devices, 
        strategy=strategy,
        sync_batchnorm=sync_batchnorm, 
        precision="16-mixed", 
        accumulate_grad_batches=accum,
        callbacks=callbacks, 
        enable_checkpointing=True, 
        logger=logger, 
        log_every_n_steps=10, 
        num_sanity_val_steps=0,
        gradient_clip_val=gradient_clip_val 
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    if evaluate_on_test:
        print("\n[EVAL MODE] Loading Best Checkpoint and running on TEST set...")
        
        test_folder_name = "test" if (data_root / "test").exists() else "val"
        test_ds = StripDataset(data_root / test_folder_name, crop_height=None, augment=False, seed=config_dict['seed'])
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_variable_height, pin_memory=True)
        trainer.test(model, test_loader, ckpt_path="best")
        
    if final_mode:
        print("\n--- Saving Deployment Artifacts ---")
        best_path = checkpoint_callback.best_model_path
        
        if best_path and os.path.exists(best_path):
            print(f"[INFO] Loading BEST model state from: {best_path}")
            model = PageBreakModule.load_from_checkpoint(best_path)
            
            print("[INFO] Re-validating best model to capture accurate metrics...")
            val_results = trainer.validate(model, val_loader, verbose=False)
            
            final_metrics = {}
            for k, v in val_results[0].items():
                if isinstance(v, torch.Tensor):
                    final_metrics[k] = v.item()
                else:
                    final_metrics[k] = v

            deploy_dir = Path("deployment")
            deploy_dir.mkdir(exist_ok=True)
            
            torch.save(model.model.state_dict(), deploy_dir / "best_model.pth")
            
            # Config saving for models
            model_args = getattr(model.model, "config", None)
            if model_args is None:
                # Fallback for checkpoints that may not define config
                model_args = {}
            
            with open(deploy_dir / "model_config.json", "w") as f:
                json.dump(model_args, f, indent=4)
                
            with open(deploy_dir / "final_metrics.json", "w") as f:
                json.dump(final_metrics, f, indent=4)
        else:
            raise RuntimeError(f"[CRITICAL] Could not find 'best_model_path'. Checkpoint: {best_path}")

    best_score = 0.0
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                best_score = data.get("best_score", 0.0)
        except Exception as e:
            print(f"[WARN] Could not read result file: {e}")
        finally:
            os.remove(result_file)
            
    return 1.0 - best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Page Break Detector model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--layers", type=int, default=10, help="Number of residual blocks in the model (default: 10).")
    parser.add_argument("--hidden_dim", type=int, default=80, help="Hidden dimension size of the model (default: 80).")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for convolutions (default: 3).")
    parser.add_argument("--lr", type=float, default=3.6e-3, help="Learning rate (default: 3.6e-3).")
    parser.add_argument("--batch_size", type=int, default=32, help="Physical batch size (default: 32).")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps (default: 1).")
    parser.add_argument("--target_batch_size", type=int, default=32, help="Target effective batch size (used for scaling) (default: 32).")
    parser.add_argument("--crop_height", type=int, default=2048, help="Height of image crops (default: 2048).")
    parser.add_argument("--pos_weight", type=float, default=20.0, help="Weight for positive class in BCE loss (default: 20.0).")
    parser.add_argument("--emd_weight", type=float, default=0.5, help="Weight for EMD loss component (default: 0.5).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6).")
    parser.add_argument("--activation", type=str, default="ReLU", help="Activation function (ReLU, GELU, SiLU) (default: ReLU).")
    parser.add_argument("--dropout", type=float, default=0.025, help="Dropout rate (default: 0.025).")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer name (default: AdamW).")
    parser.add_argument("--weight_decay", type=float, default=6e-6, help="Weight decay (default: 6e-6).")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="Learning rate scheduler (default: CosineAnnealingLR).")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Fraction of dataset to use (default: 1.0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--final", action="store_true", help="Run in final deployment mode (train+val combined, test evaluation).")
    parser.add_argument("--width_stride", type=int, default=2, help="Stride for width reduction (default: 2).")
    parser.add_argument("--evaluate_on_test", action="store_true", help="Evaluate on test set after training.")
    
    # Configurable Aim Repo
    parser.add_argument("--aim_repo", type=str, default="aim://127.0.0.1:53800", help="Aim repository URL (default: aim://127.0.0.1:53800).")
    parser.add_argument("--experiment_name", type=str, default="Standalone_PageBreak", help="Experiment name for tracking (default: Standalone_PageBreak).")

    args = parser.parse_args()
    config = vars(args)
    
    best_loss = train_model(config, final_mode=args.final, evaluate_on_test=args.evaluate_on_test)
    print(f"Finished. Objective (1 - SoftF1): {best_loss}")