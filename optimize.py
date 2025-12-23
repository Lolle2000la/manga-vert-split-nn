#!/usr/bin/env python3
"""
Hyperparameter Optimization Script.

This script uses Optuna to optimize the hyperparameters of the Page Break Detector
model, including architecture (layers, hidden dim) and training (lr, weight decay).
"""

import argparse
import atexit
import gc
import json
import multiprocessing
import os
import signal
import sys
import time
import warnings
import sqlite3

import optuna
import torch
import torch._dynamo
from optuna.samplers import TPESampler
from sqlalchemy.exc import OperationalError

from page_break_trainer import train_model

torch.set_float32_matmul_precision('medium')

# --- Handle VRAM Fragmentation ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# --- HARDWARE STABILITY FIXES ---
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def cleanup_zombies():
    """
    Kills PyTorch DDP spawn workers but tries to spare unrelated 
    processes.
    """
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
        except Exception as e:
            pass

atexit.register(cleanup_zombies)

# --- GRACEFUL STOPPING LOGIC ---
STOP_REQUESTED = False

def signal_handler(signum, frame):
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        print("\n\n" + "="*50)
        print("[INFO] 🛑 GRACEFUL STOP REQUESTED!")
        print("[INFO] The script will finish the current trial and then exit.")
        print("[INFO] Press Ctrl+C again to abort immediately.")
        print("="*50 + "\n")
        STOP_REQUESTED = True
    else:
        print("\n[WARN] ✋ Immediate stop requested! Aborting trial...")
        signal.signal(signal.SIGINT, signal.default_int_handler)
        raise KeyboardInterrupt

def graceful_stop_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        print(f"[INFO] Stopping study gracefully after Trial {trial.number}.")
        study.stop()

# --- OPTIMIZATION OBJECTIVE ---
def objective(trial: optuna.trial.Trial) -> float:
    # Reset Dynamo at start to ensure clean slate
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    # --- 1. STATIC SEARCH SPACE ---
    optimizer_name = "AdamW"
    scheduler_name = "CosineAnnealingLR"
    kernel_size = 3
    width_stride = 2
    target_batch_size = 32
    
    # --- 2. TUNABLE PARAMETERS ---
    layers = trial.suggest_int("layers", 8, 12)
    hidden_dim = trial.suggest_int("hidden_dim_size", 48, 128, step=16)
    activation = trial.suggest_categorical("activation", ["ReLU", "GELU", "SiLU"])
    dropout = trial.suggest_float("dropout", 0.0, 0.05)
    pos_weight = trial.suggest_float("pos_weight", 2.0, 25.0)
    emd_weight = trial.suggest_float("emd_weight", 0.01, 2.0, log=True)
    
    lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    current_phys_batch = args.batch_size
    
    # --- DYNAMIC COMPILATION FLAG ---
    use_compile = True 

    # --- OOM / CRASH RETRY LOOP ---
    while current_phys_batch >= 1:
        
        # Check and remove sentinel file from previous run if it exists
        if os.path.exists(".pruned_lock"):
            os.remove(".pruned_lock")
        
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0: num_gpus = 1
        max_math_batch = max(1, target_batch_size // num_gpus)
        actual_phys_batch = min(current_phys_batch, max_math_batch)
        total_hardware_batch = actual_phys_batch * num_gpus
        accum = max(1, target_batch_size // total_hardware_batch)
        
        config = {
            "data_dir": args.data_dir,
            "batch_size": actual_phys_batch,
            "accumulate_grad_batches": accum, 
            "crop_height": args.crop_height,            
            "epochs": 50,                   
            "patience": 6, 
            "layers": layers, "hidden_dim": hidden_dim, "kernel_size": kernel_size, "dropout": dropout,
            "activation": activation, "pos_weight": pos_weight, "emd_weight": emd_weight,
            "target_batch_size": target_batch_size,
            "width_stride": width_stride, 
            "optimizer": optimizer_name, "lr": lr, "weight_decay": weight_decay,
            "scheduler": scheduler_name, 
            "sample_ratio": args.sample_ratio, "seed": trial.number, 
            "compile_model": use_compile,
            "trial_number": trial.number,
            "aim_repo": args.aim_repo, 
            "experiment_name": args.study_name # PASS AIM CONFIG HERE
        }

        try:
            print(f"[Trial {trial.number}] Attempting PhysBatch={actual_phys_batch} (Accum={accum}, Compile={use_compile})...")
            # We do NOT pass logger anymore; train_model builds callbacks from config
            val_result = train_model(config, trial=trial)
            return val_result
        
        except optuna.TrialPruned as e:
            print(f"[Trial {trial.number}] Pruned: {str(e)}")
            raise e 

        except Exception as e:
            error_str = str(e).lower()
            
            # --- ROBUST PRUNING CHECK (SENTINEL FILE) ---
            if os.path.exists(".pruned_lock"):
                print(f"[Trial {trial.number}] Pruned (detected via Sentinel File).")
                try: os.remove(".pruned_lock")
                except: pass
                raise optuna.TrialPruned()

            if "trialpruned" in error_str:
                print(f"[Trial {trial.number}] Pruned (detected via Exception String).")
                raise optuna.TrialPruned()

            # PRIORITY: Check for OOM *FIRST*
            elif "out of memory" in error_str:
                print(f"[Trial {trial.number}] OOM at batch_size={actual_phys_batch}.")
                if current_phys_batch <= 4:
                    print(f"[Trial {trial.number}] STRICT MODE: Batch size dropping <= 4. Pruning.")
                    raise optuna.TrialPruned()
                
                print("Reducing batch size and retrying...")
                current_phys_batch //= 2
                continue 

            # Check for ACTUAL COMPILATION ERRORS
            elif ("inductor" in error_str or 
                  "illegal memory access" in error_str or 
                  "acceleratorerror" in error_str):
                
                if use_compile:
                    print(f"\n[WARN] Trial {trial.number} crashed due to Compilation/CUDA Error.")
                    print("[INFO] Disabling torch.compile and RETRYING immediately...")
                    use_compile = False
                    continue 
                else:
                    raise e
            
            elif "keyboardinterrupt" in error_str or isinstance(e, KeyboardInterrupt):
                print(f"\n[WARN] Trial {trial.number} aborted by user.")
                raise RuntimeError("User Aborted Trial")
            
            else:
                raise e
        
        finally:
            try:
                gc.collect()
                torch.cuda.empty_cache()
                cleanup_zombies()
                if os.path.exists(".pruned_lock"): os.remove(".pruned_lock")
            except Exception as cleanup_error:
                print(f"[WARN] Cleanup failed (Ignored to preserve Trial status): {cleanup_error}")
    
    # Fallback if loop finishes without return (should not happen if logic is correct)
    raise optuna.TrialPruned("Batch size reduced to < 1")
        
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for the Page Break Detector using Optuna.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_manga.db", help="Optuna storage URL (default: sqlite:///optuna_manga.db).") 
    parser.add_argument("--study_name", type=str, default="page_break_refined_emd", help="Name of the Optuna study (default: page_break_refined_emd).")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run (default: 50).")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Fraction of the dataset to use for optimization (default: 0.1).")
    parser.add_argument("--batch_size", type=int, default=32, help="Physical batch size for training (default: 32).") 
    parser.add_argument("--crop_height", type=int, default=2048, help="Height of the image crops (default: 2048).")
    
    # Configurable Aim Repo
    parser.add_argument("--aim_repo", type=str, default="aim://127.0.0.1:53800", help="Aim repository URL for experiment tracking (default: aim://127.0.0.1:53800).")

    # Flags to control post-training steps
    parser.add_argument("--final", action="store_true", help="Run a final full training on the best model after optimization")
    parser.add_argument("--evaluate_on_test", action="store_true", help="Run evaluation on the test set for the best model")
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    target_trials = args.n_trials

    # --- ROBUST OPTIMIZATION LOOP ---
    study = None
    while True:
        try:
            # 1. Connect / Create Study
            study = optuna.create_study(
                direction="minimize", 
                storage=args.storage,
                study_name=args.study_name,
                load_if_exists=True,
                # Unseeded for better exploration
                sampler=TPESampler(seed=None, multivariate=True),
                # Warmup to allow CosineAnnealingLR to work
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=12, interval_steps=1)
            )
            
            completed_trials = len(study.trials)
            remaining_trials = target_trials - completed_trials
            
            if remaining_trials <= 0:
                print(f"Study complete ({completed_trials} trials done).")
                break
            
            print(f"\n--- Starting Optimization Chunk ({remaining_trials} left) ---")
            
            study.optimize(
                objective, 
                n_trials=remaining_trials,
                callbacks=[graceful_stop_callback]
            )
            break

        except (OperationalError, sqlite3.OperationalError, OSError) as e:
            print(f"\n[CRITICAL] Network/DB Connection Lost: {e}")
            time.sleep(60)
            continue
        except KeyboardInterrupt:
            STOP_REQUESTED = True
            break
        except Exception as e:
            print(f"\n[CRITICAL] Unexpected Error: {e}")
            time.sleep(60)
            continue

    print("\n" + "="*50)
    if study is not None and len(study.trials) > 0:
        print("OPTIMIZATION FINISHED. BEST PARAMS:")
        print(study.best_params)
    print("="*50 + "\n")

    # --- POST-OPTIMIZATION STEPS ---
    if not STOP_REQUESTED and study is not None:
        best = study.best_params
        if hasattr(torch, "_dynamo"): torch._dynamo.reset()

        # Fixed params are not in 'best' dict, we must add them back manually
        best['width_stride'] = 2
        best['target_batch_size'] = 32
        best['activation'] = best.get('activation', "ReLU") 
        best['optimizer'] = "AdamW"
        best['kernel_size'] = 3
        
        # Map 'hidden_dim_size' -> 'hidden_dim'
        if 'hidden_dim_size' in best:
            best['hidden_dim'] = best['hidden_dim_size']
        
        # --- PHASE 1: EVALUATION RUN ---
        if args.evaluate_on_test:
            print("\n--- PHASE 1: EVALUATION RUN (Test Set Check) ---")
            use_compile = True 
            final_phys_batch = args.batch_size
            
            while final_phys_batch >= 1:
                num_gpus = torch.cuda.device_count() or 1
                max_math = max(1, best['target_batch_size'] // num_gpus)
                actual_batch = min(final_phys_batch, max_math)
                accum = max(1, best['target_batch_size'] // (actual_batch * num_gpus))
                
                eval_config = {
                    "data_dir": args.data_dir,
                    "batch_size": actual_batch,   
                    "accumulate_grad_batches": accum,
                    "crop_height": args.crop_height,
                    "epochs": 100,  
                    "patience": 10, 
                    "sample_ratio": 1.0, 
                    "seed": 42,
                    "compile_model": use_compile, 
                    "aim_repo": args.aim_repo,
                    "experiment_name": args.study_name + "_Eval",
                    **best 
                }
                
                try:
                    train_model(eval_config, final_mode=False, evaluate_on_test=True)
                    print("Evaluation Run Complete.")
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if "out of memory" in error_str:
                        print("OOM on Eval Build. Reducing...")
                        gc.collect(); torch.cuda.empty_cache()
                        final_phys_batch //= 2
                    elif use_compile and ("inductor" in error_str or "illegal memory access" in error_str):
                        print("Compiler crash on Eval. Disabling compilation...")
                        use_compile = False
                        gc.collect(); torch.cuda.empty_cache()
                    else:
                        raise e

        # --- PHASE 2: DEPLOYMENT RUN ---
        if args.final:
            print("\n--- PHASE 2: DEPLOYMENT RUN (Full Data Training) ---")
            if hasattr(torch, "_dynamo"): torch._dynamo.reset()
            final_phys_batch = args.batch_size 
            use_compile = True 
            
            while final_phys_batch >= 1:
                num_gpus = torch.cuda.device_count() or 1
                max_math = max(1, best['target_batch_size'] // num_gpus)
                actual_batch = min(final_phys_batch, max_math)
                accum = max(1, best['target_batch_size'] // (actual_batch * num_gpus))
                
                deploy_config = {
                    "data_dir": args.data_dir,
                    "batch_size": actual_batch,   
                    "accumulate_grad_batches": accum,
                    "crop_height": args.crop_height,
                    "epochs": 100,  
                    "patience": 10, 
                    "sample_ratio": 1.0, 
                    "seed": 42,
                    "compile_model": use_compile,
                    "aim_repo": args.aim_repo,
                    "experiment_name": args.study_name + "_Final",
                    **best 
                }

                try:
                    train_model(deploy_config, final_mode=True)
                    print("\nAll Done! Check 'deployment/' folder.")
                    break 
                except Exception as e:
                    error_str = str(e).lower()
                    if "out of memory" in error_str:
                        print("OOM on Deployment. Reducing...")
                        gc.collect(); torch.cuda.empty_cache()
                        final_phys_batch //= 2
                    elif use_compile and ("inductor" in error_str or "illegal memory access" in error_str):
                        print("Compiler crash on Deployment. Disabling compilation...")
                        use_compile = False
                        gc.collect(); torch.cuda.empty_cache()
                    else:
                        raise e