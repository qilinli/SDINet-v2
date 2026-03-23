"""
main_b.py — entry point for Approach-B (multi-damage) training.

Parallel to main.py (v1 single-damage).  The key differences:
  - Uses ModelConfigB → MidnB head (presence + severity per location)
  - Uses PresenceSeverityLoss instead of MSE + cross-entropy
  - Calls do_training_b / val_one_epoch_b which report F1 instead of loc accuracy
  - Checkpoints saved under 'multi-damage-B-<uuid>.pt'

To run a quick comparison:
    python main.py     # v1 baseline
    python main_b.py   # v2 Approach-B

Both write comparable val_mse (distributed-map MSE) to the plots so you can
overlay them for ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

# --- server compatibility overrides (same as main.py) ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

from lib.data_safetensors import get_dataloaders
from lib.model import ModelConfigB, build_criterion_b, build_model
from lib.training import do_training_b, get_opt_and_sched
from lib.visualization import plot_training_results


@dataclass(frozen=True)
class RunConfigB:
    # dataset
    subset_name: str = "double"   # "single" | "double" — use double for multi-damage
    snr: float = -1.0
    epochs: int = 200

    # dataloaders
    data_root: str = "data/safetensors/unc=0"
    num_workers: int = 0
    train_batch_size: int = 128
    eval_batch_size: int = 32
    split_seed: int = 42

    # outputs
    save_dir: str = "saved_results_b"
    show_plots: bool = False
    save_uuid_checkpoint: bool = True
    run_real_test: bool = True

    # --- model / loss knobs ---
    # pos_weight: BCE positive-class weight.
    #   Single-damage (K=1): (70-1)/1 = 69.
    #   Double-damage (K=2): (70-2)/2 = 34.  Adjust to match your subset.
    presence_pos_weight: float = 34.0
    severity_weight: float = 1.0


def main(cfg: RunConfigB = RunConfigB()) -> None:
    train_dl, val_dl, test_dl = get_dataloaders(
        cfg.subset_name,
        cfg.snr,
        root=cfg.data_root,
        num_workers=cfg.num_workers,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        seed=cfg.split_seed,
    )

    model_cfg = ModelConfigB(
        presence_pos_weight=cfg.presence_pos_weight,
        severity_weight=cfg.severity_weight,
    )
    model     = build_model(model_cfg)
    criterion = build_criterion_b(model_cfg)

    ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )

    opt, sched = get_opt_and_sched(model, train_dl, cfg.epochs)

    train_losses, val_losses, _eval_dl, val_f1s, val_mses, trained_model = do_training_b(
        model, opt, sched, train_dl, test_dl, cfg.epochs, criterion, ema=ema
    )

    if cfg.save_uuid_checkpoint:
        states_dir = Path("states")
        states_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = states_dir / f"b-{cfg.subset_name}-{uuid4()}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"[checkpoint] Saved: {ckpt_path}")

    # val_f1s plays the role of val_accs in main.py — same plot layout
    plot_training_results(
        train_losses,
        val_losses,
        val_f1s,    # ← F1 instead of location accuracy
        val_mses,
        save_dir=cfg.save_dir,
        show=cfg.show_plots,
    )

    if cfg.run_real_test:
        from lib.testing import do_real_test_b
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            do_real_test_b(trained_model, device=device, print_result=True)
        except (FileNotFoundError, OSError) as e:
            print(f"[do_real_test_b] Skipping (missing benchmark .mat): {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SDINet Approach-B")
    parser.add_argument("--subset", default="double", choices=["single", "double"])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    # pos_weight rule of thumb: (L - K) / K  where L=70
    pos_weight = 69.0 if args.subset == "single" else 34.0
    main(RunConfigB(subset_name=args.subset, epochs=args.epochs, presence_pos_weight=pos_weight))
