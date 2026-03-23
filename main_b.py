"""
main_b.py — Approach-B (multi-damage) training on single+double combined data.

Trains MidnB (presence + severity per location) on the union of the single-
and double-damage subsets.  Val loss, map_mse and F1 are recorded every epoch
and saved as training curves.  A full evaluate_all metric report is printed
on the held-out test split and the real .mat benchmark after training.

Usage::

    python main_b.py            # 200 epochs (default)
    python main_b.py --epochs 50
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

# --- server compatibility overrides ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

from lib.data_safetensors import get_combined_dataloaders
from lib.model import ModelConfigB, build_criterion_b, build_model
from lib.training import do_training_b, get_opt_and_sched
from lib.visualization import plot_training_results


@dataclass(frozen=True)
class RunConfigB:
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

    # loss knobs
    # pos_weight rule of thumb: (L - K_mean) / K_mean
    # Combined single+double → mean K ≈ 1.5 → (70 - 1.5) / 1.5 ≈ 45.7
    presence_pos_weight: float = 45.7
    severity_weight: float = 1.0


def main(cfg: RunConfigB = RunConfigB()) -> None:
    train_dl, val_dl, test_dl = get_combined_dataloaders(
        ["single", "double"],
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
        model, opt, sched, train_dl, val_dl, cfg.epochs, criterion, ema=ema
    )

    if cfg.save_uuid_checkpoint:
        states_dir = Path("states")
        states_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = states_dir / f"b-combined-{uuid4()}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"[checkpoint] Saved: {ckpt_path}")

    # val_f1s plays the role of val_accs — same plot layout as main.py
    plot_training_results(
        train_losses, val_losses, val_f1s, val_mses,
        save_dir=cfg.save_dir, show=cfg.show_plots,
    )

    from lib.testing import eval_on_loader_b, do_real_test_b
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[test set]")
    results = eval_on_loader_b(trained_model, test_dl, device)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    if cfg.run_real_test:
        print("\n[real benchmark]")
        try:
            do_real_test_b(trained_model, device=device, print_result=True)
        except (FileNotFoundError, OSError) as e:
            print(f"  Skipping (missing benchmark .mat): {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SDINet Approach-B on single+double combined")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    main(RunConfigB(epochs=args.epochs))
