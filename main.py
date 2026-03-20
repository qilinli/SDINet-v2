from __future__ import annotations

from dataclasses import dataclass

import torch
from pathlib import Path
from uuid import uuid4

from lib.data_safetensors import get_dataloaders
from lib.model import ModelConfig, build_model
from lib.training import do_training, get_opt_and_sched
from lib.visualization import plot_training_results

# --- SERVER COMPATIBILITY OVERRIDES ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import os
# Disable NVLink lookups which are causing the symbol error
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# Force PyTorch to ignore NVML (NVIDIA Management Library) for device queries
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"


@dataclass(frozen=True)
class RunConfig:
    # subset folder name under `data_root`:
    #   data/safetensors/unc={0|1}/{single|double|undamaged}
    subset_name: str = "single"
    snr: float = -1.0
    epochs: int = 200

    # dataloaders
    # choose uncertainty split here (unc=0 or unc=1)
    data_root: str = "data/safetensors/unc=0"
    num_workers: int = 0
    train_batch_size: int = 128
    eval_batch_size: int = 32
    split_seed: int = 42

    # outputs
    save_dir: str = "saved_results"
    show_plots: bool = False

    # evaluation on the real-world benchmark .mat file
    run_real_test: bool = True

    # save trained model weights after training (notebook-style UUID file)
    save_uuid_checkpoint: bool = True


def main(cfg: RunConfig = RunConfig()) -> None:
    train_dl, val_dl, test_dl = get_dataloaders(
        cfg.subset_name,
        cfg.snr,
        root=cfg.data_root,
        num_workers=cfg.num_workers,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        seed=cfg.split_seed,
    )

    model = build_model(
        ModelConfig(
            in_channels=1,
            time_len=500,
            n_sensors=65,
            structure=(6, 6, 6),
            embed_dim=768,
            out_channels=71,
            importance_dropout=0.5,
            temperature=1e-2,
            val_temperature=1e-2,
            neck_dropout=0.0,
        )
    )
    ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )

    opt, sched = get_opt_and_sched(model, train_dl, cfg.epochs)
    train_losses, val_losses, _eval_dl, *other_stats = do_training(
        model, opt, sched, train_dl, test_dl, cfg.epochs, ema=ema
    )

    val_accs = other_stats[0] if len(other_stats) > 0 else None
    val_mses = other_stats[1] if len(other_stats) > 1 else None

    # `do_training` returns the accelerator-prepared model as the last element.
    trained_model = other_stats[2] if len(other_stats) > 2 else model

    if cfg.save_uuid_checkpoint:
        states_dir = Path("states")
        states_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = states_dir / f"single-damage-dense-{uuid4()}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"[checkpoint] Saved: {ckpt_path}")
    plot_training_results(
        train_losses,
        val_losses,
        val_accs,
        val_mses,
        save_dir=cfg.save_dir,
        show=cfg.show_plots,
    )

    if cfg.run_real_test:
        from lib.testing import do_real_test

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            do_real_test(trained_model, device=device, print_result=True)
        except (FileNotFoundError, OSError) as e:
            print(f"[do_real_test] Skipping (missing benchmark .mat): {e}")


if __name__ == "__main__":
    main()