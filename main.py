from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

from lib.data_safetensors import get_combined_dataloaders
from lib.model import ModelConfig, build_model
from lib.training import do_training, get_opt_and_sched
from lib.visualization import plot_training_results

# --- server compatibility overrides ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"


@dataclass(frozen=True)
class RunConfig:
    snr: float = -1.0
    epochs: int = 200

    # dataloaders
    data_root: str = "data/safetensors/unc=0"
    num_workers: int = 0
    train_batch_size: int = 256
    eval_batch_size: int = 32
    split_seed: int = 42

    # outputs
    save_dir: str = "saved_results"
    show_plots: bool = False
    save_uuid_checkpoint: bool = True
    run_real_test: bool = True


def main(cfg: RunConfig = RunConfig()) -> None:
    train_dl, val_dl, test_dl = get_combined_dataloaders(
        ["single", "double"],
        cfg.snr,
        root=cfg.data_root,
        num_workers=cfg.num_workers,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        seed=cfg.split_seed,
    )

    model = build_model(ModelConfig())
    ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )

    opt, sched = get_opt_and_sched(model, train_dl, cfg.epochs)
    train_losses, val_losses, _eval_dl, val_accs, val_mses, trained_model = do_training(
        model, opt, sched, train_dl, val_dl, cfg.epochs, ema=ema
    )

    if cfg.save_uuid_checkpoint:
        states_dir = Path("states")
        states_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = states_dir / f"v1-combined-{uuid4()}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"[checkpoint] Saved: {ckpt_path}")

    plot_training_results(
        train_losses, val_losses, val_accs, val_mses,
        save_dir=cfg.save_dir, show=cfg.show_plots,
        val_acc_label="Val top-k recall",
    )

    from lib.data_safetensors import get_dataloaders
    from lib.testing import eval_on_loader_v1, do_real_test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_single = get_dataloaders("single", cfg.snr, root=cfg.data_root,
                                        num_workers=cfg.num_workers,
                                        eval_batch_size=cfg.eval_batch_size, seed=cfg.split_seed)
    _, _, test_double = get_dataloaders("double", cfg.snr, root=cfg.data_root,
                                        num_workers=cfg.num_workers,
                                        eval_batch_size=cfg.eval_batch_size, seed=cfg.split_seed)

    for label, dl in [("single", test_single), ("double", test_double)]:
        print(f"\n[test / {label}]")
        for k, v in eval_on_loader_v1(trained_model, dl, device).items():
            print(f"  {k}: {v:.4f}")

    if cfg.run_real_test:
        print("\n[real benchmark]")
        try:
            do_real_test(trained_model, device=device, print_result=True)
        except (FileNotFoundError, OSError) as e:
            print(f"  Skipping (missing benchmark .mat): {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SDINet v1 on single+double combined")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    main(RunConfig(epochs=args.epochs))
