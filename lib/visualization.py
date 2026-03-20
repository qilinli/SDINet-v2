from __future__ import annotations

from pathlib import Path


def plot_training_results(
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float] | None = None,
    val_mses: list[float] | None = None,
    *,
    save_dir: str | Path = "saved_results",
    show: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot_training_results] Skipping plots (matplotlib import failed): {e}")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(train_losses, label="Train", linewidth=0.8)
    ax.plot(val_losses, label="Val", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curves.png", dpi=200)

    if val_accs:
        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.plot(val_accs, color="blue", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Acc (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "val_accuracy.png", dpi=200)

    if val_mses:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(val_mses, color="blue", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "val_mse.png", dpi=200)

    if show:
        plt.show()

