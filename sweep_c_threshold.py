"""
Quick sweep: is_obj threshold for C-head on Qatar double-damage.

Instead of pure argmax (active iff argmax ≠ ∅), activate a slot when
    is_obj[b,k] = 1 - P(∅)[b,k] > θ
and sweep θ ∈ [0.05, 0.95].

Also evaluates single-damage test set at each θ so we can see the
single/double tradeoff.

Usage:
    python sweep_c_threshold.py --c states/qatar/c-combined-<uuid>.pt
"""

import argparse
import torch
import numpy as np
from lib.model import ModelConfig, ModelConfigC, load_model_c_from_checkpoint
from lib.datasets import get_dataset
from lib.metrics import f1_from_counts, PRESENCE_NORM_THRESH

CKPT = "states/qatar/c-combined-22d0db27-e316-441a-90ff-121a5d7e6405.pt"


@torch.inference_mode()
def collect_outputs(model, dl, device):
    L, S, Y = [], [], []
    for x, y in dl:
        l, s = model(x.to(device))
        L.append(l.cpu()); S.append(s.cpu()); Y.append(y.cpu())
    return torch.cat(L), torch.cat(S), torch.cat(Y)


def eval_at_threshold(loc_logits, y_norm, theta):
    """Evaluate set-level F1, recall, mean_k_pred at a given is_obj threshold."""
    is_obj   = 1.0 - loc_logits.softmax(-1)[..., -1]   # (N, K)
    pred_loc = loc_logits[..., :-1].argmax(-1)          # (N, K)
    active   = is_obj > theta                           # (N, K) bool

    y_pres = y_norm > PRESENCE_NORM_THRESH

    tp = fp = fn = 0
    for b in range(y_norm.size(0)):
        pred_set = set(pred_loc[b, active[b]].tolist())
        true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    f1, prec, rec = f1_from_counts(tp, fp, fn)
    mean_k_pred = active.float().sum(-1).mean().item()
    mean_k_true = y_pres.float().sum(-1).mean().item()
    return f1, prec, rec, mean_k_pred, mean_k_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default=CKPT)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset("qatar")

    model = load_model_c_from_checkpoint(args.c, device=device)
    model.eval()

    dl_kwargs = dict(
        root=dataset.default_root, num_workers=0, eval_batch_size=64,
        seed=42, train_batch_size=64, snr=-1.0,
        window_size=args.window_size, overlap=args.overlap,
        downsample=args.downsample,
    )

    # Single-damage test set
    print("Collecting single-damage test outputs...")
    (_, single_dl), = [(lbl, dl) for lbl, dl in dataset.get_test_loaders(**dl_kwargs)]
    L_s, S_s, Y_s = collect_outputs(model, single_dl, device)

    # Double-damage test set
    print("Collecting double-damage test outputs...")
    double_dl = dataset.get_extra_test_loader(**dl_kwargs)
    L_d, S_d, Y_d = collect_outputs(model, double_dl, device)

    thetas = np.round(np.arange(0.05, 1.00, 0.05), 2)

    hdr = f"{'θ':>6}  {'single':^38}  {'double':^38}"
    sub = f"{'':>6}  {'f1':>7} {'prec':>7} {'rec':>7} {'k_pred':>7} {'k_true':>7}  {'f1':>7} {'prec':>7} {'rec':>7} {'k_pred':>7} {'k_true':>7}"
    print("\n" + hdr)
    print(sub)
    print("-" * len(sub))

    # argmax baseline (θ=0 equivalent but using actual argmax logic)
    from lib.metrics import _c_slot_decode
    active0, pred_loc0, _ = _c_slot_decode(L_s)
    y_pres_s = Y_s > PRESENCE_NORM_THRESH
    tp=fp=fn=0
    for b in range(Y_s.size(0)):
        ps = set(pred_loc0[b, active0[b]].tolist())
        ts = set(y_pres_s[b].nonzero(as_tuple=False)[:,0].tolist())
        tp+=len(ps&ts); fp+=len(ps-ts); fn+=len(ts-ps)
    f1_s0,pr_s0,re_s0 = f1_from_counts(tp,fp,fn)
    k_s0 = active0.float().sum(-1).mean().item()

    active0d, pred_loc0d, _ = _c_slot_decode(L_d)
    y_pres_d = Y_d > PRESENCE_NORM_THRESH
    tp=fp=fn=0
    for b in range(Y_d.size(0)):
        ps = set(pred_loc0d[b, active0d[b]].tolist())
        ts = set(y_pres_d[b].nonzero(as_tuple=False)[:,0].tolist())
        tp+=len(ps&ts); fp+=len(ps-ts); fn+=len(ts-ps)
    f1_d0,pr_d0,re_d0 = f1_from_counts(tp,fp,fn)
    k_d0 = active0d.float().sum(-1).mean().item()
    print(f"{'argmax':>6}  {f1_s0:>7.4f} {pr_s0:>7.4f} {re_s0:>7.4f} {k_s0:>7.3f} {Y_s.gt(PRESENCE_NORM_THRESH).float().sum(-1).mean():>7.3f}  "
          f"{f1_d0:>7.4f} {pr_d0:>7.4f} {re_d0:>7.4f} {k_d0:>7.3f} {Y_d.gt(PRESENCE_NORM_THRESH).float().sum(-1).mean():>7.3f}")

    for theta in thetas:
        f1_s, pr_s, re_s, kp_s, kt_s = eval_at_threshold(L_s, Y_s, theta)
        f1_d, pr_d, re_d, kp_d, kt_d = eval_at_threshold(L_d, Y_d, theta)
        print(f"{theta:>6.2f}  {f1_s:>7.4f} {pr_s:>7.4f} {re_s:>7.4f} {kp_s:>7.3f} {kt_s:>7.3f}  "
              f"{f1_d:>7.4f} {pr_d:>7.4f} {re_d:>7.4f} {kp_d:>7.3f} {kt_d:>7.3f}")


if __name__ == "__main__":
    main()
