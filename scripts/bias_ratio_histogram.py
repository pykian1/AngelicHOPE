
import argparse
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
 
from hope_implementation.capacity import relu_self_kernel
from hope_implementation.models import VGG8
 
# (ratio, error at rho=0.9, error at rho=0.5) 
_TABLE = [(0.00, 0.0, -0.2), (0.25, -1.5, -8.2), (0.50, -3.1, -14.7),
          (1.00, -5.1, -23.6), (2.00, -7.3, -32.6)]
 
 
def lookup_error(ratio, col):
    """Interpolate the measured error table. col: 1 -> rho=0.9, 2 -> rho=0.5."""
    xs = [t[0] for t in _TABLE]
    ys = [t[col] for t in _TABLE]
    return float(np.interp(ratio, xs, ys))
 
 
def out_norms(next_module, n_filters):
    """||w_out,i||_2 per filter, Appendix B.1.
 
    The outgoing vector for filter i is the slice of the DOWNSTREAM tensor on
    its input-channel axis: K[:, i, :, :] for a conv, W[:, i] for a linear.
    Either way the input-channel axis is dim 1, so norm over its complement.

    """
    w = next_module.weight.detach().cpu().double()
    dims = [d for d in range(w.dim()) if d != 1]
    norms = w.permute(1, *dims).reshape(w.shape[1], -1).norm(dim=1)
    assert norms.numel() == n_filters, (
        f"pairing mismatch: downstream expects {norms.numel()} input channels, "
        f"BN has {n_filters}")
    return norms
 
 
def collect(model):
    """(bn_name, gamma, beta, capacity) per BN layer, paired with its downstream layer."""
    # named_modules() yields definition order, which for nn.Sequential is
    # forward order. Pair by position in this flat list rather than by parsing
    # names -- module names are not uniformly dotted (e.g. "classifier").
    flat = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d))]
 
    packs = []
    for i, (bn_name, bn) in enumerate(flat):
        if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        nxt = next((m for _, m in flat[i + 1:]
                    if isinstance(m, (nn.Conv2d, nn.Linear))), None)
        if nxt is None:
            print(f"  (skipping {bn_name}: no downstream layer)")
            continue
 
        gamma = bn.weight.detach().cpu().double()
        beta = bn.bias.detach().cpu().double()
        w_norm = out_norms(nxt, gamma.numel())
        capacity = w_norm * torch.sqrt(relu_self_kernel(gamma, beta))
        packs.append((bn_name, gamma, beta, capacity))
    return packs
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/vgg8_baseline.pt")
    ap.add_argument("--keep", type=float, default=0.5,
                    help="fraction of filters, by capacity, treated as surviving")
    ap.add_argument("--gamma-floor", type=float, default=0.05,
                    help="inactive if |gamma| < floor * median|gamma| in that layer")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
 
    model = VGG8()
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
 
    packs = collect(model)
    print(f"\ncheckpoint: {args.checkpoint}   surviving = top {args.keep:.0%} by capacity\n")
    print(f"{'layer':<16} {'N':>5} {'dead':>5} {'kept':>5} | {'ALL':>17} | "
          f"{'ACTIVE + TOP-K':>17} | {'err@.9':>7}")
    print(f"{'':<16} {'':>5} {'':>5} {'':>5} | {'median':>8} {'p90':>8} | "
          f"{'median':>8} {'p90':>8} |")
    print("-" * 92)
 
    rows = []
    for name, gamma, beta, cap in packs:
        ratio = beta.abs() / gamma.abs().clamp_min(1e-12)
 
        # Appendix E.3.1 footnote: rho_hat is only defined for ACTIVE features,
        # ||w_eff||_2 > 0. Since w_eff = (gamma/sigma) w_raw, a collapsed gamma
        # means w_eff -> 0 and the pair geometry is undefined. Such a filter is
        # a constant-output unit, not a mergeable one -- and note that capacity
        # does NOT screen it out: relu_self_kernel(gamma->0, beta>0) -> beta^2,
        # so a collapsed filter with positive beta has real capacity.
        active = gamma.abs() > args.gamma_floor * gamma.abs().quantile(.9)
        n_dead = int((~active).sum())
 
        cap_a, ratio_a = cap[active], ratio[active]
        k = max(1, int(round(args.keep * len(cap_a))))
        surv = ratio_a[torch.topk(cap_a, k).indices]
 
        allr = ratio.clamp(max=1e3)
        err = lookup_error(surv.quantile(0.9).item(), 1)
        rows.append((name, ratio, surv, err))
        print(f"{name:<16} {len(cap):>5} {n_dead:>5} {k:>5} | "
              f"{allr.median():>8.2f} {allr.quantile(0.9):>8.2f} | "
              f"{surv.median():>8.2f} {surv.quantile(0.9):>8.2f} | {err:>6.1f}%")
 
    pooled = torch.cat([r[2] for r in rows])
    p90 = pooled.quantile(0.9).item()
    if p90 > _TABLE[-1][0]:
        print(f"\n  WARNING: p90 {p90:.3g} exceeds the measured table (max {_TABLE[-1][0]}).")
        print("  np.interp clamps at the last row, so the reported error is a floor,")
        print("  not an estimate. Investigate before trusting it.")
    print("-" * 92)
    print(f"{'POOLED':<16} {len(pooled):>5} {'':>5} {'':>5} | {'':>8} {'':>8} | "
          f"{pooled.median():>8.2f} {p90:>8.2f} | {lookup_error(p90, 1):>6.1f}%")
 
    print(f"\nAt the surviving p90 of {p90:.2f}, Eq. (5) carries")
    print(f"  ~{lookup_error(p90, 1):.1f}% error on merge-relevant pairs (rho_hat ~ 0.9)")
    print(f"  ~{lookup_error(p90, 2):.1f}% error at rho_hat ~ 0.5")
    e = abs(lookup_error(p90, 1))
    if e < 3:
        print("\n=> Approximation defensible. Document the number, skip Eq. (83).")
    elif e < 8:
        print("\n=> Borderline. Run the merge-ordering Spearman test before deciding.")
    else:
        print("\n=> Exact bivariate form warranted.")
 

    os.makedirs(args.out, exist_ok=True)
    ncol = 3
    nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.7 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
 
    bins = np.linspace(0, 2.5, 41)
    for ax, (name, ratio, surv, err) in zip(axes, rows):
        ax.hist(ratio.clamp(max=2.5).numpy(), bins=bins, color="#c8cfd8",
                label="all filters")
        ax.hist(surv.clamp(max=2.5).numpy(), bins=bins, color="#4c72b0",
                label=f"active, top {args.keep:.0%}")
        for t, _, _ in _TABLE[1:]:
            ax.axvline(t, color="#c44e52", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"{name}   surviving p90 {surv.quantile(0.9):.2f}  ({err:+.1f}%)",
                     fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(rows):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, frameon=False)
 
    fig.suptitle(r"$|\beta|/|\gamma|$ before and after capacity filtering"
                 "\n(annotation: Eq. (5) error at $\\hat{\\rho}=0.9$)", y=1.01)
    fig.supxlabel(r"$|\beta| / |\gamma|$   (clipped at 2.5)")
    fig.tight_layout()
    path = os.path.join(args.out, "bias_ratio_histogram_gamma-floor=.05.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {path}")
 
 
if __name__ == "__main__":
    main()