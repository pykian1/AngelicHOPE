
import argparse
import math
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
 
from hope_implementation.bivariate import c_k_exact
from hope_implementation.capacity import relu_self_kernel
from hope_implementation.reference import mc_cross
 
DTYPE = torch.float64
GAMMA_I, GAMMA_J = 1.3, 0.7
RATIOS = (0.0, 0.25, 0.5, 1.0, 2.0)
RHOS = (0.0, 0.5, 0.9)
 
 
def approx(gamma_i, beta_i, gamma_j, beta_j, rho_hat):

    g = torch.tensor([gamma_i, gamma_j], dtype=DTYPE)
    b = torch.tensor([beta_i, beta_j], dtype=DTYPE)
    k = relu_self_kernel(g, b)
    r = max(-1.0, min(1.0, rho_hat))
    bracket = (1 / math.pi) * (
        math.sqrt(max(0.0, 1 - r * r)) + (math.pi - math.acos(r)) * r
    )
    return bracket * math.sqrt(k[0].item() * k[1].item())
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
 
    print(f"gamma_i={GAMMA_I}, gamma_j={GAMMA_J}, {args.samples:,} MC samples\n")
    print(f"{'|b|/|g|':>8} {'rho':>6} {'monte carlo':>13} {'exact (83)':>13} "
          f"{'err':>9} | {'approx (5)':>12} {'err':>9}")
    print("-" * 80)
 
    grid = np.zeros((len(RATIOS), len(RHOS)))
    for a, ratio in enumerate(RATIOS):
        for c, rho in enumerate(RHOS):
            bi, bj = ratio * GAMMA_I, ratio * GAMMA_J
            mc, _ = mc_cross(GAMMA_I, bi, GAMMA_J, bj, rho, m=args.samples)
            ex = c_k_exact(GAMMA_I, bi, GAMMA_J, bj, rho).item()
            ap_ = approx(GAMMA_I, bi, GAMMA_J, bj, rho)
            grid[a, c] = 100 * (ap_ - mc) / mc
            print(f"{ratio:>8.2f} {rho:>6.2f} {mc:>13.6f} {ex:>13.6f} "
                  f"{(ex - mc) / mc:>+8.3%} | {ap_:>12.6f} {(ap_ - mc) / mc:>+8.2%}")
        print()
 
    print("Negative = the approximation UNDERestimates K(i,j).")
    print("Merging only ever selects high-rho pairs, so the rho=0.9 column is")
    print("the one that governs merge decisions; rho=0 is a regime the greedy")
    print("loop never samples from.")
 
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, rho in enumerate(RHOS):
        ax.plot(RATIOS, grid[:, c], "o-", label=rf"$\hat{{\rho}} = {rho}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$|\beta| / |\gamma|$")
    ax.set_ylabel("Eq. (5) relative error (%)")
    ax.set_title("Zero-bias approximation error vs bias ratio")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out, "approx_vs_exact.png")
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")
 
 
if __name__ == "__main__":
    main()