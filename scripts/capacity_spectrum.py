"""


Sorted capacity spectrum per layer

left panel: shows the aboslute neuron capacity log y
            layers sit at different heights which shows the scale bias that layer transition 
            costs serve to correct

right panel: same curves divided by each layer's max. strips scaleout so what is left is shape
            shows how fast capacity decays within a layer

"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from hope_implementation.models import load_frozen
from hope_implementation.state import build_state

CKPT="checkpoints/vgg8_baseline.pt"
FREE_CAP=1e-8
FLOOR=1e-12



def spectr(cs): #(name, sorted capacities decscending) per pack, live neurons only
    out = []
    for p, s in zip(cs.packs, cs.states):
        cap = s.cap[s.kept()]
        out.append((p.name, torch.sort(cap, descending=True).values))

    return out


def table(rows):
    print(f"\n{'layer':<18} {'N':>5} {'free':>5} {'max':>10} {'median':>10} "
            f"{'min>0':>10} {'decade':>7}")
    
    print("~"*70)

    totalN, totalFree = 0, 0

    for name, cap in rows:
        free = int((cap< FREE_CAP).sum())
        live = cap[cap>= FREE_CAP]

        span = float(torch.log10(live.max() / live.min())) if live.numel() else 0.0

        print(f"{name:<18} {len(cap):>5} {free:>5} {cap.max():>10.3e} "
              f"{cap.median():>10.3e} {live.min():>10.3e} {span:>7.2f}")

        totalN += len(cap)
        totalFree += free

    print("~"*70)
    print(f"{'TOTAL':<18} {totalN:>5} {totalFree:>5}   "
          f"-> predicted density after free prune "
          f"{(totalN - totalFree) / totalN:.4f}")






def plot(rows, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4.5))
    cmap = plt.cm.viridis(torch.linspace(0, .9, len(rows)).numpy())


    for (name, cap), c in zip(rows, cmap):
        x = torch.linspace(0,1, len(cap))
        y=cap.clamp_min(FLOOR)
        ax1.plot(x, y, color=c, lw=1.6, label=name)
        ax2.plot(x, y/y.max(), color=c, lw=1.6)


    ax1.axhline(FREE_CAP, ls="--", c=".4", lw=1)
    ax1.text(.02, FREE_CAP*1.6, f"free prune cutoff {FREE_CAP:.0e}", fontsize=8, color="0.3")
    ax1.set_yscale("log")
    ax1.set_xlabel("normalized neuron ranke within layer")
    ax1.set_ylabel(r"capacity $\|f_i\|_\mathcal{H}$")
    ax1.set_title("absolute capacity")
    ax1.legend(fontsize=7, frameon=False)
    ax2.set_yscale("log")
    ax2.set_xlabel("neuron rank within layer (normalised)")
    ax2.set_ylabel(r"$\|f_i\| \; / \; \max_j \|f_j\|$")
    ax2.set_title("scale-normalised (shape only)")
 
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
 


    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=CKPT)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model=load_frozen(args.checkpoint, device=torch.device("cpu"))

    cs=build_state(model)

    print(f"checkpoints: {args.checkpoint}")
    print(f"packs {len(cs.packs)} neurons {cs.n0_total} "
        f"density {cs.density():.4f}")

    rows = spectr(cs)
    table(rows)

    stem=os.path.splitext(os.path.basename(args.checkpoint))[0]

    plot(rows, os.path.join(args.out, f"capacity_spectrum_{stem}.png"))


if __name__ == "__main__":
    main()