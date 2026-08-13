"""





SAME AS OTHER DENSITY SWEEP JUST MIN WIDTH = 2




"""

from __future__ import annotations
import argparse
import csv
import os
import time
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from hope_implementation.models import VGG8, get_device, load_frozen
from hope_implementation.state import EPS, Action, CompressionState, best_prune, execute_prune, step, write_back, build_state

CKPT="checkpoints/vgg8_baseline.pt"
DATA=os.path.expanduser("~/data/cifar10")


#CIFAR10 EMPIRCAL MEAN + STD 
MEAN=(.4914, .4822, .4465)
STD=(.2470, .2435, .2616)
BASELINES = ("L1-input", "L1-joint", "BN-scale")


def active_mask(cs: CompressionState, gamma_floor: float=.05, w_floor: float=1e-12) -> list[torch.Tensor]: #per layer bool masks of units with defined HOPE geometry
    masks= []
    for p in cs.packs:
        g = p.gamma.abs()
        thr=gamma_floor * g.quantile(0.9)
        w_norm = p.w_eff.reshape(p.n, -1).norm(dim=1)
        masks.append((g>thr) & (w_norm>w_floor))
    return masks

def free_prune(cs: CompressionState, masks: list[torch.Tensor]) -> int: #remove every unit outside active set
    n= 0
    for li, m in enumerate(masks):
        for i in torch.nonzero(~m, as_tuple=False).flatten().tolist():
            if bool(cs.states[li].alive[i]):
                execute_prune(cs, Action("prune", li, i))
                n+=1
    cs.history.clear() # freep rune is not part of any method's trajectory
    return n

def density_p90(cs: CompressionState, masks: list[torch.Tensors], n_active0: int)-> float:
    alive = sum(int((s.alive & m).sum()) for s, m in zip(cs.states, masks))
    return alive / n_active0


#score sources
def frozen_scores(cs: CompressionState, kind: str) -> list[torch.Tensor]: #snapshot a static magnitude score per unit, per layer
    """L1-Input : ||w_in, i||_l over raw incoming kernel
        L1-Join: ||w_in, i||_l + ||w_out, i||_l, concat form
        BN-Scale: ||gamma_i|| """

    
    out = []
    for p in cs.packs:
        if kind == "L1-input":
            s = p.w_raw.abs().sum(dim=1)
        elif kind == "L1-joint":
            s=p.w_raw.abs().sum(dim=1) + p.w_out.reshape(p.n, -1).abs().sum(dim=1)
        elif kind == "BN-scale":
            s = p.gamma.abs()
        else:
            raise ValueError("unknown baseline")
        out.append(s.clone())

    return out


def static_generator(scores: list[torch.Tensor]): # wrap a frozen score as a step() generator

    def gen(cs: CompressionState, li: int):
        idx = cs.states[li].kept()
        if idx.numel() == 0: 
            return None
        s = scores[li][idx]
        k = int(torch.argmin(s))

        return Action("prune", li, int(idx[k]), j_cost = float(s[k]), dp=1.0)
    return gen



def test_loader(batch_size: int = 256) -> DataLoader:
    tf =v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD)

    ])

    ds = datasets.CIFAR10(root=DATA, train=False, transform=tf, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True) 

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(X).argmax(1) == y).sum().item()

    return correct / len(loader.dataset)

# ----- sweep



def run_method(name: str, ckpt: str, targets: np.ndarray, loader: DataLoader, device, use_merge: bool, verbose: bool) -> list[dict]:
    # one progressive trajectory, sampled at each target density, progressive means one trajectory per method not a fresh run per target**


    model = load_frozen(ckpt, device=torch.device("cpu"))
    cs = build_state(model)
    model.to(device)
    masks = active_mask(cs)
    n_active0 = int(sum(int(m.sum()) for m in masks))
    n_free = free_prune(cs, masks)


    if name == "HOPE":
        gens = (best_prune,)
        min_width = 2
        if use_merge:
            from hope_implementation.merge import best_merge
            gens = (best_prune, best_merge)
            min_width = 2
    else:
        gens=(static_generator(frozen_scores(cs, name)),)
        min_width = 2
    if verbose:
        print(f"\n{name}: free_pruned {n_free} inactive units ,"
            f"active set = {n_active0} / {cs.n0_total} "
            f"(paper density {cs.density():.4f})")


    rows, t0, first = [], time.time(), True
    for t in targets:
        stalled=False
        while density_p90(cs, masks, n_active0) > t:
            if step(cs, generators=gens, min_width=min_width) is None:
                stalled=True
                break
        write_back(cs, check=first)
        first=False
        d90 = density_p90(cs, masks, n_active0)
        acc = accuracy(model, loader, device)

        n_merge = sum(a.kind == "merge" for a in cs.history)
        rows.append({
            "method": name,
            "density_p90": round(d90, 6),
            "density_paper": round(cs.density(), 6),
            "accuracy": round(acc, 6),
            "n_actions": len(cs.history),
            "n_merges": n_merge
        })

        if verbose:
            print(f" d90 {d90:.4f} | paper {cs.density():.4f} | "
                f"acc {100*acc:6.2f}% | {len(cs.history):5d} actions "
                f"({n_merge} merges)")
        if stalled:
            print(f" ! {name} stalled at d90 = {d90:.4f}: no action left"
                f"(every layer at min_width={min_width})")
            break
    if verbose:
        print(f" {name} done in {(time.time()-t0)/60:.1f} min")
    return rows



def plot(rows: list[dict], out_path: str, baseline_acc: float) -> None:
    methods = ["HOPE", *BASELINES]
    colors = {"HOPE": "#c44e52", "L1-input": "#4c72b0",
              "L1-joint": "#55a868", "BN-scale": "#8172b2"}
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        if not r:
            continue
        style = dict(color=colors[m], lw=2.0 if m == "HOPE" else 1.4,
                     marker="o", ms=3, label=m, zorder=3 if m == "HOPE" else 2)
        ax1.plot([x["density_p90"] for x in r],
                 [100 * x["accuracy"] for x in r], **style)
        ax2.plot([x["density_paper"] for x in r],
                 [100 * x["accuracy"] for x in r], **style)
 
    for ax in (ax1, ax2):
        ax.axhline(100 * baseline_acc, ls="--", c="0.4", lw=0.9)
        ax.axhline(10.0, ls=":", c="0.6", lw=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("CIFAR-10 test accuracy (%)")
 
    ax1.text(0.02, 100 * baseline_acc + 0.8,
             f"uncompressed {100*baseline_acc:.2f}%", fontsize=8, color="0.3",
             transform=ax1.get_yaxis_transform())
    ax1.set_xlabel(r"density over active set   $|alive \cap A| \, / \, |A|$")
    ax1.set_title("p90-active density (common start point)")
    ax2.set_xlabel("density (paper definition: active / initial)")
    ax2.set_title("paper definition")
    ax1.legend(frameon=False, fontsize=9)
    ax1.invert_xaxis()
    ax2.invert_xaxis()
 
    fig.suptitle("Accuracy vs density, VGG-8 / CIFAR-10, no fine-tuning (PRUNING ONLY MIN WIDTH = 2)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nwrote {out_path}")
 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=CKPT)
    ap.add_argument("--min-density", type=float, default=0.30)
    ap.add_argument("--step", type=float, default=0.02, help="density grid spacing (0.02 -> 36 points to 0.30)")
    ap.add_argument("--merge", action="store_true", help="run hope with the full prune+merge generator set")
    ap.add_argument("--methods", default="HOPE," + ",".join(BASELINES))
    ap.add_argument("--out", default="logs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device= get_device()
    loader=test_loader()

    base=load_frozen(args.checkpoint, device=device)
    baseline_acc = accuracy(base, loader, device)
    targets = np.arange(1.0, args.min_density - 1e-9, -args.step)
    print(f"checkpoint {args.checkpoint} || device {device}")
    print(f"uncompressed test acc {100*baseline_acc:.2f}%")

    del base
    rows: list[dict] = []


    for name in [m.strip() for m in args.methods.split(",") if m.strip()]:
        rows += run_method(name, args.checkpoint, targets, loader, device, args.merge, verbose=True)

    suffix = ("_merge" if args.merge else "")
    csv_path=os.path.join(args.out, f"density_sweep{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {csv_path}")

    plot(rows, os.path.join(args.out, f"accuracy_vs_density_min_width_2.png"), baseline_acc)


if __name__ == "__main__":
    main()