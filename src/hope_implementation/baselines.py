"""

Structured magnitude baseliens found in section 11.1

        L1-Input : ||w_in, i||_l over raw incoming kernel
        L1-Join: ||w_in, i||_l + ||w_out, i||_l, concat form
        BN-Scale: ||gamma_i|| 


    
section 11 describes these baselines as a single global cutoff on a static score

our implementation using the _sync_upstream method and _sync_downstream mutate w_raw and w_out as pruning proceeds
recomputing a score at each step would silently upgrade the baseline into an iterative method the paper does not described

we implement frozen_scores() to remain faithful to the paper
    
    
    
"""

from __future__ import annotations
import torch

from .state import Action, CompressionState, LayerPack
BASELINES = ("L1-input", "L1-joint", "BN-scale")



def _score(pack: LayerPack, kind: str) -> torch.Tensor:
    if kind == "L1-input":
        return pack.w_raw.abs().sum(dim=1)
    elif kind == "L1-joint":
        return pack.w_raw.abs().sum(dim=1)+pack.w_out.reshape(pack.n, -1).abs().sum(dim=1)
    elif kind == "BN-scale":
        return pack.gamma.abs()
    else:
        raise ValueError(f"unknown baseline")


def frozen_scores(cs: CompressionState, kind: str) -> list[torch.Tensor]: #snapshot a static magnitude score per unit, per layer
    return [_score(p, kind).clone() for p in cs.packs]

def _cheapest(cs: CompressionState, li: int, score: torch.Tensor):
    idx = cs.states[li].kept()
    if idx.numel() == 0:
        return None
    s = score[idx]
    k = int(torch.argmin(s))
    return Action("prune", li, int(idx[k]), j_cost=float(s[k]), dp=1.0)

def static_generator(scores: list[torch.Tensor]): # wrap a frozen score as a step() generator

    def gen(cs: CompressionState, li: int):
        return _cheapest(cs, li, scores[li])

    return gen

def live_generator(kind: str): #recompute the magnitude score from current pack state at every step
    def gen(cs: CompressionState, li: int):
        return _cheapest(cs, li, _score(cs.packs[li], kind))
    return gen

