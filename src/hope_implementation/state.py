"""


state & topology layer for HOPE  scoped for pruning+merging on our VGG8 

LayerPack: struct + params for one prunable layer

LayerState: what changes as you compress; alive mask, capacities, remaining layer capacity, number of active neurons

CompressionState: packs + sates + topology + pair caches

Action: the unit the greedy looop selects over







"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

from .capacity import relu_self_kernel

EPS=1e-12 # spec in alg1 of HOPE paper
Kind = Literal["prune", "merge"]

RECACHE_ON_NEIGHBOUR_CHANGE = False

@dataclass
class LayerPack:
    """
    (1 prunable layer: (conv/linear)) -> bn -> relu-> next
    physical parameters are source and every tensor is a clone
    effective weights + biases are derived by refold() and must be regenerated upon raw weight changes 
    """


    idx: int
    name: str
    kind: Literal["conv", "linear"]

    layer: nn.Module
    bn: nn.Module
    next_layer: nn.Module

    n0: int #init neuron count
    k_out: int #kh * kw of next_layer or 1 if lin
    bn_eps: float


    #raw parameters

    w_raw: torch.Tensor # [N, d_in] d_in = fan in of one neuron
    gamma: torch.Tensor # [N]
    beta: torch.Tensor # [N]
    mu: torch.Tensor # [N] running mean
    var: torch.Tensor # [N] running var
    w_out: torch.Tensor # [N, n_next, k_out] neuron layer, next layers output channel, next layers receptive field

    #derived from paper section 3
    w_eff: torch.Tensor = field(init=False) # in our use case: 
    b_eff: torch.Tensor = field(init=False)


    dp: torch.Tensor = field(init=False) # [N] static deltaP^init
    e0: float = field(init=False) # initial uncompressed capacity

    prev: Optional[int] = None
    next: Optional[int] = None

    def __post_init__(self) -> None:
        self.w_eff = torch.empty_like(self.w_raw)
        self.b_eff = torch.empty_like(self.beta)
        self.refold() 
        d_in = self.w_raw.shape[1] # fan in of one neuron
        c = self.w_out.shape[1] * self.k_out #fan out of one neuron
        self.dp = torch.full((self.n0,), float(d_in + c+ 4))

        #unused for pruning/merging used for ploting E_active / e0 per layer
        self.e0 = float(
            (
                self.w_out.reshape(self.n0, -1).norm(dim=1) * 
                torch.sqrt(relu_self_kernel(self.gamma, self.beta).clamp_min(0.0))
            ).sum()
        )


    def refold(self, rows: Optional[torch.Tensor]=None) -> None:
        # effecitve weight = (gamma / rad(var + eps)) * w_raw | effecitive bias = beta - scale*mu

        if rows is None:
            rows = torch.arange(self.n0)
        scale = self.gamma[rows] / torch.sqrt(self.var[rows] + self.bn_eps)
        self.w_eff[rows] = scale[:, None] * self.w_raw[rows]
        self.b_eff[rows] = self.beta[rows] - scale*self.mu[rows]


    def augmented(self) -> torch.Tensor:
        # w~_in = [w_eff, b] in section 7, shape = [N, d_in+1]
        return torch.cat([self.w_eff, self.b_eff[:,None]], dim=1)

    @property
    def n(self) -> int:
        return self.n0 #


@dataclass
class LayerState:
    alive: torch.Tensor #[N]
    out_norm: torch.Tensor = field(init=False) #[N]
    k_self: torch.Tensor = field(init=False) #[N]
    cap: torch.Tensor = field(init=False) #[N]
    e_rem: float = field(init=False, default=0.0)
    n_active: int = field(init=False, default=0)


    @classmethod

    def build(cls,pack:LayerPack) -> "LayerState":
        st = cls(torch.ones(pack.n,dtype= torch.bool))
        st.refresh(pack)
        return st

    def refresh(self,pack:LayerPack) -> None:
        self.k_self= relu_self_kernel(pack.gamma,pack.beta)
        self.out_norm = pack.w_out.reshape(pack.n,-1).norm(dim=1)
        cap = self.out_norm * torch.sqrt(self.k_self.clamp_min(0.0))
        cap[~self.alive] = 0.0
        self.cap = cap
        self.e_rem = float(cap[self.alive].sum())
        self.n_active = int(self.alive.sum())

    def kept(self) -> torch.Tensor:
        #surviving channel indices
        return torch.nonzero(self.alive,as_tuple=False).squeeze(1)


@dataclass 
class PairCache:
    """ per layer merge constants seee appendix b.4

    u* and v* are not stored, recomputed just in time for winning pair only
    both are [N,N] dead rows/cols go unread
    """

    b: torch.Tensor # [N,N]
    rho_hat: torch.Tensor # [N,N]
    dirty: bool =False

@dataclass
class Action: #unit optimizer selects over
    kind: Kind
    layer: int
    i: int #survivor: pruned or parent slot on merge
    j: Optional[int] = None #merge child j > i always
    j_cost: float = 0.0 # Cost
    dp: float=1.0 #deltaP^init

    @property
    def dr(self) -> float:
        #distortation rate: J/dp
        return self.j_cost / max(self.dp, EPS)


@dataclass 
class CompressionState: # network
    model: nn.Module
    packs: list[LayerPack]
    states: list[LayerState]
    by_layer: dict[int, int] #id(module) -> pack index
    n0_total: int
    pairs: dict[int, PairCache] = field(default_factory=dict)
    history: list[Action] = field(default_factory=list)

    def density(self) -> float:
        # active/initial neurons across network
        return sum(s.n_active for s in self.states) / self.n0_total

    def refresh_all(self) -> None:
        for p, s in zip(self.packs, self.states):
            s.refresh(p)



# ---------------------------- BUILD

def _read_out(nxt: nn.Module, n:int) -> tuple[torch.Tensor, int]: #read outoging weights
    """
    outgoing weights [N, n_next, k_out] 
    conv2d next: [c_out2, N, kh, kw] -> [N, c_out2, kh*kw]
    linear next: [out, N] ->            [N, out, 1]
    every downstream slice is w_out[:, i, :]
    """
    w = nxt.weight.detach()
    if isinstance(nxt, nn.Conv2d):
        c2, _, kh, kw = w.shape
        return w.permute(1, 0, 2, 3).reshape(n, c2, kh*kw).clone(), kh*kw
    if isinstance(nxt, nn.Linear):
        return w.t().reshape(n, -1, 1).clone(), 1


def _write_out(nxt: nn.Module, w_out: torch.Tensor) -> torch.Tensor: #write outgoing weights
    """ inverse of _read_out"""
    n = w_out.shape[0]
    if isinstance(nxt, nn.Conv2d):
        c2 = w_out.shape[1]
        kh, kw = nxt.weight.shape[2:]
        return w_out.reshape(n, c2, kh, kw).permute(1, 0, 2, 3).contiguous()

    return w_out.reshape(n, -1).t().contiguous()


def build_state(model: nn.Module) -> CompressionState:
    """
    builds packs, states, + topology from frozen model
    assumes VGG8 triples appear in execution order, pool + flatten ignored
    """

    named = [
        (n,m) for n, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d))
    ]

    packs: list[LayerPack] = []
    for t in range(len(named) - 2):
        (name, a), (_, bn), (_, nxt) = named[t], named[t+1], named[t+2]
        if not isinstance(a, (nn.Conv2d, nn.Linear)):
            continue
        if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        n = a.weight.shape[0]

        w_out, k_out = _read_out(nxt, n)

        packs.append(
            LayerPack(
                idx=len(packs),
                name=name,
                kind="conv" if isinstance(a, nn.Conv2d) else "linear",
                layer=a,
                bn=bn,
                next_layer=nxt,
                n0=n,
                k_out=k_out,
                bn_eps=bn.eps,
                w_raw=a.weight.detach().reshape(n,-1).clone(),
                gamma=bn.weight.detach().clone(),
                beta=bn.bias.detach().clone(),
                mu=bn.running_mean.detach().clone(),
                var=bn.running_var.detach().clone(),
                w_out=w_out,
            )
        )

        # topology by identity 

        by_layer={id(p.layer): p.idx for p in packs}
        for p in packs:
            q = by_layer.get(id(p.next_layer))
            if q is not None:
                p.next=q
                packs[q].prev = p.idx

    return CompressionState(model=model, packs=packs, states=[LayerState.build(p) for p in packs], by_layer=by_layer, n0_total =sum(p.n0 for p in packs))


def _sync_upstream(cs: CompressionState, li: int, channel: int) -> None:
    """
    layer lis raw incoming weights for channel are upstream layers outgoing weights for that channel, pushes 
    current value up

    w_raw[channel] = [d_in] = n_up * k_out_up and reshapes exactly onto up.w_out[:, channel, :] = [n_up, k_out_up]

    call after any write to w_raw 
    never mentioned in paper but pruning neuron i of layer l shrinks ||w_out, k|| for every k in 
    layer l-1, so upstream capacities can go stale without it 
    """

    p = cs.packs[li]
    if p.prev is None:
        return
    up = cs.packs[p.prev]
    up.w_out[:, channel, :] = p.w_raw[channel].reshape(up.n, up.k_out)
    cs.states[p.prev].refresh(up)

    if RECACHE_ON_NEIGHBOUR_CHANGE and p.prev in cs.pairs:
        cs.pairs[p.prev].dirty=True


def _kill(cs: CompressionState, li: int, i: int) -> None:
    # zero every parameter of neuron i and drop it from alive mask

    p, s = cs.packs[li], cs.states[li]

    p.w_raw[i] = 0.0
    p.gamma[i] = 0.0
    p.beta[i] = 0.0
    p.mu[i] = 0.0
    p.var[i] = 1.0
    p.w_out[i] = 0.0
    p.refold(torch.tensor([i]))
    s.alive[i] = False
    _sync_upstream(cs, li, i)
    s.refresh(p)


# --- pair cache
def build_pair_cache(cs: CompressionState, li: int) -> PairCache:
    """
    wip ref eq 10

    b[i,j] needs:
     per pair the principal eigenvector of A restrcited to its rank2 subspace (eq14)
     +- u (eq11) K = cross_kernel_m
     b[i,j] = || sum_k K(u_c, w~_in_k) w_out_k || / rad(K(u_c, u_c))
     batch rank 2 singular value decomp over all airs

    """

    raise NotImplementedError

def refresh_pair_row(cs: CompressionState, li: int, i: int) -> None:
    """
    update after merge alg 1 26-29

    recopmute row & col i of b and rho-hat against every live neighbour
    """

    raise NotImplementedError



# CANDIDATE GENERATION FOR ACTION PER LAYER

def best_prune(cs: CompressionState, li: int) -> Optional[Action]:
    """
    Cheapest prune in current layer
    J = N* ||f_i|| / (E_a - ||f_i||) 

    dp static, N_Active and E_rem are live, 
    
    """
    p,s = cs.packs[li], cs.states[li]
    idx = s.kept()
    if idx.numel() == 0:
        return None
    cap = s.cap[idx]
    j = s.n_active*cap/(s.e_rem - cap).clamp_min(EPS)
    dr = j/p.dp[idx]
    k = int(torch.argmin(dr))
    return Action("prune", li, int(idx[k]), j_cost=float(j[k]), dp=float(p.dp[idx[k]]))



def best_merge(cs: CompressionState, li: int) -> Optional[Action]:
    """ Cheapest merge in curr layer

    eq 6 of paper w appen B.4 consts

    a = || f_i||^2 + ||f_j||^2      (live from capacity )
    b = <psi*, f_i + f_j>   (cached)
    E_rem = E_a - ||f_i|| - ||f_j||

    s* = (a+ b*E_rem) / (2*E_rem + b)   eq 40
    J = N rad(2 s*^2 - 2 b s* + a) / (E_rem + s*)
    """
    pass




def execute_prune(cs: CompressionState, act: Action) -> None:
    """ projects neuron i to null op"""
    if not bool(cs.states[act.layer].alive[act.i]):
        raise ValueError(f"{cs.packs[act.layer].name}[{act.i}] already pruned")
    _kill(cs, act.layer, act.i)
    cs.history.append(act)



def execute_merge(cs: CompressionState, act: Action) -> None:
    """collapse pair into rank 1 parent written into slot i"""

    raise NotImplementedError




#  ---- progressive encoding loop (greedy loop)



Generator = Callable[[CompressionState,int], Optional[Action]]
_EXEC = {"prune":execute_prune, "merge":execute_merge}

def step(cs:CompressionState, generators: tuple[Generator, ...]= (best_prune,), min_width: int=1,) -> Optional[Action]:
    #scan all actions and execute only the argmin. 

    if best_merge in generators:
        min_width = max(min_width,2)

    best: Optional[Action] =None

    for li,s in enumerate(cs.states):
        if s.n_active <= min_width:
            continue
        for gen in generators:
            a=gen(cs,li)
            if a is not None and (best is None or a.dr<best.dr):
                best=a
    if best in None:
        return None
    _EXEC[best.kind](cs,best)
    return best


def compress_to_density(cs:CompressionState, target:float, **kw) -> list[Action]: #step makes one decision, this function is deciding how many decisions to make
    while cs.density()> target:
        if step(cs,**kw) is None:
            break
    return cs.history





# -- writeback


@torch.no_grad()
def write_back(cs: CompressionState, check: bool = True) -> None:
    """
    push back params onto real modules, uniform over prune and merge (pack carries physical params)

    masking not slicing

    every intermediate conv2d weight is described as packs[l].w_raw & _write_out(packs[l-1].w_out)
    these agree iff _sync_upstream is correct
    """

    for p in cs.packs:
        w_from_out = _write_out(p.next_layer, p.w_out)
        if check and p.next is not None:
            down = cs.packs[p.next]
            assert torch.allclose(
                w_from_out.reshape(down.n, -1), down.w_raw, atol=1e-5,
            ), f"upstream sync broken between {p.name} & {down.name}"

            
        p.layer.weight.copy_(p.w_raw.reshape(p.layer.weight.shape))
        p.bn.weight.copy_(p.gamma)
        p.bn.bias.copy_(p.beta)
        p.bn.running_mean.copy_(p.mu)
        p.bn.running_var.copy_(p.var)
        p.next_layer.weight.copy_(w_from_out)












