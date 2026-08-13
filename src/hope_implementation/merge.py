from __future__ import annotations
 
from typing import Optional
 
import torch
 
from .state import CompressionState, PairCache, Action, EPS, _kill, _sync_upstream, _sync_downstream
from .capacity import relu_self_kernel
from .kernel import bracket, warp
 
_F64 = torch.float64

def _pair_direction(aii,aij,ajj,oii,oij,ojj):
    """
    Gin = [[aii,aij], [aij, ajj]]
    Gout = [[oii,oij], [oij,ojj]]
    with u = Wc: ||Au||2 = c_T Gin Gout Gin c  and  ||u||2 = c_T Gin c
    u_hat = argmax_{||u||=1} ||(w_out_i w~_i^T + w_out_j w~_j^T) u||   (Eq 14)

    The optimum lies in span{w~_i, w~_j} (that is A's row space), so instead of
    an SVD we solve for coefficients u = c1 w~_i + c2 w~_j.

     Everything is elementwise over broadcastable tensors of pair scalars, so
    one call solves Eq 14 for a whole N x N mesh at once (the B.4 cache build)
    or for a single pair (JIT execution). Returns (c1, c2) with ||u|| = 1.
    """

    tiny = 1e-30
    tr = aii +ajj
    det = (aii*ajj - aij*aij).clamp_min(0.0)

    #closed-form 2x2 sqrtm: S = (Gin + sqrt(det) I) / sqrt(tr + 2 sqrt(det))
    sdet = torch.sqrt(det)
    denom = torch.sqrt((tr + 2.0 * sdet).clamp_min(tiny))
    s11 = (aii+sdet)/denom
    s12 = (aij)/denom
    s22 = (ajj+sdet)/denom

    #B = S Gout S (symmetric 2x2)

    t11 = s11 * oii + s12 * oij
    t12 = s11 * oij + s12 * ojj
    t21 = s12 * oii + s22 * oij
    t22 = s12 * oij + s22 * ojj
    b11 = t11 * s11 + t12 * s12
    b12 = t11 * s12 + t12 * s22
    b22 = t21 * s12 + t22 * s22

    # top eigenpair of [[b11,b12],[b12,b22]], symmetric 2x2
    lam = 0.5 * (b11+b22) + torch.sqrt((0.5 * (b11-b22)) ** 2 + b12 * b12)

    d1a, d2a = b12, lam - b11
    d1b, d2b = lam-b22, b12
    use_a = (d1a * d1a +d2a*d2a) >= (d1b*d1b + d2b*d2b)
    d1 = torch.where(use_a, d1a, d1b)
    d2 = torch.where(use_a, d2a, d2b)
    # both forms vanish only if B is (near-)isotropic: any direction is optimal
    dn = d1 * d1 + d2 * d2
    d1 = torch.where(dn < tiny, torch.where(b11 >= b22, torch.ones_like(d1), torch.zeros_like(d1)), d1)
    d2 = torch.where(dn < tiny, torch.where(b11 >= b22, torch.zeros_like(d2), torch.ones_like(d2)), d2)
 
    # undo the whitening: c = S^-1 d
    detS = (s11 * s22 - s12 * s12)
    # duplicates / colinear children make Gin singular. A is then rank-1 along w~_i, so u = w~_i/||w~_i||
    colinear = det < 1e-10 * (aii * ajj).clamp_min(tiny)
    detS = torch.where(colinear, torch.ones_like(detS), detS)
    c1 = (s22 * d1 - s12 * d2) / detS
    c2 = (-s12 * d1 + s11 * d2) / detS
    c1 = torch.where(colinear, 1.0 / torch.sqrt(aii.clamp_min(tiny)), c1)
    c2 = torch.where(colinear, torch.zeros_like(c2), c2)
 
    # renormalize so ||u||^2 = c^T Gin c = 1 exactly (Eq 14's constraint)
    q = (c1 * c1 * aii + 2.0 * c1 * c2 * aij + c2 * c2 * ajj).clamp_min(tiny)
    inv = 1.0 / torch.sqrt(q)
    return c1 * inv, c2 * inv


def _alignment(c1, c2 , eii, eij, ejj, oii, oij,ojj, gam_i, gam_j, beta_i, beta_j, rho_ij, ksi, ksj, sci, scj):
    """
    c1, c2: coefficients of u = c1wi + c2wj
    eii, eij, ejj : norm(w_i_eff), norm(w_i_eff, w_j_eff), norm(w_j_eff)
    oii, oij, ojj : norm(w_out) same as above

    evaluating Eq 10/13 objective

    _pair_direction solves the linearized problem (Eq. 14, the eigenvector u_hat
). _alignment is the exact, non-linearized objective
    """

    tiny = 1e-30

    uw_sq = (c1 * c1 * eii + 2.0 * c1 * c2 * eij + c2 * c2 * ejj).clamp_min(tiny)
    uw_norm = uw_sq.sqrt()

    #first moment, beta
    beta_u = c1 * beta_i + c2 * beta_j

    #second moment, gamma
    gam_u = torch.sqrt((c1*c1*gam_i*gam_i + c2*c2*gam_j*gam_j
                    + 2.0*c1*c2*gam_i.abs()*gam_j.abs()*rho_ij).clamp_min(0.0))

    
    ku = relu_self_kernel(gam_u, beta_u)
    sc_u = gam_u / uw_norm

    # goal: K(u,w_i) and K(u, w_j)
    dot_i = c1 * eii + c2 * eij                              # u_w . w_eff_i
    dot_j = c1 * eij + c2 * ejj
    cos_i = dot_i / (uw_norm * torch.sqrt(eii.clamp_min(tiny)))   # rho_eff(u, i)
    cos_j = dot_j / (uw_norm * torch.sqrt(ejj.clamp_min(tiny)))
    k_ui = bracket(warp(cos_i, sc_u * sci)) * torch.sqrt((ku * ksi).clamp_min(0.0))
    k_uj = bracket(warp(cos_j, sc_u * scj)) * torch.sqrt((ku * ksj).clamp_min(0.0))

    #the numerator
    num_sq = (k_ui*k_ui*oii + 2.0*k_ui*k_uj* oij + k_uj*k_uj*ojj).clamp_min(0.0)

    return torch.sqrt(num_sq) / torch.sqrt(ku.clamp_min(tiny)), k_ui, k_uj, ku


def _pair_core(aii, aij, ajj, oii, oij, ojj, eii, eij, ejj,
               gam_i, gam_j, bet_i, bet_j, ks_i, ks_j, sc_i, sc_j,
               extras: bool = False):
    """
     b and rho_hat for pairs, given their raw Gram/BN scalars -- the single
    implementation behind the cache build, the row refresh, and JIT execution.

    extras=False -> (b, rho_hat): the two scalars B.4 caches per pair.
    extras=True  -> (b, rho_hat, c1, c2, k_ui, k_uj, ku) with Eq 13's sign already folded into (c1, c2)
    
    """
    cos_ij = eij / (torch.sqrt(eii.clamp_min(1e-30)) * torch.sqrt(ejj.clamp_min(1e-30)))
    rho_hat = warp(cos_ij, sc_i * sc_j)                 # Eq 4

    c1, c2 = _pair_direction(aii, aij, ajj, oii, oij, ojj)   # Eq 14: u_hat

    args = (eii, eij, ejj, oii, oij, ojj,
            gam_i, gam_j, bet_i, bet_j, rho_hat, ks_i, ks_j, sc_i, sc_j)

    # Eq 11
    pos = _alignment(c1, c2, *args)
    neg = _alignment(-c1,-c2,*args)
    flip = neg[0] > pos[0]

    # b = <rho*, fi + fj>: the inner product between the optimal parent's unit direction and the sum of the two children
    b= torch.where(flip, neg[0], pos[0])
    if not extras:
        return b, rho_hat
    sgn = torch.where(flip, -torch.ones_like(c1), torch.ones_like(c1))
    return (b, rho_hat, sgn * c1, sgn * c2,
            torch.where(flip, neg[1], pos[1]),
            torch.where(flip, neg[2], pos[2]),
            torch.where(flip, neg[3], pos[3]))

            

def _layer_geometry(pack, rows: Optional[torch.Tensor] = None): # effeective space quantities computed directly
    """
    NOTE: our first version went through the augmented gram and substracted b_eff * b_eff which cancels
    whenever ||w_eff|| ^ 2 << b_eff^2 -- features.0 (d_in = 27, no upstream BN) is the specific case
    thus our new correction is below 


    """
    w_eff = pack.w_eff.to(_F64)
    b_eff = pack.b_eff.to(_F64)

    wo = pack.w_out.reshape(pack.n, -1).to(_F64)
    eff_d = (w_eff * w_eff).sum(dim=1) # >= 0 by construction, doesn't need to be clamped
    out_d = (wo * wo).sum(dim=1)
    aug_d = eff_d + b_eff*b_eff

    left_e = w_eff if rows is None else w_eff[rows]
    left_b = b_eff if rows is None else b_eff[rows]
    left_w = wo if rows is None else wo[rows]
    geff = left_e @ w_eff.T # direct, not by subtraction
    gaug=geff + left_b[:, None] * b_eff[None, :]

    gout = left_w@wo.T

    ks = relu_self_kernel(pack.gamma.to(_F64), pack.beta.to(_F64))    # K(k,k)
    sc = pack.gamma.to(_F64).abs() / torch.sqrt(eff_d.clamp_min(1e-30))
    return gaug, geff, gout, aug_d, eff_d, out_d, ks, sc


# ------------------ cache build
"""
def build_pair_cache(cs: CompressionState, li: int) -> PairCache: #InitializeCaches for one layer for every pair at once and keep only scalars b & rho_hat
    pack =cs.packs[li]
    gaug, geff,gout, aug_d, eff_d, out_d,ks, sc = _layer_geometry(pack)
    gam, bet = pack.gamma.to(_F64), pack.beta.to(_F64)
    n = pack.n
    dev = pack.gamma.device
    r=torch.arange(n, device=dev)[:, None]    # [N,1] x [1,N] broadcast = ever (i,j) at once
    c = torch.arange(n, device=dev)[None, :]
    b, rho_hat =_pair_core(
        aug_d[r], gaug, aug_d[c], out_d[r], gout, out_d[c],
        eff_d[r], geff, eff_d[c], gam[r], gam[c], bet[r], bet[c], ks[r], ks[c], sc[r], sc[c])
    pc = PairCache(b=b, rho_hat=rho_hat, dirty=False)
    cs.pairs[li] =pc
    return pc """ #this is our previous version that is inefficient

def build_pair_cache(cs: CompressionState, li: int, block: int = 128) -> PairCache: #InitializeCaches for one layer, block the rows so we never hold N^2 intermediates
    pack =cs.packs[li]
    gaug, geff,gout, aug_d, eff_d, out_d,ks, sc = _layer_geometry(pack)
    gam, bet = pack.gamma.to(_F64), pack.beta.to(_F64)
    n = pack.n
    dev = pack.gamma.device
    b = torch.empty(n, n, dtype=_F64, device=dev)
    rho_hat = torch.empty(n, n, dtype=_F64, device=dev)
    c = torch.arange(n, device=dev)[None, :]
    for lo in range(0, n, block):   # _pair_direction alone holds ~30 temporaries, so
        hi = min(lo + block, n)     # doing it [block,N] at a time instead of [N,N] is the whole win
        r = torch.arange(lo, hi, device=dev)[:, None]
        b[lo:hi], rho_hat[lo:hi] = _pair_core(
            aug_d[r], gaug[lo:hi], aug_d[c], out_d[r], gout[lo:hi], out_d[c],
            eff_d[r], geff[lo:hi], eff_d[c], gam[r], gam[c], bet[r], bet[c],
            ks[r], ks[c], sc[r], sc[c])
    pc = PairCache(b=b, rho_hat=rho_hat, dirty=False)
    cs.pairs[li] =pc
    return pc

def refresh_pair_row(cs: CompressionState, li: int, i: int) -> None: # after a merge deploys a parent into slot i, recompute the cached (b, rho-hat) of (i, k) for every live neighbor k
    pc=cs.pairs.get(li)
    if pc is None or pc.dirty:
        build_pair_cache(cs, li)
        return
    pack=cs.packs[li]
    alive = cs.states[li].kept()
    gaug,geff, gout, aug_d,eff_d, out_d, ks, sc = _layer_geometry(pack, rows=torch.as_tensor([i], device=pack.gamma.device))                     
    gam, bet = pack.gamma.to(_F64),pack.beta.to(_F64)
    b_row, r_row= _pair_core(                               
        aug_d[i], gaug[0, alive], aug_d[alive],
        out_d[i], gout[0, alive], out_d[alive],
        eff_d[i], geff[0, alive], eff_d[alive],
        gam[i], gam[alive], bet[i], bet[alive], ks[i], ks[alive], sc[i], sc[alive])
    pc.b[i, alive]= b_row    
    pc.b[alive, i] =b_row 
    pc.rho_hat[i, alive] = r_row
    pc.rho_hat[alive, i]= r_row


def _get_pair_cache(cs: CompressionState, li: int) -> PairCache:
    pc = cs.pairs.get(li)
    if pc is None or pc.dirty:
        pc = build_pair_cache(cs, li)
    return pc

# ------ generation

def best_merge(cs: CompressionState, li: int) -> Optional[Action]: #cheapest merge in layer li
    p, s = cs.packs[li], cs.states[li]
    if s.n_active < 2:
        return None
    pc = _get_pair_cache(cs, li)
    
    idx=s.kept()
    idx = idx[p.w_eff[idx].norm(dim=1) > EPS]
    m=idx.numel()
    if m < 2:
        return None


    """
    ii = idx[:, None].expand(m, m)
    jj= idx[None, :].expand(m, m)
    cap =s.cap.to(_F64)
    cap_i, cap_j = cap[ii], cap[jj]
""" # Our previous iteration is unnecessarily memory intensive as 
    # b.4 only ever wants i < j so we build that index set directly instead of the [m,m] mesh we mask half of away

    ti, tj = torch.triu_indices(m, m, offset = 1, device=idx.device)
    gi, gj = idx[ti], idx[tj]
    cap = s.cap.to(_F64)
    cap_i, cap_j = cap[gi], cap[gj]

    a = cap_i * cap_i + cap_j * cap_j # a = ||f_i|| ^ 2 + ||f_j||^2 
    b = pc.b[gi, gj]    # b = <psi*, f_i+f_j>
    e_rem = (s.e_rem-cap_i-cap_j).clamp_min(EPS)    # E_rem = max(E_a - ||f_i|| - ||f_j||, eps)
    s_star = (a + b * e_rem) / (2.0*e_rem + b).clamp_min(EPS)

    j_cost= s.n_active* torch.sqrt((2.0*s_star*s_star-2.0*b*s_star+a).clamp_min(0.0)) / (e_rem + s_star).clamp_min(EPS)
    # eq 6 w/ D^2 = ||f_i - f_p|||^ 2 + ||f_j - f_p|| ^2 ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    dp = p.dp[gj] #one neurons static footprint, *edit casted dp to float64 in state.py
    dr= j_cost / dp.clamp_min(EPS)     #DR = J / dP (Alg 1 l. 14)
     # _mask= torch.triu(torch.ones(m,m, dtype=torch.bool, device=dr.device), diagonal=1) & (b>0.0)
    dr=torch.where(b > 0.0, dr, torch.full_like(dr, float("inf"))) # undirected pairs, i < j 
    k = int(torch.argmin(dr))
    if not torch.isfinite(dr[k]):
        return None
    # r, c= divmod(k, m)

    return Action("merge", li, int(gi[k]), j=int(gj[k]), j_cost=float(j_cost[k]), dp=float(dp[k]))



def _physical_recovery(C1, C2, gam_i, gam_j, bet_i, bet_j, b_i, b_j, w_eff_i, w_eff_j, rho_ij, bn_eps):
    #phyiscal parameters for the parent w~*_in under anchoring by sigma^2_p 

    beta_p = C1 * bet_i + C2 * bet_j    # Eq 18
    b_p = C1 * b_i + C2 * b_j# Eq 16
    mu_p = beta_p - b_p     # Eq 17
    gamma_p = torch.sqrt((C1*C1*gam_i*gam_i+C2*C2*gam_j*gam_j+ 2.0*C1*C2*gam_i.abs()*gam_j.abs()*rho_ij).clamp_min(0.0))  # rho_hat from Eq 4
    var_p = torch.clamp(gamma_p * gamma_p - bn_eps, min=0.0)  
    w_raw_p = C1 * w_eff_i + C2 * w_eff_j   # eq 16 & 17 (w_raw = w_eff)
    if float(gamma_p * gamma_p) < bn_eps: # inactive regime B.5
        w_raw_p = torch.zeros_like(w_raw_p)
        mu_p = torch.zeros_like(mu_p)
        beta_p = b_p
    return w_raw_p, gamma_p, beta_p, mu_p, var_p


def execute_merge(cs: CompressionState, act: Action) -> None: #collapse pair (i, j) into the rank-1 parent deployed into slot i
    li, i, j = act.layer, act.i, act.j
    if j is None:
        raise ValueError("merge action carries no child j")
    p, s = cs.packs[li], cs.states[li]
    if not (bool(s.alive[i]) and bool(s.alive[j])):
        raise ValueError(f"{p.name}: merge targets ({i},{j}) not both alive")
    
    gaug, geff, gout, aug_d, eff_d, out_d, ks, sc = _layer_geometry(p, rows=torch.as_tensor([i], device=p.gamma.device))
    gam, bet = p.gamma.to(_F64), p.beta.to(_F64)
    b_val, rho_ij, c1, c2, k_ui, k_uj, ku = _pair_core( # eq 13 & 14
        aug_d[i], gaug[0, j], aug_d[j],
        out_d[i], gout[0, j], out_d[j],
        eff_d[i], geff[0, j], eff_d[j],
        gam[i], gam[j], bet[i], bet[j], ks[i], ks[j], sc[i], sc[j], extras=True)

    cap= s.cap.to(_F64)
    a=cap[i]**2 + cap[j]**2 # ||f_i|| ^ 2 + ||f_j||^2
    e_rem = torch.clamp(torch.as_tensor(s.e_rem, dtype=_F64) - cap[i] - cap[j], min=EPS)  # B.4
    s_star = (a + b_val * e_rem) / (2.0 * e_rem + b_val).clamp_min(EPS)  # Eq 40 (= Eq 12's s*)
    wo = p.w_out.reshape(p.n, -1).to(_F64)

    v_num = k_ui * wo[i] + k_uj * wo[j]     #eq 13 (v*= normalized sum_k K(u_c, w~_k) w_out_k)
    v_star = v_num / v_num.norm().clamp_min(1e-30)

    r_f = (torch.sqrt(aug_d[i] + aug_d[j])/( torch.sqrt((out_d[i] + out_d[j]).clamp_min(1e-30))).clamp_min(1e-30)) # ||W_in||_F/||W_out||_F
    k_self_qrt = ku.clamp_min(1e-30)** 0.25    # Eq 15: K_self^(1/4), K_self = K(u,u)
    amp_in=torch.sqrt(s_star * r_f) /k_self_qrt
    amp_out = torch.sqrt(s_star / r_f)/ k_self_qrt


    # our (c1, c2) are unit norm coefficients thus folding in eq 15s input amplitude yields exactly eq 16s proj coefficients

    C1, C2=amp_in *c1, amp_in*c2 
    w_raw_p, gamma_p, beta_p, mu_p, var_p = _physical_recovery(
        C1, C2, gam[i], gam[j], bet[i], bet[j],
        p.b_eff[i].to(_F64), p.b_eff[j].to(_F64),
        p.w_eff[i].to(_F64), p.w_eff[j].to(_F64), rho_ij, p.bn_eps)
    w_out_p = (amp_out * v_star).reshape(p.w_out.shape[1], p.k_out)

    #deploy into slot i
    dt = p.w_raw.dtype
    p.w_raw[i] = w_raw_p.to(dt)
    p.gamma[i] = gamma_p.to(dt)
    p.beta[i] = beta_p.to(dt)
    p.mu[i] = mu_p.to(dt)
    p.var[i] = var_p.to(dt)
    p.w_out[i] = w_out_p.to(dt)
    p.refold(torch.tensor([i], device=p.gamma.device))
    _sync_upstream(cs, li, i)   # parents raw weight is the upstream outgoing weight slice
    _sync_downstream(cs, li, i)  # parents outgoign weight is the downstream raw weight slice
    _kill(cs, li, j) # delete child j(zeroes, syncs, refreshes capacities)
    refresh_pair_row(cs, li, i) #repair parents cache row
    cs.history.append(act)




from .state import register_exec
register_exec("merge", execute_merge)

