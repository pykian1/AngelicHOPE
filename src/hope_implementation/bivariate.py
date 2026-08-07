"""
given scale of VGG8, 512x512 layer takes few seconds in float64 on cpu
we can use plackett's identity for an exact form as validation reference
as the papaer approximates cross kernel ( i,j) with zero bias arccos
to avoid a bivariate normal cdf per pair


thus, phi_2 is evaluated by plackett's identity to reduce to a 1-d integral

phi_2(h,k; rho) = phi(h)*phi(k) + integral from 0 to rho of (phi_2(h,k;r))


"""


import math

import numpy as np
import torch

DTYPE = torch.float64
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
 
def _phi(z):
    return torch.exp(-0.5 * z * z) * _INV_SQRT_2PI
 
 
def _Phi(z):
    return torch.special.ndtr(z)
 
 
def bivariate_phi(h, k, r):
    """Standard bivariate normal PDF at (h, k) with correlation r."""
    one_minus = 1.0 - r * r
    quad = (h * h - 2.0 * r * h * k + k * k) / (2.0 * one_minus)
    return torch.exp(-quad) / (2.0 * math.pi * torch.sqrt(one_minus))
 
 
def bivariate_Phi(h, k, rho, n_nodes: int = 48):
    """Standard bivariate normal CDF via Plackett's identity.
 
    h, k and rho broadcast together. 48 nodes is well past convergence in
    float64; 24 is fine if memory is tight (cost scales as size * n_nodes).
    """
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    x = torch.tensor(x, dtype=DTYPE)
    w = torch.tensor(w, dtype=DTYPE)
 
    h, k, rho = torch.broadcast_tensors(
        torch.as_tensor(h, dtype=DTYPE),
        torch.as_tensor(k, dtype=DTYPE),
        torch.as_tensor(rho, dtype=DTYPE),
    )
 
    # map [-1, 1] -> [0, rho]; trailing axis holds the quadrature nodes
    half = rho.unsqueeze(-1) / 2.0
    r = half * (x + 1.0)
    integrand = bivariate_phi(h.unsqueeze(-1), k.unsqueeze(-1), r)
    integral = (integrand * w).sum(-1) * half.squeeze(-1)
 
    return _Phi(h) * _Phi(k) + integral
 
 
def c_k_exact(gamma_i, beta_i, gamma_j, beta_j, rho_hat, n_nodes: int = 48):

    gi = torch.as_tensor(gamma_i, dtype=DTYPE).abs()
    gj = torch.as_tensor(gamma_j, dtype=DTYPE).abs()
    bi = torch.as_tensor(beta_i, dtype=DTYPE)
    bj = torch.as_tensor(beta_j, dtype=DTYPE)
    rho = torch.as_tensor(rho_hat, dtype=DTYPE).clamp(-1 + 1e-12, 1 - 1e-12)
 
    ci = bi / gi.clamp_min(1e-12)
    cj = bj / gj.clamp_min(1e-12)
 
    root = torch.sqrt(1.0 - rho * rho)
    ci_given_j = (ci - rho * cj) / root
    cj_given_i = (cj - rho * ci) / root
 
    term1 = (ci * cj + rho) * bivariate_Phi(ci, cj, rho, n_nodes)
    term2 = ci * _phi(cj) * _Phi(ci_given_j)
    term3 = cj * _phi(ci) * _Phi(cj_given_i)
    term4 = (1.0 - rho * rho) * bivariate_phi(ci, cj, rho)
 
    return gi * gj * (term1 + term2 + term3 + term4)
 