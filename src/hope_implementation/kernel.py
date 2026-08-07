import math

import torch
import torch.nn as nn
from .capacity import relu_self_kernel

def cross_kernel(w_eff_i: torch.Tensor, w_eff_j: torch.Tensor, gamma_i: torch.Tensor, beta_i:torch.Tensor, gamma_j: torch.Tensor, beta_j:torch.Tensor) -> torch.Tensor:
    rho_eff = torch.cosine_similarity(w_eff_i,w_eff_j,dim=1,eps=1e-12)
    rho_eff = torch.clamp(rho_eff, -1 + 1e-6, 1 - 1e-6)
    scale_i = torch.abs(gamma_i) / torch.clamp(w_eff_i.norm(dim=1), min=1e-12)
    scale_j = torch.abs(gamma_j) / torch.clamp(w_eff_j.norm(dim=1), min=1e-12)

    kappa = (rho_eff/(1-torch.square(rho_eff)))*scale_i*scale_j

    corr = (2*kappa) / (1 + torch.sqrt(1+4*torch.square(kappa)))
    corr = torch.clamp(corr, -1.0, 1.0)

    bracket = (1/math.pi) * (torch.sqrt(1-torch.square(corr)) + ((math.pi - torch.arccos(corr))* corr))
    cross_kernel = bracket * torch.sqrt(relu_self_kernel(gamma_i,beta_i) * relu_self_kernel(gamma_j,beta_j))

    return cross_kernel