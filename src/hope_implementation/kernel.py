import math

import torch
import torch.nn as nn
from .capacity import relu_self_kernel

#section 5

def cross_kernel_e(w_eff_i: torch.Tensor, w_eff_j: torch.Tensor, gamma_i: torch.Tensor, beta_i:torch.Tensor, gamma_j: torch.Tensor, beta_j:torch.Tensor) -> torch.Tensor:
    #elementwise cross_kernel

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



def cross_kernel_m(w_eff: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor: 

    #nxn matrix crosskernel
    
    #normalize and compute cosine similarity
    norms = w_eff.norm(dim=1).clamp_min(1e-12) #[N]
    w_hat = w_eff / norms[:, None] # [N, n]

    rho_eff = w_hat @ w_hat.T # [N,N] 

    rho_eff = rho_eff.clamp(-1 + 1e-6, 1 - 1e-6)

    scale = gamma.abs() / norms #[N]
    scale_outer = scale[:, None] * scale[None, :] # [N,N]

    kappa = (rho_eff / ((1.0-rho_eff)*(1.0+rho_eff))) * scale_outer

    corr = (2*kappa) / (1+ torch.sqrt(1+4 * (torch.square(kappa))))

    corr = corr.clamp(-1.0, 1.0)

    brack = (1/math.pi) * (torch.sqrt(1 - torch.square(corr)) + ((math.pi - torch.arccos(corr))*corr))
    self_kernel = relu_self_kernel(gamma, beta)#  [N]
    sqrt_self = torch.sqrt(self_kernel)

    cross_kernel = brack * (sqrt_self[:, None] * sqrt_self[None, :]) # [N, N]
    
    cross_kernel.diagonal().copy_(self_kernel)

    return .5 * (cross_kernel + cross_kernel.T)





def warp(rho_eff: torch.Tensor, scale_prod: torch.Tensor) -> torch.Tensor:
    rho = rho_eff.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    kappa = (rho / ((1.0-rho)*(1.0+rho))) * scale_prod
    return (2.0*kappa / (1.0 + torch.sqrt(1.0+4.0*kappa*kappa))).clamp(-1.0, 1.0)

def bracket(rho_hat: torch.Tensor) -> torch.Tensor:
    r = rho_hat.clamp(-1.0, 1.0)
    return (1.0 / math.pi) * (torch.sqrt((1.0-r*r).clamp_min(0.0))+ (math.pi - torch.arccos(r))* r)