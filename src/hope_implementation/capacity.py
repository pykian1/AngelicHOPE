import math

import torch
import torch.nn as nn
from .folding import LayerPack

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
DTYPE = torch.float64

#section 5


def relu_self_kernel(gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor: 
    #Clamping |gamma| keeps the gamma -> 0 limit finite
    gamma = gamma.to(DTYPE)
    beta = beta.to(DTYPE)
    eps = 1e-12
    scale = torch.abs(gamma).clamp_min(eps)
    ratio = beta / scale
    #pdf
    pdf = torch.exp(-0.5 * ratio * ratio) * _INV_SQRT_2PI
    #CDF 
    cdf = torch.special.ndtr(ratio)
    self_kernel = (torch.square(gamma) + torch.square(beta)) * cdf + beta * scale * pdf

    return self_kernel.clamp_min(0.0)



def hope_capacity(w_out: torch.Tensor, gamma:torch.Tensor, beta:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    
    assert w_out.shape[0] == gamma.shape[0] == beta.shape[0], (
        f"neuron count mismatch: w_out {tuple(w_out.shape)}, "
        f"gamma {tuple(gamma.shape)}, beta {tuple(beta.shape)}"
    )
    w_norm = w_out.to(DTYPE).norm(dim=1)
    capacity = w_norm * torch.sqrt(relu_self_kernel(gamma, beta))
    return capacity, capacity.sum()


