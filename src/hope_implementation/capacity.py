import math

import torch
import torch.nn as nn

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

#section 5


def relu_self_kernel(gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor: 
    #Clamping |gamma| keeps the gamma -> 0 limit finite
    
   
    eps = 1e-12
    scale = torch.abs(gamma).clamp_min(eps)
    ratio = beta / scale
    #pdf
    pdf = torch.exp(-0.5 * ratio * ratio) * _INV_SQRT_2PI
    #CDF 
    cdf = torch.special.ndtr(ratio)
    self_kernel = (torch.square(gamma) + torch.square(beta)) * cdf + beta * scale * pdf

    return self_kernel



def hope_capacity(w_out: torch.Tensor, gamma:torch.Tensor, beta:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    w_norm = w_out.norm(dim=1)
    capacity = w_norm * torch.sqrt(relu_self_kernel(gamma, beta))
    return capacity, capacity.sum()


