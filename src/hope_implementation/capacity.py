import math

import torch
import torch.nn as nn


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
DTYPE = torch.float64
_PACK_DEV = torch.device("cpu")   # float64 is unsupported on MPS

#section 5


def relu_self_kernel(gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor: 
    #Clamping |gamma| keeps the gamma -> 0 limit finite
    device = gamma.device
    gamma = gamma.detach().to(_PACK_DEV).to(DTYPE)
    beta = beta.detach().to(_PACK_DEV).to(DTYPE)
    eps = 1e-12
    scale = torch.abs(gamma).clamp_min(eps)
    ratio = beta / scale
    #pdf
    pdf = torch.exp(-0.5 * ratio * ratio) * _INV_SQRT_2PI
    #CDF 
    cdf = torch.special.ndtr(ratio)
    self_kernel = (torch.square(gamma) + torch.square(beta)) * cdf + beta * scale * pdf
    out = self_kernel.clamp_min(0.0)
    out_dtype = torch.float32 if device.type == "mps" else DTYPE
    return out.to(device=device, dtype=out_dtype)



def hope_capacity(w_out: torch.Tensor, gamma:torch.Tensor, beta:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    
    assert w_out.shape[0] == gamma.shape[0] == beta.shape[0], (
        f"neuron count mismatch: w_out {tuple(w_out.shape)}, "
        f"gamma {tuple(gamma.shape)}, beta {tuple(beta.shape)}"
    )
    device = w_out.device
    w_norm = w_out.detach().reshape(w_out.shape[0], -1).to(_PACK_DEV).to(DTYPE).norm(dim=1)
    capacity = w_norm * torch.sqrt(relu_self_kernel(gamma, beta)).to(_PACK_DEV).to(DTYPE)
    out_dtype = torch.float32 if device.type == "mps" else DTYPE
    capacity = capacity.to(device=device, dtype=out_dtype)
    return capacity, capacity.sum()


