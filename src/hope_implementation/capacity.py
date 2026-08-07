import torch
import math

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

def relu_self_kernel(gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    #Clamping |gamma| keeps the gamma -> 0 limit finite

    scale = torch.abs(gamma).clamp_min(eps)
    ratio = beta / scale

    #pdf
    pdf = torch.exp(-0.5 * ratio * ratio) * _INV_SQRT_2PI

    #CDF 
    cdf = torch.special.ndtr(ratio)

    self_kernel = (torch.square(gamma) + torch.square(beta)) * cdf + beta * scale * pdf

    return self_kernel



def hope_capacity(model) -> dict[str, torch.Tensor]:
    pass 