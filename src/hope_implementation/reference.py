"""
monte carlo reference implementations of relu kernels
"""

import math

import torch

DTYPE=torch.float64




def mc_self(gamma, beta, m=4_000_000, generator=None): #E[relu(y)^2] for y~ N(beta, gamma^2)
    z = torch.randn(m, dtype=DTYPE, generator=generator)
    y = beta + abs(gamma) * z
    s = torch.relu(y) ** 2

    return s.mean().item(), (s.std() / math.sqrt(m)).item()

def mc_cross(gamma_i, beta_i, gamma_j, beta_j, rho_hat, m=4_000_000, generator=None): #e[relu(y_i) relu(y_j)]
    z1 = torch.randn(m, dtype=DTYPE, generator=generator)
    z2 = torch.randn(m, dtype=DTYPE, generator=generator)

    y_i = beta_i + abs(gamma_i)*z1
    y_j = beta_j + abs(gamma_j) * (rho_hat * z1 + math.sqrt(1-rho_hat**2)*z2)

    s = torch.relu(y_i) * torch.relu(y_j)

    return s.mean().item(), (s.std() / math.sqrt(m)).item()


def z_score(estimate, reference, std_err):
    if std_err == 0.0:
        return None
    return abs(estimate-reference) / std_err