import math
 
import pytest
import torch
 
from hope_implementation.capacity import relu_self_kernel
from hope_implementation.kernel import cross_kernel_m
from hope_implementation.reference import mc_cross, mc_self, z_score
 
DTYPE = torch.float64
M = 500_000          # samples per config; raise for a tighter check
MAX_SIGMA = 4.0      # ~1 in 16k false-failure rate per config
SEED = 0
 
 
def _generator():
    g = torch.Generator()
    g.manual_seed(SEED)
    return g
 
 
@pytest.mark.parametrize("gamma", [-2.0, -0.3, 0.05, 0.8, 3.0])
@pytest.mark.parametrize("beta", [-2.0, 0.0, 0.7, 2.5])
def test_self_kernel_matches_mc(gamma, beta):
    ref = relu_self_kernel(
        torch.tensor(gamma, dtype=DTYPE), torch.tensor(beta, dtype=DTYPE)
    ).item()
    est, se = mc_self(gamma, beta, m=M, generator=_generator())
    z = z_score(est, ref, se)
    if z is None:
        # Kernel vanishes numerically: beta is many sigma below zero, so no
        # sample survives the ReLU and the sample variance is exactly 0. The
        # z-score is undefined; assert the closed form is negligible instead.
        assert est == 0.0 and ref < 1e-10, f"MC gave 0 but closed form is {ref:.3e}"
        return
    assert z < MAX_SIGMA, f"closed form {ref:.6f} vs MC {est:.6f} ({z:.1f} sigma)"
 
 
@pytest.mark.parametrize("rho", [-0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999])
def test_cross_kernel_matches_mc_zero_bias(rho):
    gi, gj = 1.3, 0.7
 
    # drive the real code path: build a 2-filter layer whose cosine similarity
    # is rho, rather than calling an internal helper with rho directly
    w = torch.zeros(2, 2, dtype=DTYPE)
    w[0, 0] = 1.0
    w[1, 0] = rho
    w[1, 1] = math.sqrt(max(0.0, 1.0 - rho * rho))
    gamma = torch.tensor([gi, gj], dtype=DTYPE)
    beta = torch.zeros(2, dtype=DTYPE)
 
    K = cross_kernel_m(w, gamma, beta)
    # cross_kernel_m warps rho_eff through kappa , so recover the
    # rho_hat it actually used by inverting the bracket, and sample at that.
    ref = K[0, 1].item()
    rho_hat = _invert_bracket(ref / math.sqrt(
        relu_self_kernel(gamma, beta)[0].item()
        * relu_self_kernel(gamma, beta)[1].item()))
 
    est, se = mc_cross(gi, 0.0, gj, 0.0, rho_hat, m=M, generator=_generator())
    z = z_score(est, ref, se)
    assert z is not None and z < MAX_SIGMA, (
        f"rho={rho}: closed form {ref:.6f} vs MC {est:.6f} ({z:.1f} sigma)")
 
 
def _invert_bracket(value, lo=-1.0, hi=1.0, iters=80):

    def bracket(r):
        return (1 / math.pi) * (
            math.sqrt(max(0.0, 1 - r * r)) + (math.pi - math.acos(max(-1.0, min(1.0, r)))) * r
        )
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bracket(mid) < value:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
