

import itertools

import torch

from hope_implementation.capacity import relu_self_kernel
from hope_implementation.kernel import cross_kernel_e, cross_kernel_m

torch.manual_seed(0)
N, n = 12, 64
DTYPE= torch.float64
TOL=1e-12

def make_layer(gamma_sign="mixed", beta_mode="mixed"):
    w_eff = torch.randn(N, n, dtype=DTYPE)

    gamma = torch.rand(N, dtype=DTYPE) * 2 + 0.1

    if gamma_sign=="mixed":
        gamma[::2] *= -1
    elif gamma_sign == "negative":
        gamma *= -1
    
    if beta_mode=="zero":
        beta=torch.zeros(N, dtype=DTYPE)
    elif beta_mode=="positive":
        beta = torch.rand(N, dtype=DTYPE)*2
    else:
        beta=torch.randn(N, dtype=DTYPE)* 1.5
    
    return w_eff, gamma, beta


def elementwise_matrix(w_eff, gamma, beta):

    K = torch.zeros(N, N, dtype=DTYPE)
    for i, j in itertools.product(range(N), repeat=2):
        K[i, j ] = cross_kernel_e (
            w_eff[i: i+1], w_eff[j: j+1],
            gamma[i: i+1], beta[i: i+1],
            gamma[j: j+1], beta[j: j+1]
        )
        
    return K


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"

    print(f" [{status}] {name}{(' ' + detail) if detail else ''}")
    return condition


def main():
    all_ok = True

    for gamma_sign in ("positive", "mixed"):
        for beta_mode in ("zero", "positive", "mixed"):
            print(f"\ngamma={gamma_sign}, beta={beta_mode}")
            w_eff, gamma, beta, = make_layer(gamma_sign, beta_mode)

            k_mat = cross_kernel_m(w_eff, gamma, beta)
            k_ref = elementwise_matrix(w_eff, gamma, beta)


            off = ~torch.eye(N, dtype=torch.bool)
            delta = (k_mat - k_ref)[off].abs().max().item()
            rel = delta / k_ref[off].abs().max().item()


            all_ok &= check("elementwise == matrix(off-diagonal)", rel<1e-10, f"max rel err {rel:.2e}")
            k_self = relu_self_kernel(gamma, beta)
            all_ok &= check("diagonal == K(i, i)", torch.allclose(k_mat.diagonal(), k_self, atol=TOL))

            all_ok &= check("symmetric", (k_mat - k_mat.T).abs().max().item() == 0.0)
            bound = torch.sqrt(k_self[:, None] * k_self[None, :])
            slack = (k_mat.abs() - bound).max().item()
            all_ok &= check("Cauchy-Schwarz |K(i,j)| <= sqrt(KiiKjj)", slack <= 1e-12, f"max slack {slack:.2e}")
 
            all_ok &= check("no NaN / Inf", torch.isfinite(k_mat).all().item())
    
    print("\nsign regression")
    w_eff, gamma, beta = make_layer("positive", "mixed")
    k_pos = cross_kernel_m(w_eff, gamma, beta)
    k_neg = cross_kernel_m(w_eff, -gamma, beta)
    k_flip = cross_kernel_m(w_eff, gamma * torch.tensor([1.0, -1.0]* (N//2), dtype=DTYPE), beta)

    all_ok &= check("K invariant to global gamma sign flip", torch.allclose(k_pos, k_neg, atol=TOL), f"max diff {(k_pos - k_neg).abs().max().item():.2e}")
    all_ok &= check("K invariant to per-neuron gamma sign flip", torch.allclose(k_pos, k_flip, atol=TOL), f"max diff {(k_pos - k_flip).abs().max().item():.2e}")
   
    print("\nanalytic anchors (beta = 0)")
    w = torch.zeros(2, n, dtype=DTYPE)
    w[0, 0] = 1.0
    g = torch.tensor([1.3, 0.7], dtype=DTYPE)
    b = torch.zeros(2, dtype=DTYPE)


    w[1] = w[0]  # identical -> rho = 1 -> Cauchy-Schwarz equality
    K = cross_kernel_m(w, g, b)
    expect = torch.sqrt(relu_self_kernel(g[0], b[0]) * relu_self_kernel(g[1], b[1]))
    rel = abs(K[0, 1].item() - expect.item()) / expect.item()
    all_ok &= check("rho=1 recovers sqrt(Kii Kjj)", rel < 1e-5, f"rel err {rel:.2e} (clamp floor ~1.1e-6)")
 
    w[1] = torch.zeros(n, dtype=DTYPE)
    w[1, 1] = 1.0  # orthogonal -> rho = 0 -> E[relu]E[relu]
    K = cross_kernel_m(w, g, b)
    expect = (g.abs()[0] / (2 * torch.pi) ** 0.5) * (g.abs()[1] / (2 * torch.pi) ** 0.5)
    all_ok &= check("rho=0 gives E[relu(yi)]E[relu(yj)]", abs(K[0, 1].item() - expect.item()) < 1e-9, f"got {K[0, 1].item():.6f} vs {expect.item():.6f}")
 
    w[1] = -w[0]  # anti-aligned -> rho = -1 -> never co-fire -> 0
    K = cross_kernel_m(w, g, b)
    all_ok &= check("rho=-1 gives 0", abs(K[0, 1].item()) < 1e-6, f"got {K[0, 1].item():.2e}")
 
    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())