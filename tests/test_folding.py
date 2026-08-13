"""BN folding (eq 1) against torch's reference fusion, on the real checkpoint."""
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_weights

from hope_implementation.folding import fold_bn
from hope_implementation.models import load_frozen
from hope_implementation.state import build_state

CKPT = Path(__file__).parents[1] / "checkpoints" / "vgg8_baseline.pt"
LIM = 1e-4
BN = nn.modules.batchnorm._BatchNorm

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="baseline checkpoint not present")


@pytest.fixture(scope="module")
def model():
    return load_frozen(str(CKPT), device=torch.device("cpu"))


def find_pairs(model):
    """Every (conv2d|linear) immediately followed by a BN."""
    pairs = []
    for parent_name, block in model.named_modules():
        if not isinstance(block, nn.Sequential):
            continue
        kids = list(block.named_children())
        for (n1, m1), (_, m2) in zip(kids, kids[1:]):
            if isinstance(m1, (nn.Conv2d, nn.Linear)) and isinstance(m2, BN):
                pairs.append((f"{parent_name}.{n1}", m1, m2))
    return pairs


def test_folded_forward_matches_module(model):
    torch.manual_seed(0)
    pairs = find_pairs(model)
    assert pairs, "no conv/BN pairs found"

    for name, lin, bn in pairs:
        if isinstance(lin, nn.Conv2d):
            x = torch.randn(4, lin.in_channels, 8, 8)
            ref = bn(lin(x))
            w, b = fold_bn(lin, bn)
            got = F.conv2d(x, w, b, lin.stride, lin.padding, lin.dilation, lin.groups)
        else:
            x = torch.randn(4, lin.in_features)
            ref = bn(lin(x))
            w, b = fold_bn(lin, bn)
            got = F.linear(x, w, b)
        delta = (ref - got).abs().max().item()
        assert delta < LIM, f"{name}: max|delta| {delta:.2e}"


def test_matches_torch_fusion(model):
    name, conv, bn = next((n, l, b) for n, l, b in find_pairs(model) if isinstance(l, nn.Conv2d))
    our_w, our_b = fold_bn(conv, bn)
    ref_w, ref_b = fuse_conv_bn_weights(
        conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )
    assert (our_w - ref_w).abs().max() < 1e-6, name
    assert (our_b - ref_b).abs().max() < 1e-6, name


def test_train_mode_guard(model):
    _, conv, bn = next((n, l, b) for n, l, b in find_pairs(model) if isinstance(l, nn.Conv2d))
    bn.train()
    try:
        with pytest.raises(AssertionError):
            fold_bn(conv, bn)
    finally:
        bn.eval()

def test_refold_matches_fold_bn(model):
    """LayerPack.refold (eq 1, used by the pipeline) must agree with fold_bn
    (eq 1, validated against torch fusion above)."""
    cs = build_state(model)
    for p in cs.packs:
        w_ref, b_ref = fold_bn(p.layer, p.bn)
        w_ref = w_ref.detach().reshape(p.n, -1).double()
        b_ref = b_ref.detach().double()
        assert torch.allclose(p.w_eff, w_ref, atol=1e-10), f"{p.name}: w_eff"
        assert torch.allclose(p.b_eff, b_ref, atol=1e-10), f"{p.name}: b_eff"