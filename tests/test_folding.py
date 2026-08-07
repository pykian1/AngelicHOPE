import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_weights

from hope_implementation.models import load_frozen
from hope_implementation.folding import fold_bn

CKPT = "checkpoints/vgg8_baseline.pt"
LIM = 1e-4
BN = nn.modules.batchnorm._BatchNorm

def find_paris(model): # every (conv2d|linear) immediately followed by bn
    pairs = []
    for parentName, block, in model.named_modules():
        if not isinstance(block, nn.Sequential):
            continue
        kids = list(block.named_children())
        for (n1, m1), (_, m2) in zip(kids, kids[1:]):
            if isinstance(m1, (nn.Conv2d, nn.Linear)) and isinstance(m2, BN):
                pairs.append((f"{parentName}.{n1}", m1, m2))

    return pairs

def main():
    torch.manual_seed(0)
    model = load_frozen(CKPT, device=torch.device("cpu"))

    pairs = find_paris(model)
    print(f"found {len(pairs)} conv/BN pairs\n")

    worst = 0.0

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

        worst = max(worst,delta)

        gamma = bn.weight.detach()
        dead = (gamma.abs() < 1e-2).float().mean().item()

        status = "ok " if delta < LIM else "fail"

        print(
            f"{status} {name:<18} max|delta| {delta:.2e}  "
            f"dead gamma {100*dead:4.1f}%")

    print(f"\nworst caross all pairs: {worst:.2e}")

    name, conv, bn = next((n, l, b) for n, l, b in pairs if isinstance(l, nn.Conv2d))

    our_w, our_b = fold_bn(conv, bn)

    ref_w, ref_b = fuse_conv_bn_weights(
        conv.weight, conv.bias, bn.running_mean, bn.running_var, 
        bn.eps, bn.weight, bn.bias
    )

    print(
        f"vs torch fusion({name}): "
        f"w {(our_w - ref_w).abs().max():.2e} "
        f"b {(our_b - ref_b).abs().max():.2e} ")

    bn.train()
    try:
        fold_bn(conv, bn)
        print("train mode guard did not fire :()")
    except AssertionError:
        print("train mode guard fires :) ")
    bn.eval()


if __name__ == "__main__":
    main()




