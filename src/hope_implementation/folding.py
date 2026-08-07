import torch
import torch.nn as nn
from typing import Tuple




#section 3 of the paper

def fold_bn(conv: nn.modules.conv._ConvNd, bn: nn.modules.batchnorm._BatchNorm) -> Tuple[torch.Tensor, torch.Tensor]:

    assert not bn.training, ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

    conv_w = conv.weight.detach() #don't need to worry about bias since bias is None

    #extract bn stats: 

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps



    std = torch.sqrt(var + eps)

    _scale = gamma/std

    
    view_shape = [conv_w.shape[0]] + [1]*(conv_w.dim() - 1 )

    scale_reshape = _scale.view(*view_shape)



    w_fold = scale_reshape * conv_w
    b_fold = beta - _scale*mean

    return w_fold, b_fold


# section b1 of paper

from dataclasses import dataclass

@dataclass
class LayerPack:
    layer: nn.Module
    bn:nn.Module
    next_layer: nn.Module
    w_eff: torch.Tensor
    b_eff:torch.Tensor
    w_out:torch.Tensor
    gamma:torch.Tensor
    beta:torch.Tensor
    dp: torch.Tensor

def build_layer_packs(model: nn.Module) -> list[LayerPack]:
    seq = [m for m in model.modules() if isinstance(m,(nn.Conv2d, nn.BatchNorm1d, nn.Linear,nn.BatchNorm2d ))]
    packs = []

    for i in range(len(seq)-2):
        a, bn = seq[i], seq[i+1]
        if not isinstance(a,(nn.Linear,nn.Conv2d)):
            continue
        if not isinstance(bn,(nn.BatchNorm2d, nn.BatchNorm1d)):
            continue
        
        nxt = seq[i+2]
        w_fold, b_fold = fold_bn(a,bn)
        N = w_fold.shape[0] #number of neurons which = number of output units of a
        if isinstance(nxt, nn.Conv2d):
            w_out = nxt.weight.detach().transpose(0,1).reshape(N,-1)
        else:
            w_out = nxt.weight.detach().t() #(N, out_features)

        w_eff = w_fold.reshape(N,-1)
        d_in = w_eff.shape[1]
        d_out = w_out.shape[1]
        dp = torch.full((N,),float(d_in+d_out+4))

        packs.append(LayerPack(
            layer=a,bn=bn,
            next_layer=nxt, 
            w_eff=w_fold.reshape(N,-1), #[N,d_in]
            b_eff=b_fold,
            w_out=w_out,
            gamma= bn.weight.detach(),
            beta=bn.bias.detach(),
            dp=dp
        ))
    return packs
