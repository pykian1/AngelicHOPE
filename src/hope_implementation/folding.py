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

    
    view_shape = [conv.shape[0]] + [1]*(conv_w.dim() - 1 )

    scale_reshape = _scale.view(*view_shape)



    w_fold = scale_reshape * conv_w
    b_fold = beta - _scale*mean

    return w_fold, b_fold