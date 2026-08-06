import torch
import torch.nn as nn




def _block(cin, cout, pool=True):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), 
        nn.BatchNorm2d(cout), 
        nn.ReLU(inplace=True)
    ]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return layers
 
class Angel(nn.Module):

    def __init__(self, num_classes=10, width=1):

        super().__init__()
        c1, c2, c3 = 32*width, 64*width, 128*width


        self.features = nn.Sequential(
            *_block(3, c1, pool=False), *_block(c1, c1),
            *_block(c1, c2, pool=False), *_block(c2, c2),
            *_block(c2, c3, pool=False), *_block(c3, c3)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c3, num_classes))

        
    def forward(self, x):

        return self.classifier(self.pool(self.features(x)))


#paper reference architecture; refer to appendix g.2 of hope paper:




class VGG8(nn.Module):


    def __init__(self, num_classes=10, w=(128, 256, 512, 512)):
        super().__init__()

        c1, c2, c3, c4 = w

        self.features = nn.Sequential(
            *_block(3, c1), *_block(c1, c2),
            *_block(c2, c3, pool=False), *_block(c3, c3),
            *_block(c3, c4, pool=False), *_block(c4, c4)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.classifier(self.bottleneck(self.pool(self.features(x))))

        


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_frozen(path, model=None, device=None, **model_kwargs):
    device = device or get_device()
    if model is None:
        model = VGG8(**model_kwargs)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return freeze(model.to(device))