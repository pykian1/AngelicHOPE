import torch
import torch.nn as nn



#quality of life
def _qol(cin, cout, pool=True):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return layers

#basic cnn 
class angel(nn.Module):

    def __init__(self, num_classes=10, width=1):

        super().__init__()
        c1, c2, c3 = 32*width, 64*width, 128*width


        self.features = nn.Sequential(
            *_qol(3, c1, pool=False), *_qol(c1, c1),
            *_qol(c1, c2, pool=False), *_qol(c2, c2),
            *_qol(c2, c3, pool=False), *_qol(c3, c3)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(self.flatten(), nn.Linear(c3, num_classes))

        
    def forward(self, x):

        return self.classifier(self.pool(self.features(x)))


#paper reference architecture; refer to appendix g.2 of hope paper:




class VGG8(nn.Module):


    def __init__(self, num_classes=10, w=(128, 256, 512, 512)):
        super().__init__()

        c1, c2, c3, c4 = w

        self.features = nn.Sequential(
            *_qol(3, c1), *_qol(c1, c2),
            *_qol(c2, c3, pool=False), *_qol(c3, c3),
            *_qol(c3, c4, pool=False), *_qol(c4, c4)
        )

        self.pool = nn.AdapativeAvgPool2d(1)
        self.classifier = nn.Sequential(self.flatten(), nn.Linear(c4, num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

        

        