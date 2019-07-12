import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

        # create new model by removing the last layer
        self.model = torch.nn.Sequential(*(list(model.children())[:-1][:-2]))

    def forward(self, x):
        return self.model(x)
