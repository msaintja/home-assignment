import torch.nn as nn
import torchvision


class ResNet152(nn.Module):
    """ResNet152 model, may use pretrained ImageNet weights.
    A good, lightweight model for CAM/Grad-CAM.
    """
    def __init__(self, pretrained=False, *_args, **_kwargs):
        super(ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=pretrained)
        
        if pretrained:
            for param in self.resnet152.parameters():
                param.requires_grad = False

        resnet_features = self.resnet152.fc.in_features
        # re-implement final layer as output_size is different (1 score per image)
        self.resnet152.fc = nn.Linear(resnet_features, 1)
        #normalization to get 0-1 scores
        self.final_sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.resnet152(x)
        # Uncomment if using BCELoss instead of BCEWithLogitsLoss
        # x = self.final_sigmoid(x)
        return x

