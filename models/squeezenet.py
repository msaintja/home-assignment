import torch.nn as nn
import torchvision


class SqueezeNet(nn.Module):
    """SqueezeNet model, may use pretrained ImageNet weights.
    A good, lightweight model for CAM/Grad-CAM.
    """
    def __init__(self, pretrained=False, *_args, **_kwargs):
        super(SqueezeNet, self).__init__()
        self.squeezenet = torchvision.models.squeezenet1_0(pretrained=pretrained)
        
        if pretrained:
            for param in self.squeezenet.parameters():
                param.requires_grad = False

        squeezenet_features = self.squeezenet.classifier[1].in_channels
        # re-implement final layer as output_size is different (1 score per image)
        self.squeezenet.classifier[1] = nn.Conv2d(squeezenet_features, 1, kernel_size=(1,1), stride=(1,1))
        #normalization to get 0-1 scores
        self.final_sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.squeezenet(x)
        # Uncomment if using BCELoss instead of BCEWithLogitsLoss
        # x = self.final_sigmoid(x)
        return x
