import torch.nn as nn
import torchvision


class LogisticRegression(nn.Module):
    """Standard logistic regression model."""
    def __init__(self, input_size=600, *_args, **_kwargs):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        # Flatten images (batch_size, channels, h, w) into (batch_size, 1) to have a score per image
        self.linear = torch.nn.Linear(3 * self.input_size * self.input_size, 1)
        #normalization to get 0-1 scores
        self.final_sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = x.view(-1, 3 * self.input_size * self.input_size)
        x = self.linear(x)
        # Uncomment if using BCELoss instead of BCEWithLogitsLoss
        # x = self.final_sigmoid(x)
        return x
