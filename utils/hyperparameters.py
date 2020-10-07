import os
import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.baseline import LogisticRegression
from models.squeezenet import SqueezeNet
from models.resnet import ResNet152
# from models.transformer_encoder import TransformerEncoder

class Hyperparameters:
    """Hyperparameters configuration class"""
    def __init__(self):
        self.use_cuda = False
        self.cuda_device = 0
        self.epochs = 35
        self.batch_size = 6324
        self.image_size = 224
        self.random_seed = 30318

        # Default model
        self.model_name = "resnet"
        self.pretrained = True
        self.weights_save_path = "tomato-predictor.pt"

        # Default optimizer
        self.optimizer_name = "sgd"
        self.weight_decay = 0.0001
        self.lr = 0.0005

        # Default logdir for Tensorboard
        self.log_dir = "logs"
        self.writer = None

        # Placeholders for further loading
        self.model = None
        self.criterion = None
        self.num_workers = None
        self.pin_memory = None
        self.test = None


    def load_from_args(self, args):
        """Load hyperparameters from the arguments passed to the main.py file

        Args:
            args (dict): Dictionary of attributes/flags which should be set to specific values
        """

        # Any key from flags
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()
                setattr(self, key, val)        

        # Handle use_cuda flag
        if self.use_cuda == "default":
            self.use_cuda = torch.cuda.is_available()
        elif self.use_cuda == "yes":
            self.use_cuda = True
        else:
            self.use_cuda = False

        # Tensorboard
        log_dir = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{self.model_name}"
        log_path = os.path.join(self.log_dir, log_dir)
        self.writer = SummaryWriter(log_path)

        # Instantiate a model from a selection
        self.model = {
            "logistic": LogisticRegression,
            "squeezenet": SqueezeNet,
            "resnet": ResNet152,
        }.get(self.model_name, None)(input_size=self.image_size, pretrained=self.pretrained)

        # Criterion - initially BCELoss, but this helps with balancing classes
        # To balance: add pos_weight=torch.Tensor([6]) [only with BCEWithLogitsLoss]
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([6]))

        if self.use_cuda:
            torch.cuda.set_device(self.cuda_device)
            self.criterion = self.criterion.cuda()
            self.model.cuda()
            self.num_workers = 4
            self.pin_memory = True

        # Choose the optimizer and relevant hyperparameters
        self.optimizer = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam
            # add more here if needed
        }.get(self.optimizer_name, None)(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)


    def get_full_hps_dict(self):
        """Returns the list of hyperparameters as a flat dict for tensorboard"""
        parameters = ["weight_decay", "lr", "epochs", "batch_size", "optimizer_name"]

        hps = {}
        for param in parameters:
            value = getattr(self, param)
            hps[param] = value

        return hps

    def __str__(self):
        """Nicely lists hyperparameters when object is printed"""
        parameters = ["use_cuda", "cuda_device", "weight_decay",
        "lr", "epochs", "batch_size", "model_name", "pretrained",
        "optimizer_name", "image_size",
        "weights_save_path", "logdir", "random_seed"]
        info_str = ""
        for i, param in enumerate(parameters):
            value = getattr(self, param)
            info_str += f"[{str(i)}] {param}: {str(value)}\n"

        return info_str
