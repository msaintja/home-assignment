import argparse
import copy
import os
import numpy as np
from PIL import Image
from sklearn.metrics import recall_score, f1_score
import torch
from torchvision import transforms
from dataset import TomatoesDataset
from utils.hyperparameters import Hyperparameters

def train(hps):
    """Train a model according to user specification (hps)

    Args:
        hps (Hyperparameters): object describing hyperparameters (model type, optimizer, etc.)
    """

    # Transforms - one with data augmentation for training
    transform = transforms.Compose([
        transforms.Resize((hps.image_size, hps.image_size)),
        transforms.ToTensor(),
    ])
    transform_train = transforms.Compose([
        transforms.Resize((hps.image_size, hps.image_size)),
        # Dishes should be flip-agnostic
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Model should be robust to contrast/lighting changes (smartphone photos)
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        ]),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = TomatoesDataset(split='train', seed=hps.random_seed, transform=transform_train)
    test_dataset = TomatoesDataset(split='test', seed=hps.random_seed, transform=transform)

    # DataLoaders
    kwargs = {'num_workers': hps.num_workers, 'pin_memory': hps.pin_memory} if hps.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=hps.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                    batch_size=hps.batch_size, shuffle=True, **kwargs)

    dataloaders = {"train": train_loader, "test": test_loader}

    # Main training/validation loop
    best_weights = None
    best_accuracy = 0.0

    for epoch in range(hps.epochs):
        print("----------------------------------------------------------------------")
        print(f"Epoch {epoch+1}/{hps.epochs}")
        print("----------------------------------------------------------------------")

        # Factoring forward pass code (some bits are unique to training)
        for phase in ["train", "test"]:
            if phase == "train":
                hps.model.train()
            else:
                hps.model.eval()

            batch_losses = []
            batch_outputs = []
            batch_labels = []
            
            # Using relevant dataloader to retrieve data
            for inputs, labels in dataloaders[phase]:
                # moving X,y data to the appropriate device (if GPU)
                inputs = inputs.to(f"cuda:{hps.cuda_device}" if hps.use_cuda else "cpu")
                labels = labels.to(f"cuda:{hps.cuda_device}" if hps.use_cuda else "cpu")
                # to make labels of size (batch, 1), like the outputs
                labels = torch.unsqueeze(labels, 1)

                # zero the parameter gradients
                hps.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = hps.model(inputs)
                    loss = hps.criterion(outputs, labels)

                    # Update the optimizer with a step if we're training
                    if phase == "train":
                        loss.backward()
                        hps.optimizer.step()

                # Bookeeping for further statistics
                batch_losses.append(loss.item())
                batch_outputs.append(torch.squeeze(outputs).detach().cpu().numpy())
                batch_labels.append(torch.squeeze(labels).detach().cpu().numpy())
                

            epoch_loss = np.mean(np.array(batch_losses))
            outputs = np.concatenate(batch_outputs)

            #make a 0 or 1 decision
            # Uncomment for BCEWithLogitsLoss
            sigmoid = lambda x: 1/(1 + np.exp(-x))
            # Uncomment for BCELoss (sigmoid layer added in model file)
            # sigmoid = lambda x: x
            
            outputs = np.around(sigmoid(outputs))
            labels = np.concatenate(batch_labels)
            epoch_accuracy = np.mean(outputs == labels)

            recall = recall_score(labels, outputs)
            f1 = f1_score(labels, outputs)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
            # Bookeeping for tensorboard
            hps.writer.add_scalar(f"{phase.capitalize()}/Loss", epoch_loss, epoch)
            hps.writer.add_scalar(f"{phase.capitalize()}/Accuracy", epoch_accuracy, epoch)
            hps.writer.add_scalar(f"{phase.capitalize()}/Recall", recall, epoch)
            hps.writer.add_scalar(f"{phase.capitalize()}/f1-score", f1, epoch)

            # Keep track of the best test acc and weights
            if phase == 'test' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_weights = copy.deepcopy(hps.model.state_dict())

    # Check that we didn't land here without any training
    assert best_weights is not None, "No training was performed."
    # Save the hyperparameters used + metric for tensorboard Hparam module
    hps.writer.add_hparams(hps.get_full_hps_dict(), {"hparam/best_accuracy": best_accuracy})
    # Save the weights for has_tomatoes()
    torch.save(best_weights, hps.weights_save_path)

def has_tomatoes(hps):
    """Given a trained model, indicates whether or not the image is/contains a tomato-related ingredient.

    Args:
        hps (Hyperparameters): object describing hyperparameters (model/image location, GPU devices, etc.)

    Returns:
        [bool]: True if hps.test image is/contains a tomato-related ingredient.
    """

    assert hps.test and os.path.isfile(hps.test), f"Could not load candidate image {hps.test}"
    assert hps.weights_save_path and os.path.isfile(hps.weights_save_path), f"Could not load weights {hps.weights_save_path}"
    # Loading weights
    hps.model.load_state_dict(torch.load(hps.weights_save_path))
    # Test-only mode
    hps.model.eval()

    # Load the image as usual
    image = Image.open(hps.test).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((hps.image_size, hps.image_size)),
        transforms.ToTensor(),
    ])
    # Batch of one image, unsqueeze as models expect a batch_size dim
    image = transform(image).unsqueeze(0)
    image = image.to(f"cuda:{hps.cuda_device}" if hps.use_cuda else "cpu")

    with torch.set_grad_enabled(False):
        output = hps.model(image).detach().cpu().numpy()

        # Uncomment for BCEWithLogitsLoss
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        # Uncomment for BCELoss (sigmoid layer added in model file)
        # sigmoid = lambda x: x
        
        smooth_score = np.squeeze(sigmoid(output))
        # Make a 0/1 decision
        prediction = np.around(smooth_score)
        print("Predicted " + ("" if prediction == 1 else "not a ") + f"tomato with score {smooth_score:.4f}")
        # Save it for tensorboard - nice viz
        hps.writer.add_image(f"{hps.test}/Score:{smooth_score:.4f}", torch.squeeze(image))
        
        # Return a boolean as per function specification
        return prediction == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tomato allergies trainer")
    parser.add_argument("-c", "--use-cuda", choices=["yes", "no", "default"], default="default", help="Use cuda for pytorch models")
    parser.add_argument("-i", "--cuda-device", type=int, help="If cuda-enabled, ID of GPU to use")
    parser.add_argument("-m", "--model-name", choices=["logistic", "squeezenet", "resnet", "transformer"], help="Model class name")
    parser.add_argument("-p", "--pretrained", action="store_true", help="Whether to use a pretrained model on ImageNet")
    parser.add_argument("-o", "--optimizer-name", choices=["sgd", "adam"], default="sgd", help="Optimizer criterion name")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs for train mode")
    parser.add_argument("-r", "--lr", type=float, help="Learning rate for train mode")
    parser.add_argument("-d", "--weight-decay", type=float, help="Weight decay (L2 penalty-based regularization)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Image batch size")
    parser.add_argument("-s", "--image-size", type=int, default=224, help="Resizing images to this size (e.g. 227 for SqueezeNet)")
    parser.add_argument("-w", "--weights-save-path", type=str, default="tomato-predictor.pt", help="Weights save path for training (w) and predicting (r)")
    parser.add_argument("-t", "--test", type=str, default="", help="Image to test a trained model on - outputs True if contains a tomato")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="Log directory for Tensorboard")
    parser.add_argument("-g", "--random-seed", type=int, default=30318, help="Random seed for train/test splitting")
    args = parser.parse_args()

    hps = Hyperparameters()
    hps.load_from_args(args.__dict__)
    print("Hyperparameters:")
    print("----------------------------------------------------------------------")
    print(hps)
    print("----------------------------------------------------------------------")

    if hps.test:
        print(has_tomatoes(hps))
    else:
        train(hps)

    # Close the Tensorboard writer
    hps.writer.close()
