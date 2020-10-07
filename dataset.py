import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class TomatoesDataset(Dataset):
    """Dataset for our tomato images and annotations."""

    def __init__(self, split, seed=30318, path="dataset", img_folder="assignment_imgs", transform=None):
        """Constructor for a TomatoDataset

        The default expected structure is the following:
        
            home-assignment
            │   ...
            │
            ├───dataset
            │   │   img_annotations.json
            │   │   label_mapping.csv
            │   │
            │   └───assignment_imgs
            │           ...

        Args:
            split (string): Whether to make this a training or testing dataset
            seed (int, optional): Random seed for train/test splitting, which must be shared between the two TomatoDataset objects. Defaults to 30318.
            path (str, optional): Relative path to the folder containing annotations. Defaults to "dataset".
            img_folder (str, optional): Relative path to the folder containing annotations, within [path] folder. Defaults to "assignment_imgs".
            transform ([type], optional): torchvision.transforms object which may be applied to the images. Defaults to None.
        """
        # Only two possible splits. Extend this for a dev/validation set.
        assert split in ["train", "test"]

        # ID to class mapping for ingredients
        mapping = pd.read_csv(os.path.join(path, "label_mapping.csv"))
        # Image annotation file
        with open(os.path.join(path, "img_annotations.json")) as f:
            annot = json.load(f)

        # Restrict to tomato-related ingredients, as described in ../notebooks/EDA.ipynb
        tomatoes = mapping[['tomato' in item.lower() for item in mapping.labelling_name_en]]['labelling_id'].to_list()
        imgs_with_tomatoes = [img for img, info in annot.items() for item in info if item['id'] in tomatoes]

        """
        Identify the relavant classes and split training and test sets in a stratified way to account for class imbalance.
        random_state=seed gurantees that there will be no overlap with the two TomatoDataset objects.
        """
        y = [1 if img in imgs_with_tomatoes else 0 for img in annot.keys()]
        X_train, X_test, y_train, y_test = train_test_split(list(annot.keys()), y, test_size=0.2, random_state=seed, stratify=y)

        if split == "train":
            self.image_paths = [os.path.join(path, img_folder, x) for x in X_train]
            self.labels = np.array(y_train)
        else:
            self.image_paths = [os.path.join(path, img_folder, x) for x in X_test]
            self.labels = np.array(y_test)

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = torch.from_numpy(np.array(self.labels[index]).astype('float32')).float()

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)
