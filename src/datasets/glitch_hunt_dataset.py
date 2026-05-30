import os
import random
import PIL
import torch
from torchvision import transforms
from enum import Enum

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"

class GlitchHuntDataset(torch.utils.data.Dataset):
    """
    Custom Dataset handler for the Glitch Hunt category/good, category/bad folder structure.
    Expects source to be the category directory containing 'good' and 'bad' subdirectories.
    The train/test split is physical (separate directories), not computed at runtime.
    """
    def __init__(
        self,
        source,
        resize=256,
        imagesize=224,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        split=DatasetSplit.TRAIN,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.image_mean = image_mean
        self.image_std = image_std

        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ])

        self.imagesize = (3, imagesize, imagesize)
        self.data_to_iterate = self.get_image_data()

    def get_image_data(self):
        valid_exts = ('.jpg', '.jpeg', '.png')

        good_files = []
        bad_files = []

        for root, dirs, files in os.walk(self.source):
            folder_name = os.path.basename(root)

            if folder_name == "good":
                for f in files:
                    if f.lower().endswith(valid_exts):
                        good_files.append(os.path.join(root, f))
            elif folder_name == "bad":
                for f in files:
                    if f.lower().endswith(valid_exts):
                        bad_files.append(os.path.join(root, f))

        good_files = sorted(good_files)
        bad_files = sorted(bad_files)

        data = []
        for f in good_files:
            data.append(("glitch_hunt", "good", f))
        for f in bad_files:
            data.append(("glitch_hunt", "bad", f))

        return data

    def __len__(self):
        return len(self.data_to_iterate)

    # TODO it doesnt seems to make sense to have the segmentation mask in there.
    # TODO why is anomaly in there?
    def __getitem__(self, idx):
        classname, anomaly, image_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        
        # Apply standard Torchvision pipeline (Resize, CenterCrop, ToTensor, Normalize)
        image = self.transform_img(image)
        
        # We don't have segmentation masks, returning a placeholder mask
        mask = torch.zeros([1, *image.shape[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }

    def sample(self, n):
        indices = random.sample(range(len(self)), n)
        return torch.utils.data.Subset(self, indices)
