import json
import os
from glob import glob

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as F


class CamVidDataset(Dataset):
    def __init__(self, root_dir, split, image_size, mask_size) -> None:
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.image_files = []
        self.mask_files = []
        self.class_dict = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x * 2 - 1  # Normalize images to [-1, 1]
        ])

        print('Loading class dictionary...')
        with open(os.path.join(root_dir, 'class_dict.json'), 'r') as f:
            class_dict = json.load(f)
        self.class_dict = {(item['r'], item['g'], item['b']): item['name'] for item in class_dict}
        self.class_name_to_id = {name: i for i, name in enumerate(self.class_dict.values())}

        print('Loading image and mask files...')
        for file in os.listdir(str(os.path.join(root_dir, split, 'images'))):
            if file.endswith('.png'):
                self.image_files.append(os.path.join(root_dir, split, 'images', file))
                mask_file = file.replace('.png', '_L.png')
                self.mask_files.append(os.path.join(root_dir, split, 'masks', mask_file))

    def get_class_mask(self, mask_file: str) -> np.ndarray:
        mask = datasets.folder.default_loader(mask_file)
        mask = F.resize(mask, self.mask_size)
        mask = np.array(mask)

        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)

        for rgb, class_name in self.class_dict.items():
            equality = np.equal(mask, np.array(rgb).reshape(1, 1, 3)).all(axis=2)
            class_mask[equality] = self.class_name_to_id[class_name]

        return class_mask

    def __getitem__(self, index: int) -> dict:
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]

        image = datasets.folder.default_loader(str(image_file))
        image = F.resize(image, self.image_size)

        class_mask = self.get_class_mask(str(mask_file))

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'mask': class_mask}

    def __len__(self) -> int:
        return len(self.image_files)


# RORD Dataset Folder Structure:
# --------------------------------
# The dataset is organized into a train folder, which contains three subfolders:
#   - gt (ground truth): This folder contains images with inpainted objects, stored in .jpg format.
#   - img (original images): This folder contains the original images, also stored in .jpg format.
#   - mask (masks for inpainted objects): This folder contains binary mask images, stored in .png format, where:
#      -> White pixels (255) represent the inpainted objects. Black pixels (0) represent the background.

class RORDDataset(Dataset):
    """
    RORD dataset for inpainting.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset, either 'train' or 'val'.
        image_size (int): Size of the image.
        transform (torchvision.transforms.Compose): Transforms to apply to the images.

    Returns:
        dict: Dictionary containing the original image, ground truth image, and mask.

    Usage:
        dataset = RORDDataset(root_dir='data/RORD', split='train', image_size=512)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch in dataloader:
            inputs, labels = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
    """

    def __init__(self, root_dir, split, image_size, transform=None) -> None:
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.image_files = []
        self.gt_files = []
        self.mask_files = []

        print('Loading image, ground truth, and mask files...')
        img_dir = os.path.join(root_dir, split, 'img')
        gt_dir = os.path.join(root_dir, split, 'gt')
        mask_dir = os.path.join(root_dir, split, 'mask')

        # check if the directories exist
        if not os.path.exists(img_dir) or not os.path.exists(gt_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError("One or more directories do not exist")

        img_files = glob(img_dir + '/**/*.jpg', recursive=True)
        gt_files = glob(gt_dir + '/**/*.jpg', recursive=True)
        mask_files = glob(mask_dir + '/**/*.png', recursive=True)

        print(f'Found {len(img_files)} image files')
        print(f'Found {len(gt_files)} ground truth files')
        print(f'Found {len(mask_files)} mask files')

        img_filenames = []
        for i, file in enumerate(img_files):

            img_filenames.append(os.path.basename(file))
            if i % 100 == 0:
                print(f'Loading image file {i + 1}/{len(img_files)}: {os.path.basename(file)}')

        gt_filenames = []
        for i, file in enumerate(gt_files):

            gt_filenames.append(os.path.basename(file))
            if i % 100 == 0:
                print(f'Loading ground truth file {i + 1}/{len(gt_files)}: {os.path.basename(file)}')

        mask_filenames = []
        for i, file in enumerate(mask_files):
            mask_filenames.append(os.path.basename(file))
            if i % 100 == 0:
                print(f'Loading mask file {i + 1}/{len(mask_files)}: {os.path.basename(file)}')

        if not img_filenames:
            print("Image directory is empty")
        elif not gt_filenames:
            print("Ground truth directory is empty")
        elif not mask_filenames:
            print("Mask directory is empty")
        else:
            print(
                f'Found {len(img_filenames)} image files, {len(gt_filenames)} ground truth files, and {len(mask_filenames)} mask files')

        common_filenames = set(os.path.splitext(os.path.basename(f))[0] for f in img_filenames) & set(
            os.path.splitext(os.path.basename(f))[0] for f in gt_filenames) & set(
            os.path.splitext(os.path.basename(f))[0] for f in mask_filenames)

        print(f'Found {len(common_filenames)} common files in the directories')
        for file in common_filenames:
            self.image_files.append(os.path.join(img_dir, file + '.jpg'))
            self.gt_files.append(os.path.join(gt_dir, file + '.jpg'))
            self.mask_files.append(os.path.join(mask_dir, file + '.png'))

        if not self.image_files or not self.gt_files or not self.mask_files:
            raise ValueError("One or more of the lists is empty")

    def __getitem__(self, index: int) -> dict:
        if index < len(self.image_files) and index < len(self.gt_files) and index < len(self.mask_files):
            image_file = self.image_files[index]
            gt_file = self.gt_files[index]
            mask_file = self.mask_files[index]

            image = datasets.folder.default_loader(str(image_file))
            gt = datasets.folder.default_loader(str(gt_file))
            mask = datasets.folder.default_loader(str(mask_file))

            image = F.resize(image, (self.image_size, self.image_size))
            gt = F.resize(gt, (self.image_size, self.image_size))
            mask = F.resize(mask, (self.image_size, self.image_size))

            if self.transform:
                image = self.transform(image)
                gt = self.transform(gt)

            mask = np.array(mask)  # Convert mask to a numpy array
            mask = (mask > 0).astype(np.float32) * 255

            return {'image': image, 'gt': gt, 'mask': mask}

        else:
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        return len(self.image_files)
