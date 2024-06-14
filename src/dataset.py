import json
import os
from glob import glob

import numpy as np
import torchvision.transforms as transforms
from loguru import logger
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
            lambda x: x * 2 - 1  # Normalize images to [-1, 1]
        ])

        logger.info('Loading class dictionary...')
        with open(os.path.join(root_dir, 'class_dict.json'), 'r') as f:
            class_dict = json.load(f)
        self.class_dict = {(item['r'], item['g'], item['b']): item['name'] for item in class_dict}
        self.class_name_to_id = {name: i for i, name in enumerate(self.class_dict.values())}

        logger.info('Loading image and mask files...')
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


class RORDDataset(Dataset):
    """
    RORD dataset for inpainting.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset, either 'train' or 'val'.
        image_size (tuple): Size of the images, assumed to be square.
        transform (torchvision.transforms.Compose): Transforms to apply to the images.

    Returns:
        dict: Dictionary containing the original image, ground truth image, and mask.
    """

    def __init__(self, root_dir: str, split: str, image_size: list[int], transform: transforms.Compose = None) -> None:
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.input_tranform = transform
        self.image_files = []
        self.gt_files = []
        self.mask_files = []

        logger.info('Loading image, ground truth, and mask files...')
        img_dir = os.path.join(root_dir, split, 'img')
        gt_dir = os.path.join(root_dir, split, 'gt')
        mask_dir = os.path.join(root_dir, split, 'mask')

        logger.info(
            f'Loading image files from: {img_dir}, ground truth files from: {gt_dir}, and mask files from: {mask_dir}')

        # check if the directories exist
        if not os.path.exists(img_dir) or not os.path.exists(gt_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError("One or more directories do not exist")

        img_files = glob(img_dir + '/**/*.jpg', recursive=True)
        gt_files = glob(gt_dir + '/**/*.jpg', recursive=True)
        mask_files = glob(mask_dir + '/**/*.png', recursive=True)

        logger.info(f'Found {len(img_files)} image files')
        logger.info(f'Found {len(gt_files)} ground truth files')
        logger.info(f'Found {len(mask_files)} mask files')

        img_filenames = []
        for i, file in enumerate(img_files):
            img_filenames.append(os.path.basename(file))
            if i % 100 == 0:
                logger.info(f'Loading image file {i + 1}/{len(img_files)}: {os.path.basename(file)}')

        gt_filenames = []
        for i, file in enumerate(gt_files):
            gt_filenames.append(os.path.basename(file))
            if i % 100 == 0:
                logger.info(f'Loading ground truth file {i + 1}/{len(gt_files)}: {os.path.basename(file)}')

        mask_filenames = []
        for i, file in enumerate(mask_files):
            mask_filenames.append(os.path.basename(file))
            if i % 100 == 0:
                logger.info(f'Loading mask file {i + 1}/{len(mask_files)}: {os.path.basename(file)}')

        if not img_filenames:
            logger.info("Image directory is empty")
        elif not gt_filenames:
            logger.info("Ground truth directory is empty")
        elif not mask_filenames:
            logger.info("Mask directory is empty")
        else:
            logger.info(
                f'Found {len(img_filenames)} image files, {len(gt_filenames)} ground truth files, and {len(mask_filenames)} mask files')

        common_filenames = set(os.path.splitext(os.path.basename(f))[0] for f in img_filenames) & set(
            os.path.splitext(os.path.basename(f))[0] for f in gt_filenames) & set(
            os.path.splitext(os.path.basename(f))[0] for f in mask_filenames)

        logger.info(f'Found {len(common_filenames)} common files in the directories')
        for file in common_filenames:
            self.image_files.append(os.path.join(img_dir, file + '.jpg'))
            self.gt_files.append(os.path.join(gt_dir, file + '.jpg'))
            self.mask_files.append(os.path.join(mask_dir, file + '.png'))

        if not self.image_files or not self.gt_files or not self.mask_files:
            raise ValueError("One or more of the lists is empty")

    def __getitem__(self, index: int) -> dict:
        image_file = self.image_files[index]
        gt_file = self.gt_files[index]
        mask_file = self.mask_files[index]

        image = datasets.folder.default_loader(str(image_file))
        gt = datasets.folder.default_loader(str(gt_file))
        mask = datasets.folder.default_loader(str(mask_file))

        # image = F.resize(image, self.image_size)
        # gt = F.resize(gt, self.image_size)
        # mask = F.resize(mask, self.image_size)
        #
        # image = ToTensor()(image)
        # gt = ToTensor()(gt)
        # mask = ToTensor()(mask)

        image, gt = [transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()])(img) for img in [image, gt]]
        mask = transforms.Compose([
            transforms.Resize([size // 2 for size in self.image_size]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])(mask)

        if self.input_tranform:
            image = self.input_tranform(image)
            gt = self.input_tranform(gt)

        return {'image': image, 'gt': gt, 'mask': mask}

    def __len__(self) -> int:
        return len(self.image_files)
