"""
Dataset Classes for RORD Inpainting and COCO Segmentation

This file contains two dataset classes, RORDInpaintingDataset and COCOSegmentationDataset,
which are used for training inpainting generators and segmentation models, respectively.

Classes:
    - RORDInpaintingDataset: Handles the RORD dataset for inpainting tasks.
    - COCOSegmentationDataset: Handles the COCO dataset for segmentation tasks.
        (the used dataset was COCO 2017 version with 91 classes: see details at https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)

Usage:
    To use these dataset classes, follow the steps below:

    1. Ensure the datasets are organized as described in the 'Data Organization' section.
    2. Instantiate the dataset objects with appropriate parameters.
    3. Use the dataset objects with PyTorch's DataLoader for training models.

Example:

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # Instantiate RORD Inpainting Dataset
    rord_dataset = RORDInpaintingDataset(
        root_dir='/path/to/rord_dataset',
        split='train',
        image_size=(256, 256),
        transform=transform,
        normalize=True
    )

    # Create DataLoader for RORD dataset
    rord_loader = DataLoader(rord_dataset, batch_size=32, shuffle=True)

    # Iterate through RORD DataLoader
    for batch in rord_loader:
        images, gts = batch['image'], batch['gt']
        # Training code here...

    # Instantiate COCO Segmentation Dataset
    coco_dataset = COCOSegmentationDataset(
        root_dir='/path/to/coco_dataset',
        split='train',
        year='2017',
        image_size=(256, 256),
        mask_size=(256, 256),
        transform=transform,
        selected_class_ids=[1, 2, 3],  # Example class IDs
        normalize=True
    )

    # Create DataLoader for COCO dataset
    coco_loader = DataLoader(coco_dataset, batch_size=32, shuffle=True)

    # Iterate through COCO DataLoader
    for batch in coco_loader:
        images, masks = batch['image'], batch['gt']
        # Training code here...

Functions:
    - normalize_image(image): Normalizes a tensor image to be in the range [0, 1].

Exceptions:
    - FileNotFoundError: Raised when specified directories or files do not exist.
    - ValueError: Raised when there are mismatches or issues with the dataset files.

RORDInpaintingDataset expects the following directory structure:
    root_dir/
        ├── train/
        │   ├── img/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── gt/
        │       ├── image1.jpg
        │       ├── image2.jpg
        │       └── ...
        └── val/
            ├── img/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── gt/
                ├── image1.jpg
                ├── image2.jpg
                └── ...

COCOSegmentationDataset expects the following directory structure:
    root_dir/
        ├── train/
        │   ├── img/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── gt/
        │       ├── image1.jpg
        │       ├── image2.jpg
        │       └── ...
        └── val/
            ├── img/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── gt/
                ├── image1.jpg
                ├── image2.jpg
                └── ...

"""

import logging
import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class RORDInpaintingDataset(Dataset):
    """
    RORD dataset adapted for inpainting generator training.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset, either 'train' or 'val'.
        image_size (tuple): Size of the images, assumed to be square.
        transform (torchvision.transforms.Compose, optional): Transforms to apply to the images.
        normalize (bool, optional): Whether to normalize the images. Default is False.

    Returns:
        dict: Dictionary containing the original image and ground truth image.
    """

    def __init__(self, root_dir: str, split: str, image_size: tuple, transform: transforms.Compose = None,
                 normalize=False) -> None:
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.input_transform = transform
        self.normalize = normalize
        self.image_files = []
        self.gt_files = []

        logger_local = logging.getLogger(__name__)
        logger_local.info('Loading image and ground truth files...')
        img_dir = os.path.join(root_dir, split, 'img')
        gt_dir = os.path.join(root_dir, split, 'gt')

        logger_local.info(f'Loading image files from: {img_dir}, ground truth files from: {gt_dir}')

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            raise FileNotFoundError("One or more directories do not exist")

        img_files = glob(img_dir + '/**/*.jpg', recursive=True)
        gt_files = glob(gt_dir + '/**/*.jpg', recursive=True)

        logger_local.info(f'Found {len(img_files)} image files and {len(gt_files)} ground truth files')

        common_filenames = set(os.path.splitext(os.path.basename(f))[0] for f in img_files) & \
                           set(os.path.splitext(os.path.basename(f))[0] for f in gt_files)

        logger_local.info(f'Found {len(common_filenames)} common files in the directories')
        for file in common_filenames:
            self.image_files.append(os.path.join(img_dir, file + '.jpg'))
            self.gt_files.append(os.path.join(gt_dir, file + '.jpg'))

        self.image_files.sort()
        self.gt_files.sort()

        if not self.image_files or not self.gt_files:
            raise ValueError("One or more of the lists is empty")

        self.valid_indices = [i for i in range(len(self.image_files))
                              if os.path.exists(self.image_files[i]) and
                              os.path.exists(self.gt_files[i])]

        logger_local.info(f"Found {len(self.valid_indices)} valid samples out of {len(self.image_files)} total samples")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict:
        try:
            valid_index = self.valid_indices[index]
            image_file = self.image_files[valid_index]
            gt_file = self.gt_files[valid_index]

            image_name = os.path.splitext(os.path.basename(image_file))[0]
            gt_name = os.path.splitext(os.path.basename(gt_file))[0]

            if image_name != gt_name:
                raise ValueError("File names do not match")

            image = Image.open(str(image_file)).convert('RGB')
            # print(
            #     f"RORD Image after Image.open max: {image.getextrema()}, min: {image.getextrema()}, shape: {image.size}")
            gt = Image.open(str(gt_file)).convert('RGB')

            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()])(image)
            # print(f"Image after transforms max: {image.max()}, min: {image.min()}, shape: {image.size}")
            gt = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()])(gt)

            # Normalize images if required
            if self.normalize:
                image = normalize_image(image)
                gt = normalize_image(gt)
                # print(f"Image after normalization max: {image.max()}, min: {image.min()}, shape: {image.size}")

            if self.input_transform:
                image = self.input_transform(image)
                gt = self.input_transform(gt)

            return {'image': image, 'gt': gt}
        except Exception as e:
            logging.error(f"Error loading data at index {index}: {str(e)}")
            return {'image': torch.zeros(3, *self.image_size),
                    'gt': torch.zeros(3, *self.image_size)}


def normalize_image(image):
    """
    Normalize a tensor image to be in the range [0, 1].

    Args:
        image (torch.Tensor): The input image tensor with values in the range [0, 255].

    Returns:
        torch.Tensor: Normalized image tensor in the range [0, 1].
    """
    # Ensure the image is in the range [0, 1]
    return image / 255.0


class COCOSegmentationDataset(Dataset):
    """
    COCO segmentation dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset, either 'train' or 'val'.
        year (str): Year of the dataset.
        image_size (tuple): Size of the images, assumed to be square.
        mask_size (tuple): Size of the masks, assumed to be square.
        transform (torchvision.transforms.Compose, optional): Transforms to apply to the images.
        selected_class_ids (list, optional): List of class IDs to include in the dataset.
        normalize (bool, optional): Whether to normalize the images. Default is False.
        custom_mean (list, optional): Custom mean values for normalization.
        custom_std (list, optional): Custom standard deviation values for normalization.
    """

    def __init__(self, root_dir: str, split: str, year: str, image_size: tuple, mask_size: tuple, transform=None,
                 selected_class_ids=None, normalize=False, custom_mean=None, custom_std=None):
        self.root_dir = root_dir
        self.split = split
        self.year = year
        self.image_size = image_size
        self.mask_size = mask_size
        self.transform = transform
        self.normalize = normalize
        self.custom_mean = custom_mean
        self.custom_std = custom_std

        ann_file = os.path.join(root_dir, f'annotations/instances_{split}{year}.json')
        assert os.path.exists(ann_file), f'Annotation file not found at {ann_file}'
        self.coco = COCO(ann_file)

        self.cat_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(self.cat_ids)
        self.categories.sort(key=lambda x: x['id'])

        self.selected_class_ids = selected_class_ids if selected_class_ids else self.cat_ids
        self.num_classes = len(self.selected_class_ids)

        self.cat_id_to_class_idx = {cat_id: idx for idx, cat_id in enumerate(self.selected_class_ids)}

        self.image_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.selected_class_ids, iscrowd=False)
            if len(ann_ids) > 0:
                self.image_ids.append(img_id)

        # Set up logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict:
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        # Construct the image path based on split and year
        img_path = os.path.join(self.root_dir, f'{self.split}{self.year}', img_info['file_name'])

        # Check if image file exists
        if not os.path.exists(img_path):
            self.logger.warning(f"Image file not found: {img_path}")
            return None

        try:
            image = Image.open(img_path).convert('RGB')  # Dimension: (C=3, H, W)
        except IOError:
            self.logger.warning(f"Unable to open image file: {img_path}")
            return None

        image = transforms.Resize(self.image_size)(image)
        image = transforms.ToTensor()(image)

        if self.normalize:
            image = self.normalize_image(image)

        if self.transform:
            image = self.transform(image)

        # Initialize multichannel mask
        mask = np.zeros((self.num_classes, self.mask_size[0], self.mask_size[1]),
                        dtype=np.float32)  # Dimension: (num_classes, H, W)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.selected_class_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        if not anns:
            self.logger.warning(f"No annotations found for image: {img_path}")
            return None

        for ann in anns:
            class_idx = self.cat_id_to_class_idx[ann['category_id']]
            pixel_mask = self.coco.annToMask(ann)
            resized_mask = np.asarray(Image.fromarray(pixel_mask).resize(self.mask_size, Image.NEAREST))
            mask[class_idx] = np.maximum(mask[class_idx], resized_mask)  # Ensure all instances are included

        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return {'image': image, 'gt': mask_tensor}

    # @staticmethod
    # def normalize_image(image):
    #     return image / 255.0

# class CamVidDataset(Dataset):
#     def __init__(self, root_dir, split, image_size, mask_size) -> None:
#         self.root_dir = root_dir
#         self.split = split
#         self.image_size = image_size
#         self.mask_size = mask_size
#         self.image_files = []
#         self.mask_files = []
#         self.class_dict = {}
#         self.transform = transforms.Compose([
#             lambda x: x * 2 - 1  # Normalize images to [-1, 1]
#         ])
#
#         logger.info('Loading class dictionary...')
#         with open(os.path.join(root_dir, 'class_dict.json'), 'r') as f:
#             class_dict = json.load(f)
#         self.class_dict = {(item['r'], item['g'], item['b']): item['name'] for item in class_dict}
#         self.class_name_to_id = {name: i for i, name in enumerate(self.class_dict.values())}
#
#         logger.info('Loading image and mask files...')
#         for file in os.listdir(str(os.path.join(root_dir, split, 'images'))):
#             if file.endswith('.png'):
#                 self.image_files.append(os.path.join(root_dir, split, 'images', file))
#                 mask_file = file.replace('.png', '_L.png')
#                 self.mask_files.append(os.path.join(root_dir, split, 'masks', mask_file))
#
#     def get_class_mask(self, mask_file: str) -> np.ndarray:
#         mask = datasets.folder.default_loader(mask_file)
#         mask = F.resize(mask, self.mask_size)
#         mask = np.array(mask)
#
#         class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
#
#         for rgb, class_name in self.class_dict.items():
#             equality = np.equal(mask, np.array(rgb).reshape(1, 1, 3)).all(axis=2)
#             class_mask[equality] = self.class_name_to_id[class_name]
#
#         return class_mask
#
#     def __getitem__(self, index: int) -> dict:
#         image_file = self.image_files[index]
#         mask_file = self.mask_files[index]
#
#         image = datasets.folder.default_loader(str(image_file))
#         image = F.resize(image, self.image_size)
#
#         class_mask = self.get_class_mask(str(mask_file))
#
#         if self.transform:
#             image = self.transform(image)
#
#         return {'image': image, 'mask': class_mask}
#
#     def __len__(self) -> int:
#         return len(self.image_files)
#
#
# def normalize_image(image):
#     # Ensure the image is in the range [0, 1]
#     return image / 255.0
