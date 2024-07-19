import json
import logging
import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from loguru import logger
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import datasets


# with open('../environment.yml', 'r') as f:
#     config = yaml.safe_load(f)
#
# coco_img_mean = config['vars']['COCO_IMG_MEAN']
# coco_img_std = config['vars']['COCO_IMG_STD']


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


class RORDInpaintingDataset(Dataset):
    """
    RORD dataset adapted for inpainting generator training.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset, either 'train' or 'val'.
        image_size (tuple): Size of the images, assumed to be square.
        transform (torchvision.transforms.Compose): Transforms to apply to the images.

    Returns:
        dict: Dictionary containing the original image and ground truth image.
    """

    def __init__(self, root_dir: str, split: str, image_size: tuple, transform: transforms.Compose = None) -> None:
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.input_transform = transform
        self.image_files = []
        self.gt_files = []

        logger.info('Loading image and ground truth files...')
        img_dir = os.path.join(root_dir, split, 'img')
        gt_dir = os.path.join(root_dir, split, 'gt')

        logger.info(f'Loading image files from: {img_dir}, ground truth files from: {gt_dir}')

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            raise FileNotFoundError("One or more directories do not exist")

        img_files = glob(img_dir + '/**/*.jpg', recursive=True)
        gt_files = glob(gt_dir + '/**/*.jpg', recursive=True)

        logger.info(f'Found {len(img_files)} image files and {len(gt_files)} ground truth files')

        common_filenames = set(os.path.splitext(os.path.basename(f))[0] for f in img_files) & \
                           set(os.path.splitext(os.path.basename(f))[0] for f in gt_files)

        logger.info(f'Found {len(common_filenames)} common files in the directories')
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

        logger.info(f"Found {len(self.valid_indices)} valid samples out of {len(self.image_files)} total samples")

    def __getitem__(self, index: int) -> dict:
        try:
            valid_index = self.valid_indices[index]
            image_file = self.image_files[valid_index]
            gt_file = self.gt_files[valid_index]

            image_name = os.path.splitext(os.path.basename(image_file))[0]
            gt_name = os.path.splitext(os.path.basename(gt_file))[0]

            if image_name != gt_name:
                raise ValueError("File names do not match")

            image = datasets.folder.default_loader(str(image_file))
            gt = datasets.folder.default_loader(str(gt_file))

            image, gt = [transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()])(img) for img in [image, gt]]

            if self.input_transform:
                image = self.input_transform(image)
                gt = self.input_transform(gt)

            return {'image': image, 'gt': gt}
        except Exception as e:
            logger.error(f"Error loading data at index {index}: {str(e)}")
            return {'image': torch.zeros(3, *self.image_size),
                    'gt': torch.zeros(3, *self.image_size)}

    def __len__(self) -> int:
        return len(self.valid_indices)


class COCOSegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading COCO segmentation data.

    This dataset loads images and their corresponding segmentation masks from the COCO dataset.
    It supports selecting specific classes and creates a multi-channel mask where each channel
    corresponds to a selected class.

    Attributes:
        root_dir (str): Root directory of the COCO dataset.
        split (str): Data split ('train', 'val', etc.).
        year (str): COCO dataset year.
        image_size (tuple): Target size for the input images.
        mask_size (tuple): Target size for the segmentation masks.
        transform (callable, optional): Optional transform to be applied on the image.
        selected_class_ids (list, optional): List of class IDs to include. If None, all classes are used.

    The dataset returns a dictionary containing:
        - 'image': A normalized RGB image tensor of shape (3, H', W').
        - 'mask': A multi-channel binary mask tensor of shape (num_classes, H'', W''),
                  where each channel corresponds to a selected class.
    """

    def __init__(self, root_dir: str, split: str, year: str, image_size: tuple, mask_size: tuple, transform=None,
                 selected_class_ids=None):
        self.root_dir = root_dir
        self.split = split
        self.year = year
        self.image_size = image_size
        self.mask_size = mask_size
        self.transform = transform

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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root_dir, f'{self.split}{self.year}', img_info['file_name'])

        # Check if image file exists
        if not os.path.exists(img_path):
            self.logger.warning(f"\nImage file not found: {img_path}")
            return None

        try:
            image = Image.open(img_path).convert('RGB')  # Dimension: (C=3, H, W)
        except IOError:
            self.logger.warning(f"\nUnable to open image file: {img_path}")
            return None

        # Initialize multichannel mask
        mask = np.zeros((self.num_classes, img_info['height'], img_info['width']),
                        dtype=np.float32)  # Dimension: (num_classes, H, W)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.selected_class_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        if not anns:
            self.logger.warning(f"\nNo annotations found for image: {img_path}")
            return None

        for ann in anns:
            class_idx = self.cat_id_to_class_idx[ann['category_id']]
            pixel_mask = self.coco.annToMask(ann)
            mask[class_idx, pixel_mask == 1] = 1.0  # Update mask with class IDs

        image, mask = self.transform_data(image, mask)  # Resize and transform to tensor

        return {'image': image, 'gt': mask}

    def transform_data(self, image, mask):
        # Resize image
        image = F.resize(image, self.image_size, interpolation=F.InterpolationMode.BILINEAR)  # Dimension: (C=3, H', W')

        # Resize mask for each class
        mask = torch.as_tensor(mask, dtype=torch.float32)  # Convert to float tensor
        resized_mask = torch.zeros((self.num_classes, *self.mask_size), dtype=torch.float32)

        for i in range(self.num_classes):
            mask_channel = Image.fromarray(mask[i].numpy())
            resized_mask[i] = F.to_tensor(
                F.resize(mask_channel, self.mask_size, interpolation=F.InterpolationMode.NEAREST))

        # Convert image to tensor
        image = F.to_tensor(image)  # Dimension: (C=3, H', W')

        # Normalize image
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Dimension: (C=3, H', W')

        # Apply additional transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, resized_mask

    def get_class_names(self):
        return [self.coco.loadCats([cat_id])[0]['name'] for cat_id in self.selected_class_ids]
