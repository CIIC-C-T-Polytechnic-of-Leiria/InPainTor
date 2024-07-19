"""
    Performs inference on the test set using the trained InPainTor.

    Usage:
        python src/inference.py --model_path "path/to/model.pth" --data_dir "path/to/data" --image_size 512 --mask_size 256 --batch_size 1 --output_dir "path/to/outputs"
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Update and reload modules to reflect recent changes
import dataset
import model

importlib.reload(dataset)
importlib.reload(model)

from dataset import RORDInpaintingDataset  # Adjust according to the updated dataset
from model import InpainTor  # Adjust according to the updated model


def inference(model_path: str, data_dir: str, image_size: tuple, mask_size: tuple, batch_size: int,
              class_dict: dict, output_dir: str) -> None:
    """
    Performs inference on the test set and saves the output masks.

    Parameters:
        model_path (str): Path to the trained InPainTor.
        data_dir (str): Path to the data directory.
        image_size (tuple): Size of the input images.
        mask_size (tuple): Size of the masks.
        batch_size (int): Batch size for inference.
        class_dict (dict): Dictionary mapping class IDs to RGB values.
        output_dir (str): Directory where the output masks will be saved.
    """

    # Create test DataLoader

    test_dataset = RORDInpaintingDataset(root_dir=data_dir, split='test', image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the InPainTor
    InPainTor = InpainTor(num_classes=len(class_dict))  # Adjust the number of classes based on the dictionary
    InPainTor.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    InPainTor.eval()
    InPainTor.to('cuda')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Inference
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Inference')):
            inputs = batch['image'].to('cuda')
            outputs = InPainTor(inputs)

            # Outputs: inpainted_image and masked_out

            # Process outputs
            inpainted_images = outputs['inpainted_image'].cpu().numpy()
            masks = outputs['mask'].cpu().numpy()

            # Save the masks
            for j in range(batch_size):
                mask = masks[j].transpose(1, 2, 0)
                mask = np.argmax(mask, axis=2)
                mask = np.vectorize(class_dict.get)(mask).astype(np.uint8)
                mask = Image.fromarray(mask)
                mask.save(os.path.join(output_dir, f'mask_{i * batch_size + j}.png'))

                # Save the inpainted images
                inpainted_image = inpainted_images[j].transpose(1, 2, 0)
                inpainted_image = (inpainted_image * 255).astype(np.uint8)
                inpainted_image = Image.fromarray(inpainted_image)
                inpainted_image.save(os.path.join(output_dir, f'inpainted_image_{i * batch_size + j}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on the test set')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data/CamVid', help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the output masks')

    args = parser.parse_args()

    # Load class_dict
    with open('assets/coco_91_classes.json', 'r') as f:
        class_dict = json.load

    inference(args.model_path, args.data_dir, (args.image_size, args.image_size), (args.mask_size, args.mask_size),
              args.batch_size, class_dict, args.output_dir)
