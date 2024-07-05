# This script loads a trained model_, creates a test data loader, and performs inference on the test set. The output masks are saved to files using `torch.save`.
# You can modify the script to save the output masks in a different format or to a different location.
#
# To run the script, you can use the following command:
#     `python src/inference.py --model_path checkpoints/best_model_old.pth --data_dir data/CamVid`
# Replace `path/to/trained/model_.pth` with the path to the trained model_ file, and `path/to/dataset` with the path to the dataset directory.

import argparse
import importlib
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import model

# Reload the modules in case they have been modified
importlib.reload(dataset)
importlib.reload(model)

from dataset import CamVidDataset
from model import InpainTor


def inference(model_path: str, data_dir: str, image_size: tuple, mask_size: tuple, batch_size: int,
              class_dict: dict) -> None:
    # Create test data loader
    test_dataset = CamVidDataset(root_dir=data_dir, split='test', image_size=image_size, mask_size=mask_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model_
    model = InpainTor(num_classes=40)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    model.to('cuda')

    # Inference
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Inference')):
            inputs, _ = batch['image'].to('cuda'), batch['mask'].to('cuda')
            outputs = model(inputs)

            # Save output masks (B, C, H, W) to png using class_dict.json to map class ids to RGB values
            output = outputs.argmax(dim=1).cpu().numpy()  # Get the most likely class for each pixel
            output_rgb = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)

            for class_id, class_info in enumerate(class_dict):
                rgb = [class_info['r'], class_info['g'], class_info['b']]
                output_rgb[output[0] == class_id] = rgb

            # Save output mask
            Image.fromarray(output_rgb).save(os.path.join('outputs', f'output_mask_{i}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on the test set')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model_')
    parser.add_argument('--data_dir', type=str, default='data/CamVid', help='Path to the dataset dir')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')

    args = parser.parse_args()
    # print("Parsed arguments:", args.__dict__)

    # load class_dict
    with open(os.path.join(args.data_dir, 'class_dict.json'), 'r') as f:
        class_dict = json.load(f)

    inference(args.model_path, args.data_dir, (args.image_size, args.image_size), (args.mask_size, args.mask_size),
              args.batch_size, class_dict)
