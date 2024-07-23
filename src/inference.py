"""
inference.py

    Performs inference on the test set using the trained InPainTor.

    Usage:
        python src/inference.py --model_path "path/to/model.pth" --data_dir "path/to/data" --image_size 512 --mask_size 256 --batch_size 1 --output_dir "path/to/outputs"
"""

import argparse
import importlib
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

# Update and reload modules to reflect recent changes
import dataset
import model

importlib.reload(dataset)
importlib.reload(model)

from dataset import RORDInpaintingDataset  # Adjust according to the updated dataset
from model import InpainTor  # Adjust according to the updated model


def inference(model_path: str, data_dir: str, image_size: tuple, mask_size: tuple, batch_size: int,
              selected_classes: list, output_dir: str) -> None:
    """
    Performs inference on the test set and saves the output masks.

    Parameters:
        model_path (str): Path to the trained InPainTor.
        data_dir (str): Path to the dataset directory.
        image_size (tuple): Size of the input images.
        mask_size (tuple): Size of the masks.
        batch_size (int): Batch size for inference.
        selected_classes (list): List of classes IDs for inpainting.
        output_dir (str): Directory to save the output masks.
    """

    # Create test DataLoader
    test_dataset = RORDInpaintingDataset(root_dir=data_dir, split='debug', image_size=image_size,
                                         transform=transforms.Compose([
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                         ]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the InPainTor
    InPainTor = InpainTor(selected_classes=selected_classes)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cuda'))

    # Load only the model state dict
    InPainTor.load_state_dict(checkpoint['model_state_dict'])

    InPainTor.eval()
    InPainTor.to('cuda')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Inference
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Inference')):
            inputs = batch['image'].to('cuda')
            outputs = InPainTor(inputs)

            # Process outputs
            input_images = inputs.cpu().numpy()
            inpainted_images = outputs['inpainted_image'].cpu().numpy()
            masks = outputs['mask'].cpu().numpy()

            for j in range(batch_size):
                # Convert images from numpy arrays to PIL Images
                input_image = Image.fromarray((input_images[j].transpose(1, 2, 0) * 255).astype(np.uint8))
                inpainted_image = Image.fromarray((inpainted_images[j].transpose(1, 2, 0) * 255).astype(np.uint8))

                # Handle the mask shape
                print(f"Mask max: {masks[j].max()}, min: {masks[j].min()}")
                mask = masks[j]
                if mask.shape[0] == 1 and mask.shape[1] == 1:
                    mask = mask.squeeze()  # Remove singleton dimensions
                elif mask.shape[0] == 1:
                    mask = mask.squeeze(0)  # Remove only the first dimension if it's singleton

                # Ensure the mask is 2D
                if len(mask.shape) > 2:
                    mask = np.argmax(mask, axis=0)  # If it's one-hot encoded, convert to class labels

                mask_image = Image.fromarray((mask * 255).astype(np.uint8))

                # Create a new image with the three images side by side
                combined_width = input_image.width * 3
                combined_height = input_image.height
                combined_image = Image.new('RGB', (combined_width, combined_height))

                # Paste the images
                combined_image.paste(input_image, (0, 0))
                combined_image.paste(inpainted_image, (input_image.width, 0))
                combined_image.paste(mask_image.convert('RGB'), (input_image.width * 2, 0))

                # Save the combined image
                combined_image.save(os.path.join(output_dir, f'combined_result_{i * batch_size + j}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on the test set')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='/media/tiagociiic/easystore/RORD_dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                        help='Directory to save the output images')
    parser.add_argument('--selected_classes', type=int, nargs='+', default=[1, 72, 73, 77],
                        help='List of classes IDs for inpainting (default: person)')

    args = parser.parse_args()

    inference(args.model_path, args.data_dir, (args.image_size, args.image_size), (args.mask_size, args.mask_size),
              args.batch_size, args.selected_classes, args.output_dir)
