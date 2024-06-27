import datetime
import os
import re

import imageio
import imageio.v2 as imageio
import matplotlib as mpl
import torch
from PIL import Image
from tqdm import tqdm

if 'REPO_DIR' not in os.environ:
    os.environ['REPO_DIR'] = '/home/tiagociiic/Projects/InpainTor'

repo_dir = os.environ['REPO_DIR']


def video_from_images(image_folder: str, output_video_path: str, fps: int = 30, new_width: int = 1280,
                      new_height: int = 720) -> None:
    """
    Generate a video from a sequence of ordered images.

    Parameters:
    image_folder (str): Path to the folder containing the images.
    output_video_path (str): Path to the output video file.
    fps (int): Frames per second of the video.
    new_width (int): The desired width of the images.
    new_height (int): The desired height of the images.

    Returns:
    None
    """
    # Get all image files from the folder
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the specified folder.")
        return

    print(f"Found {len(image_files)} images. Starting video creation...")

    def adjust_image_size(img: Image.Image, new_width: int, new_height: int) -> Image.Image:
        width, height = img.size
        if (new_width, new_height) != (width, height):
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for image_file in tqdm(image_files, desc="Processing images", unit="image", colour='green'):
            image_path = os.path.join(image_folder, image_file)
            with Image.open(image_path) as img:
                adjusted_img = adjust_image_size(img, new_width, new_height)
                writer.append_data(np.asarray(adjusted_img))

    print(f"Video successfully saved at: {output_video_path}")
    print(f"Video details: {len(image_files)} frames, {fps} fps")


def plot_training_log(file_path, log_scale=False):
    """
    Plot the training and validation loss from a log file.
    Parameters:
        file_path (str): Path to the log file.
        log_scale (bool): Whether to use log scale for the y-axis.
    """

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['cmr10']
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    # Regular expression to extract data from each line
    pattern = r".+Epoch (\d+), Loss: (\d+\.\d+), Val Loss: (\d+\.\d+), Best Val Loss: (\d+\.\d+), Timestamp: (.+)"
    epoch, loss, val_loss, best_val_loss, timestamp = [], [], [], [], []

    # Open the file and read it line by line
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                epoch.append(int(match.group(1)))
                loss.append(float(match.group(2)))
                val_loss.append(float(match.group(3)))
                best_val_loss.append(float(match.group(4)))
                timestamp.append(datetime.datetime.strptime(match.group(5), "%Y-%m-%d %H:%M:%S"))

    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.plot(epoch, loss, label='Train Loss', marker='o', linestyle='-', linewidth=0.5, markersize=1)
    plt.plot(epoch, val_loss, label='Val Loss', marker='o', linestyle='-', linewidth=0.5, markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    # Show the plot
    plt.show()


def save_images_on_grid(binary_images: torch.Tensor, output_path: str) -> None:
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get the batch size from the binary images tensor
    batch_size = binary_images.shape[0]

    # Calculate the number of rows and columns for the grid
    num_rows = int(np.sqrt(batch_size))
    num_cols = int(np.ceil(batch_size / num_rows))

    # Create a grid of images
    grid_image = Image.new('L', (num_cols * 256, num_rows * 256))
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        image = binary_images[i, 0, :, :]  # Assuming the first channel is the binary mask
        image = Image.fromarray(image.numpy().astype(np.uint8))
        grid_image.paste(image, (col * 256, row * 256))

    # Save the grid image
    grid_image.save(os.path.join(output_path, f"grid_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"))


import matplotlib.pyplot as plt
import torch
import random
import numpy as np


def save_train_images(epoch: int,
                      inputs: torch.Tensor,
                      seg_target: torch.Tensor,
                      inpaint_target: torch.Tensor,
                      outputs: dict,
                      save_path: str = 'logs/images',
                      random_index: bool = False) -> None:
    # Select one or two examples from the validation set

    if random_index:
        example_idx = random.randint(0, inputs.size(0) - 1)
    else:
        example_idx = 0

    input_image = inputs[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    inpaint_target_image = inpaint_target[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    output_mask = outputs['mask'][example_idx].cpu().detach().numpy()
    output_image = outputs['inpainted_image'][example_idx].cpu().detach().numpy()

    # Handle seg_target based on its shape
    if seg_target.dim() == 3:  # If it's [B, H, W]
        seg_target_image = seg_target[example_idx].cpu().detach().numpy()
    elif seg_target.dim() == 4:  # If it's [B, C, H, W]
        seg_target_image = seg_target[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected shape for seg_target: {seg_target.shape}")

    # Convert numpy arrays back to tensors
    input_tensor = torch.from_numpy(input_image)
    inpaint_target_tensor = torch.from_numpy(inpaint_target_image)
    output_tensor = torch.from_numpy(output_image).permute(1, 2, 0)

    # Convert the tensor to float32 and normalize it to the range [0, 1]
    input_tensor = input_tensor.float()
    inpaint_target_tensor = inpaint_target_tensor.float()
    output_tensor = output_tensor.float()

    # Guarantee that the values are in the [0, 1] range
    input_tensor = torch.clamp(input_tensor, min=0, max=1)
    inpaint_target_tensor = torch.clamp(inpaint_target_tensor, min=0, max=1)
    output_tensor = torch.clamp(output_tensor, min=0, max=1)

    # Create a grid of images
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(23, 13))

    # Add images to subplots
    axs[0, 0].imshow(input_tensor)
    axs[0, 0].set_title('Input Image')
    axs[0, 1].imshow(inpaint_target_tensor)
    axs[0, 1].set_title('Inpaint Target')
    axs[0, 2].imshow(output_tensor)
    axs[0, 2].set_title('Output Image')
    axs[1, 0].imshow(seg_target_image.squeeze(), cmap='gray')
    axs[1, 0].set_title('Segmentation Target')

    # Handle output_mask - show first two channels
    if output_mask.ndim == 3 and output_mask.shape[0] >= 2:
        axs[1, 1].imshow(output_mask[0], cmap='gray')
        axs[1, 1].set_title('Output Mask (Channel 0)')
        axs[1, 2].imshow(output_mask[1], cmap='gray')
        axs[1, 2].set_title('Output Mask (Channel 1)')
    else:
        axs[1, 1].imshow(output_mask.squeeze(), cmap='gray')
        axs[1, 1].set_title('Output Mask')
        axs[1, 2].axis('off')  # Leave this subplot empty if there's only one channel

    # Save the figure to a file
    plt.savefig(f'{save_path}/grid_image_epoch_{epoch + 1}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
