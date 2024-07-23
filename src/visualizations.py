"""
visualizations.py

    Contains utility functions for visualizing training logs, saving images, and creating videos from images.

    Functions:
        video_from_images: Generate a video from a sequence of ordered images.
        plot_training_log: Plot the training and validation loss from a log file.
        save_images_on_grid: Save a grid of binary images to a file.
        denormalize: Denormalize an image using the provided mean and standard deviation.
        save_train_images_v2: Save a grid of images for visualization during training or validation.
"""

import datetime
import os
import re

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Sort image files based on epoch number
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

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


def plot_training_log(file_path, log_scale=False, log_interval=1):
    """
    Plot the training and validation loss from a log file.
    Parameters:
        file_path (str): Path to the log file.
        log_scale (bool): Whether to use log scale for the y-axis.
        log_interval (int): The interval at which the losses were logged.
    """

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['cmr10']
    mpl.rcParams['axes.formatter.use_mathtext'] = True

    # Regular expression to extract data from each line
    pattern = r".+Epoch (\d+), Loss: ([\d\.]+), Val Loss: ([\d\.]+), Best Val Loss: ([\d\.]+), Timestamp: (.+)"

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

    # Plot the results
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.plot([x for x in epoch if x % log_interval == 0],
             [loss[i] for i in range(len(loss)) if epoch[i] % log_interval == 0], label='Train Loss', marker='o',
             linestyle='-', linewidth=0.5, markersize=2)
    plt.plot([x for x in epoch if x % log_interval == 0],
             [val_loss[i] for i in range(len(val_loss)) if epoch[i] % log_interval == 0], label='Val Loss', marker='o',
             linestyle='-', linewidth=0.5, markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    # Show the plot
    plt.show(block=False)


def save_images_on_grid(binary_images: torch.Tensor, output_path: str) -> None:
    """
    Save a grid of binary images to a file. The images are assumed to be binary masks.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get the batch size from the binary images tensor
    batch_size = binary_images.shape[0]

    # Calculate the number of rows and columns for the grid
    num_rows = int(np.sqrt(batch_size))
    num_cols = int(np.ceil(batch_size / num_rows))

    # Create a grid of images
    grid_image = Image.new(mode='L', size=(num_cols * 256, num_rows * 256))
    for i in range(batch_size):
        row, col = i // num_cols, i % num_cols
        image = binary_images[i, 0, :, :]  # Assuming the first channel is the binary mask
        image = Image.fromarray(image.numpy().astype(np.uint8))
        grid_image.paste(image, (col * 256, row * 256))

    # Save the grid image
    grid_image.save(os.path.join(output_path, f"grid_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"))


def denormalize(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """
    Denormalize an image using the provided mean and standard deviation.
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = image * std + mean
    return np.clip(image, 0, 1)


def save_train_images_v2(step: int,
                         inputs: torch.Tensor,
                         seg_target: torch.Tensor,
                         inpaint_target: torch.Tensor,
                         outputs: dict,
                         phase: str,
                         is_validation: bool = False,
                         save_path: str = 'outputs/training',
                         threshold: float = 0.5) -> None:
    """
    Save a grid of images for visualization during training or validation.

    Parameters:
        step (int): The current training step.
        inputs (torch.Tensor): The input images.
        seg_target (torch.Tensor): The segmentation target.
        inpaint_target (torch.Tensor): The inpainting target.
        outputs (dict): The model outputs.
        phase (str): The current training phase ('segmentation' or 'inpainting').
        is_validation (bool): Whether the images are from the validation set.
        save_path (str): The directory to save the images.
        threshold (float): Threshold value for segmentation masks.
    """

    example_idx = 0

    # Normalize input image
    input_image = inputs[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    # input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    # print(f"DEBUG (save_train_images_v2): Input image shape: {input_image.shape}")

    # Process segmentation target
    if seg_target is not None:
        seg_target_image = seg_target[example_idx].cpu().detach().numpy()
        # print(f"DEBUG (save_train_images_v2): Original seg_target_image shape: {seg_target_image.shape}")

        # Ensure seg_target_image is in the format (height, width, channels)
        if seg_target_image.shape[0] == 4 and seg_target_image.shape[-1] != 4:
            seg_target_image = seg_target_image.transpose(1, 2, 0)

        # print(f"DEBUG (save_train_images_v2): Processed seg_target_image shape: {seg_target_image.shape}")
    else:
        seg_target_image = None

    # Process output mask
    output_mask = outputs['mask'][example_idx].cpu().detach().numpy()
    print(f"DEBUG (save_train_images_v2): Output mask shape: {output_mask.shape}")
    print(
        f"DEBUG (save_train_images_v2): Max value in output mask: {output_mask.max()}, Min value: {output_mask.min()}")

    # Apply sigmoid and thresholding to output mask
    output_mask = (output_mask > threshold).astype(float)
    print(
        f"DEBUG (save_train_images_v2): After threshold Max value in output mask: {output_mask.max()}, Min value: {output_mask.min()}")
    print(f"DEBUG (save_train_images_v2): Number of classes: {output_mask.shape[0]}")

    num_classes = max(output_mask.shape[0], seg_target_image.shape[-1] if seg_target_image is not None else 0)
    # print(f"DEBUG (save_train_images_v2): Number of classes: {num_classes}")

    # Generate colors using tab10 colormap
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_classes))[:, :3]

    # Create composite images for the output mask and segmentation target
    output_mask_composite = np.zeros((*output_mask.shape[1:], 3))
    for i in range(output_mask.shape[0]):
        output_mask_composite += np.expand_dims(output_mask[i], axis=-1) * colors[i]
    output_mask_composite = np.clip(output_mask_composite, 0, 1)

    if seg_target_image is not None:
        seg_target_composite = np.zeros((*seg_target_image.shape[:2], 3))
        for i in range(seg_target_image.shape[-1]):
            seg_target_composite += np.expand_dims(seg_target_image[..., i], axis=-1) * colors[i]
        seg_target_composite = np.clip(seg_target_composite, 0, 1)
    else:
        seg_target_composite = np.zeros_like(input_image)

    # print(f"DEBUG (save_train_images_v2): seg_target_composite shape: {seg_target_composite.shape}")
    # print(f"DEBUG (save_train_images_v2): output_mask_composite shape: {output_mask_composite.shape}")

    # Process inpainting images
    if inpaint_target is not None:
        inpaint_target_image = inpaint_target[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
        inpaint_target_image = (inpaint_target_image - inpaint_target_image.min()) / (
                inpaint_target_image.max() - inpaint_target_image.min())
    else:
        inpaint_target_image = np.zeros_like(input_image)

    if 'inpainted_image' in outputs:
        inpaint_output_image = outputs['inpainted_image'][example_idx].cpu().detach().numpy().transpose(1, 2, 0)
        inpaint_output_image = (inpaint_output_image - inpaint_output_image.min()) / (
                inpaint_output_image.max() - inpaint_output_image.min())
    else:
        inpaint_output_image = np.zeros_like(input_image)

    # Create the figure
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))

    # First row
    axs[0, 0].imshow(input_image)
    axs[0, 0].set_title('Input Image')

    axs[0, 1].imshow(inpaint_target_image)
    axs[0, 1].set_title('Inpainting Target')

    axs[0, 2].imshow(inpaint_output_image)
    axs[0, 2].set_title('Inpainting Output')

    # Second row
    axs[1, 0].imshow(seg_target_composite)
    axs[1, 0].set_title(f'Segmentation Target ({seg_target_composite.shape[0]}x{seg_target_composite.shape[1]})')

    axs[1, 1].imshow(output_mask_composite)
    axs[1, 1].set_title(f'Output Mask Composite ({output_mask_composite.shape[0]}x{output_mask_composite.shape[1]})')

    # Show the first channel of the output mask
    axs[1, 2].imshow(output_mask[0], cmap='gray')
    axs[1, 2].set_title(f'Output Mask - Channel 0 ({output_mask[0].shape[0]}x{output_mask[0].shape[1]})')

    plt.tight_layout()

    # Create the directory to save the images
    prefix = "val" if is_validation else "train"
    save_dir = os.path.join(save_path, f"{prefix}_{phase}_images")
    os.makedirs(save_dir, exist_ok=True)

    # Save the image
    plt.savefig(os.path.join(save_dir, f'{phase}_grid_image_step_{step}.png'), bbox_inches='tight')
    print(f"\nSaved image at: {os.path.join(save_dir, f'{phase}_grid_image_step_{step}.png')}")
    plt.close(fig)
