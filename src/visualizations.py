import datetime
import os
import random
import re

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

if 'REPO_DIR' not in os.environ:
    os.environ['REPO_DIR'] = '/home/tiagociiic/Projects/InpainTor'

repo_dir = os.environ['REPO_DIR']


def video_from_images(image_folder: str, output_video_path: str, fps: int = 30) -> None:
    """
    Generate a video from a sequence of ordered images.

    Parameters:
    image_folder (str): Path to the folder containing the images.
    output_video_path (str): Path to the output video file.
    fps (int): Frames per second of the video.

    Returns:
    None
    """
    images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if
              img.endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("No images found in the specified folder.")
        return

    def adjust_image_size(image_path: str) -> Image:
        img = Image.open(image_path)
        width, height = img.size
        new_width = (width + 15) // 16 * 16
        new_height = (height + 15) // 16 * 16
        if (new_width, new_height) != (width, height):
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        return img

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for image_path in images:
            image = adjust_image_size(image_path)

    print(f"Video saved at: {output_video_path}")


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


def save_train_images(epoch: int,
                      inputs: torch.Tensor,
                      seg_target: torch.Tensor,
                      inpaint_target: torch.Tensor,
                      outputs: dict,
                      save_path: str = 'logs/images'):
    # Select one or two examples from the validation set
    example_idx = random.randint(0, inputs.size(0) - 1)

    print(
        f"\ninputs.shape: {inputs.shape}, seg_target.shape: {seg_target.shape}, inpaint_target.shape: {inpaint_target.shape}")
    print(
        f"outputs_mask.shape: {outputs['mask'].shape}, outputs_inpainted_image.shape: {outputs['inpainted_image'].shape}\n")
    input_image = inputs[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    inpaint_target_image = inpaint_target[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    output_mask = outputs['mask'][example_idx].cpu().detach().numpy()
    output_image = outputs['inpainted_image'][example_idx].cpu().detach().numpy().transpose(1, 2, 0)

    # Handle seg_target based on its shape
    if seg_target.dim() == 3:  # If it's [B, H, W]
        seg_target_image = seg_target[example_idx].cpu().detach().numpy()
    elif seg_target.dim() == 4:  # If it's [B, C, H, W]
        seg_target_image = seg_target[example_idx].cpu().detach().numpy().transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected shape for seg_target: {seg_target.shape}")

    # Resize tensors to match input tensor size
    output_mask = transforms.ToPILImage()(output_mask).convert('RGB')
    output_mask = transforms.Resize((512, 512))(output_mask)
    output_mask = transforms.ToTensor()(output_mask).permute(1, 2, 0)

    seg_target_image = transforms.ToPILImage()(seg_target_image).convert('RGB')
    seg_target_image = transforms.Resize((512, 512))(seg_target_image)
    seg_target_image = transforms.ToTensor()(seg_target_image).permute(1, 2, 0)

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
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    # Add images to subplots
    axs[0, 0].imshow(input_tensor)
    axs[0, 0].set_title('Input Image')
    axs[0, 1].imshow(inpaint_target_tensor)
    axs[0, 1].set_title('Inpaint Target')
    axs[0, 2].imshow(output_tensor)
    axs[0, 2].set_title('Output Image')
    axs[1, 0].imshow(seg_target_image.squeeze(), cmap='tab20')
    axs[1, 0].set_title('Segmentation Target')
    axs[1, 1].imshow(output_mask.squeeze(), cmap='tab20')
    axs[1, 1].set_title('Output Mask')
    axs[1, 2].axis('off')  # Leave this subplot empty

    # Remove axis ticks
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])

    # Save the figure to a file
    plt.savefig(f'{save_path}/grid_image_epoch_{epoch + 1}.png', bbox_inches='tight')
    plt.close(fig)
