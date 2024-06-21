import datetime
import os
import re

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


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
