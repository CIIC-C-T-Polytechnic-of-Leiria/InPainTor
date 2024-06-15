"""
 train.py
"""

# TODO: refactor save_images method to be a class method of Trainer

import argparse
import datetime
import importlib
import os

import loguru
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import logger
import model

# Reload the modules in case they have been modified
importlib.reload(dataset)
importlib.reload(model)
importlib.reload(logger)

from data_augmentation import RandAugment
from dataset import RORDDataset
from logger import ModelPerformanceTracker
from model import InpainTor

loguru.logger.remove()  # Remove the default logger
loguru.logger.add("logs/train.log", level="INFO", rotation="10MB", compression="zip")  # Add a file logger


class Trainer:
    def __init__(self, model_: Module, loss: callable, optimizer_: optim.Optimizer, device_, training_loader,
                 valid_loader, model_name: str, augment: bool = False, num_augment_operations: int = 3,
                 augment_magnitude: int = 5, lambda_: float = 0.5, criterion_segment: callable = None,
                 criterion_inpaint: callable = None):
        self.lambda_ = lambda_
        self.logger = loguru.logger
        self.lambda_ = lambda_
        self.logger = loguru.logger.bind(trainer=True)
        self.best_model_path = None
        self.best_val_loss = float('inf')
        self.model = model_
        self.loss = loss
        self.segmentation_loss = criterion_segment
        self.inpainting_loss = criterion_inpaint
        self.optim = optimizer_
        self.device = device_
        self.train_loader = training_loader
        self.val_loader = valid_loader
        self.model_name = model_name
        self.log_train = ModelPerformanceTracker(model_name)
        self.augment = augment
        self.num_augment_operations = num_augment_operations
        self.augment_magnitude = augment_magnitude
        self.rand_augment = RandAugment(num_operations=num_augment_operations,
                                        magnitude=augment_magnitude) if augment else None

    def save_best_model(self):
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        model_path = os.path.join("checkpoints", f'best_model_{self.model_name}_{date}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            with tqdm(self.train_loader, desc=f'Epoch {epoch + 1:<2} - Training ',
                      bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
                for batch in pbar:
                    inputs = batch['image'].to(self.device)
                    self.optim.zero_grad()
                    outputs = self.model(inputs)
                    seg_target = batch['mask'].to(self.device)
                    inpaint_target = batch['gt'].to(self.device)
                    loss = self.loss(outputs, seg_target, inpaint_target, self.segmentation_loss,
                                     self.inpainting_loss, self.lambda_)

                    loss.backward()
                    self.optim.step()
                    total_loss += loss.item()
                    pbar.set_postfix({'Loss': f'{total_loss / (pbar.n + 1):.6f}'}, refresh=False)

            train_loss = total_loss / len(self.train_loader)

            self.model.eval()
            total_val_loss = 0
            with tqdm(self.val_loader, desc=f'Epoch {epoch + 1:<2} - Validation',
                      bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
                for batch in pbar:
                    inputs, labels = batch['image'].to(self.device), batch['mask'].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.inpainting_loss(outputs['mask'], labels.long())
                    total_val_loss += loss.item()
                    pbar.set_postfix({'Val Loss': f'{total_val_loss / (pbar.n + 1):10.6f}'}, refresh=False)

            val_loss = total_val_loss / len(self.val_loader)

            self.logger.info(
                f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.6f}'
            )

            self.log_train.log_metrics(epoch, train_loss, val_loss)

            self.save_images(epoch, inputs, seg_target, inpaint_target, outputs)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_best_model()
                loguru.logger.info(f'Saving best model to {self.best_model_path}')

    @staticmethod
    def save_images(epoch, inputs, seg_target, inpaint_target, outputs):

        # Select one or two examples from the validation set
        example_idx = 0
        input_image = inputs[example_idx].cpu().numpy().transpose(1, 2, 0)
        # seg_target_image = seg_target[example_idx].cpu().numpy().transpose(1, 2, 0)
        inpaint_target_image = inpaint_target[example_idx].cpu().numpy().transpose(1, 2, 0)
        output_mask = outputs['mask'][example_idx].cpu().numpy().transpose(1, 2, 0)
        output_image = outputs['inpainted_image'][example_idx].cpu().detach().numpy().transpose(1, 2, 0)

        print(f"output_image.shape: {output_image.shape}")

        # Resize input tensor to match input tensor size
        output_mask = transforms.ToPILImage()(output_mask).convert('RGB')
        output_mask = transforms.Resize((512, 512))(output_mask)
        output_mask = transforms.ToTensor()(output_mask).permute(1, 2, 0)

        # Convert numpy arrays back to tensors
        input_tensor = torch.from_numpy(input_image)

        # seg_target_tensor = seg_target_image
        inpaint_target_tensor = torch.from_numpy(inpaint_target_image)
        output_image = transforms.ToTensor()(output_image).permute(1, 2, 0)
        output_tensor = output_image
        print(f"output_tensor.shape: {output_tensor.shape}")

        # Convert the tensor to float32 and normalize it to the range [0, 1]
        input_tensor = input_tensor.float()
        # seg_target_tensor = seg_target_tensor.float()
        inpaint_target_tensor = inpaint_target_tensor.float()
        output_tensor = output_tensor.float()

        # Ensure the tensor is of shape (C, H, W)
        input_tensor = input_tensor
        # seg_target_tensor = seg_target_tensor.permute(2, 0, 1)
        inpaint_target_tensor = inpaint_target_tensor
        output_tensor = output_tensor

        print(
            f"output_tensor.shape: {output_tensor.shape}, output_mask.shape: {output_mask.shape}, inpaint_target_tensor.shape: {inpaint_target_tensor.shape}, input_tensor.shape: {input_tensor.shape}")

        # Save the images to disk in a grid with 2 columns and 2 rows, with titles
        grid = torchvision.utils.make_grid(
            [input_tensor.permute(2, 0, 1), inpaint_target_tensor.permute(2, 0, 1), output_mask.permute(2, 0, 1),
             output_tensor.permute(2, 0, 1)],
            nrow=2)
        torchvision.utils.save_image(grid, f'logs/images/epoch_{epoch}.png', nrow=2)


def composite_loss(
        outputs: dict,
        seg_target: torch.Tensor,
        inpaint_target: torch.Tensor,
        segmentation_loss: callable,
        inpaint_loss: callable,
        lambda_: float
) -> torch.Tensor:
    """
    Computes the composite loss for segmentation and inpainting tasks.

    Args:
        outputs (dict): Dictionary containing the model's output tensors.
        seg_target (torch.Tensor): Segmentation target tensor.
        inpaint_target (torch.Tensor): Inpainting target tensor.
        segmentation_loss (callable): Segmentation loss function.
        inpaint_loss (callable): Inpainting loss function.
        lambda_ (float): Weight for the composite loss.

    Returns:
        torch.Tensor: Composite loss tensor.
    """
    try:
        seg_output, inpaint_output = outputs['mask'], outputs['inpainted_image']
        seg_target_class = torch.argmax(seg_target, dim=1)
        # segmentation_loss_val = segmentation_loss(seg_output, seg_target_class.float())
        segmentation_loss_val = segmentation_loss(seg_output, seg_target_class.long())
        inpainting_loss_val = inpaint_loss(inpaint_output, inpaint_target)
        loss = lambda_ * segmentation_loss_val + (1 - lambda_) * inpainting_loss_val
        return loss
    except KeyError as e:
        loguru.logger.error("Outputs dictionary must contain 'ask' and 'inpainted_image' keys.")
        raise ValueError("Outputs dictionary must contain 'ask' and 'inpainted_image' keys.") from e
    except TypeError as e:
        loguru.logger.error(
            "Invalid input types. Check the types of outputs, seg_target, inpaint_target, segmentation_loss, "
            "and inpaint_loss."
        )
        raise ValueError(
            "Invalid input types. Check the types of outputs, seg_target, inpaint_target, segmentation_loss, "
            "and inpaint_loss."
        ) from e
    except RuntimeError as e:
        loguru.logger.error("Error computing the composite loss. Check the input tensors and loss functions.")
        raise ValueError("Error computing the composite loss. Check the input tensors and loss functions.") from e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the InPainTor model.')
    parser.add_argument('--data_dir', type=str, help='Path to dataset dir.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optim')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--augment', action='store_false', help='Apply data augmentation')
    parser.add_argument('--num_augment_operations', type=int, default=3, help='Number of data augmentation operations')
    parser.add_argument('--augment_magnitude', type=int, default=5, help='Magnitude of data augmentation operations')
    parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model')
    parser.add_argument('--lambda_', type=float, default=0.5, help='Weight for the composite loss')
    parser.add_argument('--selected_classes', type=int, nargs='+', default=[0],
                        help='List of classes IDs for inpainting (default: [0]: person)')

    args = parser.parse_args()

    # Create data loaders
    train_dataset = RORDDataset(root_dir=args.data_dir, split='train', image_size=[args.image_size, args.image_size])
    val_dataset = RORDDataset(root_dir=args.data_dir, split='val', image_size=[args.image_size, args.image_size])

    # For DEBUG purposes TODO: Remove this line
    # train_dataset = RORDDataset(root_dir=args.data_dir, split='debug', image_size=[args.image_size, args.image_size])
    # val_dataset = RORDDataset(root_dir=args.data_dir, split='debug', image_size=[args.image_size, args.image_size])

    loguru.logger.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model, criterion, and optim
    model = InpainTor(num_classes=12, selected_classes=args.selected_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    criterion_segmentation = CrossEntropyLoss()
    criterion_inpainting = MSELoss()

    # Create trainer
    trainer = Trainer(model_=model,
                      loss=composite_loss, optimizer_=optimizer,
                      device_=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      training_loader=train_loader, valid_loader=val_loader, model_name=args.model_name,
                      augment=args.augment, num_augment_operations=args.num_augment_operations,
                      augment_magnitude=args.augment_magnitude, lambda_=args.lambda_,
                      criterion_segment=criterion_segmentation, criterion_inpaint=criterion_inpainting)

    # Train the model
    trainer.train(args.num_epochs)

    # Plot the training and validation loss
    # Plotter('logs/metrics_{args.model_name}.txt').plot(log_scale=True)
