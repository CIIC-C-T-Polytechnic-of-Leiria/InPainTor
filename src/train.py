"""
train.py

    Description:
    This script trains the InPainTor model for segmentation and inpainting tasks using COCO and RORD datasets, respectively.
    The Trainer class handles the training and validation processes for both phases, logging progress, and saving model checkpoints.

    Usage:
    1. Configure the training parameters and paths by editing the argument defaults or passing them via command line.
    2. Run the script:
       ```bash
       python train_inpaintor.py --coco_data_dir <path_to_COCO> --rord_data_dir <path_to_RORD> --seg_epochs <num_epochs> --inpaint_epochs <num_epochs>

        Arguments:
        --coco_data_dir (str): Path to the COCO 2017 dataset directory. Default is '/media/tiagociiic/easystore/COCO_dataset'.
        --rord_data_dir (str): Path to the RORD dataset directory. Default is '/media/tiagociiic/easystore/RORD_dataset'.
        --seg_epochs (int): Number of epochs for segmentation training. Default is 10.
        --inpaint_epochs (int): Number of epochs for inpainting training. Default is 10.
        --batch_size (int): Batch size for training. Default is 2.
        --learning_rate (float): Learning rate for the optimizer. Default is 0.1.
        --image_size (int): Size of the input images, assumed to be square. Default is 512.
        --mask_size (int): Size of the masks, assumed to be square. Default is 256.
        --model_name (str): Name of the model. Default is 'InPainTor'.
        --log_interval (int): Log interval for training. Default is 1000.
        --resume_checkpoint (str): Path to the checkpoint to resume training from. Default is None.
        --selected_classes (int, nargs='+'): List of class IDs for inpainting . Default is [1, 72, 73, 77] (1 - person, 72 - tv, 73 - laptop, 77 - cell phone)
           see here: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
"""

import argparse
import importlib
import os
from datetime import datetime

import loguru
import torch
from torch import optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import losses
import model
import visualizations

# Reload the modules in case they have been modified
importlib.reload(dataset)
importlib.reload(model)
importlib.reload(losses)
importlib.reload(visualizations)

from model import InpainTor
from losses import InpaintingLoss
from dataset import COCOSegmentationDataset, RORDInpaintingDataset
from visualizations import plot_training_log

loguru.logger.remove()
log_file = f"logs/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
loguru.logger.add(log_file, rotation="10 MB")


class Trainer:
    """
    Trainer class to handle training and validation of the InPainTor model.

    Args:
        model_local (Module): The InPainTor model to be trained
        seg_loss_local (callable): Segmentation loss function
        inpaint_loss_local (callable): Inpainting loss function
        optimizer_local (Optimizer): The optimizer to be used for training
        device_local (torch.device): The device to be used for training
        seg_train_loader (DataLoader): DataLoader for the segmentation training dataset
        seg_val_loader (DataLoader): DataLoader for the segmentation validation dataset
        inpaint_train_loader (DataLoader): DataLoader for the inpainting training dataset
        inpaint_val_loader (DataLoader): DataLoader for the inpainting validation dataset
        model_name (str): Name of the model_
        log_interval (int): Interval for logging training progress
        scheduler_local (ReduceLROnPlateau): Learning rate scheduler
        initial_lr (float): Initial learning
    """

    def __init__(self, model_local: Module, seg_loss_local: callable,
                 inpaint_loss_local: callable, optimizer_local: Optimizer,
                 device_local: torch.device, seg_train_loader: DataLoader,
                 seg_val_loader: DataLoader, inpaint_train_loader: DataLoader,
                 inpaint_val_loader: DataLoader, model_name: str,
                 log_interval: int, scheduler_local: ReduceLROnPlateau, initial_lr: float):
        self.model = model_local
        self.seg_loss = seg_loss_local
        self.inpaint_loss = inpaint_loss_local
        self.optimizer = optimizer_local
        self.device = device_local
        self.seg_train_loader = seg_train_loader
        self.seg_val_loader = seg_val_loader
        self.inpaint_train_loader = inpaint_train_loader
        self.inpaint_val_loader = inpaint_val_loader
        self.model_name = model_name
        self.scheduler = scheduler_local
        self.log_interval = log_interval
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.logger = loguru.logger.bind(trainer=True)
        self.initial_lr = initial_lr

    def log_and_save_images(self, phase, epoch, batch_idx, total_loss, images, gt, output, is_validation=False):
        """
        Log training progress and save images for visualization.

        Args:
            phase (str): The current phase of training
            epoch (int): The current epoch
            batch_idx (int): The current batch index
            total_loss (float): The total loss for the current phase
            images (Tensor): The input images
            gt (Tensor): The ground truth images
            output (Tensor): The model output
            is_validation (bool): Whether the current phase is validation
        """

        avg_loss = total_loss / (batch_idx + 1)
        self.logger.info(
            f'{phase.capitalize()} - Epoch {epoch + 1}, Step {self.global_step}, '
            f'Train Loss: {avg_loss:.6f}')

        visualizations.save_train_images_v2(
            step=self.global_step,
            inputs=images,
            seg_target=gt if phase == "segmentation" else None,
            inpaint_target=gt if phase == "inpainting" else None,
            outputs=output,
            phase=phase,
            is_validation=is_validation,
            threshold=0.5
        )

        return avg_loss

    def save_checkpoint(self, epoch: int, phase: str, is_best: bool = False):
        if phase == "segmentation":
            state_dict = self.model.get_state_dict("encoder_segmentor")
        else:  # inpainting phase
            state_dict = self.model.get_state_dict("full")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }

        filename = (f'checkpoints/best_{self.model_name}_{phase}.pth' if is_best
                    else f'checkpoints/{self.model_name}_{phase}_epoch{epoch + 1}.pth')

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename: str) -> int:
        """
        Load a checkpoint from a file and restore the model, optimizer, scheduler, and other states.
        """

        if not os.path.exists(filename):
            print(f"Checkpoint file {filename} does not exist.")
            return 0

        checkpoint = torch.load(filename)

        if 'model_state_dict' in checkpoint:
            part = "encoder_segmentor" if "segmentation" in filename else "full"
            self.model.load_state_dict(checkpoint['model_state_dict'], part=part)
        else:
            print("Warning: Model state dict not found in checkpoint.")

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: Optimizer state dict not found in checkpoint.")

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Warning: Scheduler state dict not found in checkpoint.")

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        else:
            print("Warning: Best validation loss not found in checkpoint.")

        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        else:
            print("Warning: Global step not found in checkpoint.")

        start_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded from {filename}. Resuming from epoch {start_epoch}.")
        return start_epoch

    def train_segmentation(self, num_epoch_seg: int, resume_from=None):
        """
        Train the segmentation phase of the model_local.
        The encoder and segmentor are unfrozen, and the generator is frozen.
        """
        self.model.freeze_part("generator")
        self.model.unfreeze_part("encoder_segmentor")
        self.train_phase(num_epoch_seg, "segmentation", self.seg_train_loader, self.seg_val_loader, resume_from)

        # self.model.save_encoder_and_segmentor("checkpoints/encoder_segmentor_checkpoint.pth")

    def train_inpainting(self, inpaint_epochs: int, resume_from=None):
        """
        Train the inpainting phase of the model_local.
        The generator is unfrozen, and the encoder and segmentor are frozen.
        """
        best_seg_checkpoint = f'checkpoints/best_{self.model_name}_segmentation.pth'
        self.load_checkpoint(best_seg_checkpoint)
        self.model.freeze_part("encoder_segmentor")
        self.model.unfreeze_part("generator")

        # Reset the learning rate for inpainting
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

        # Reset the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.25, patience=2)

        self.logger.info(f"Reset learning rate to {self.initial_lr} for inpainting stage")
        self.train_phase(inpaint_epochs, "inpainting", self.inpaint_train_loader, self.inpaint_val_loader,
                         resume_from)

    def train_phase(self, num_epochs: int, phase: str, train_loader: DataLoader, val_loader: DataLoader,
                    resume_from: str = None):
        """
        Train a phase of the model_local (segmentation or inpainting).
        """
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            if not start_epoch:
                self.logger.info("Failed to load checkpoint. Starting from epoch 1.")
                start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            _ = self.train_epoch(epoch, phase, train_loader)
            val_loss = self.validate_epoch(epoch, phase, val_loader)

            self.scheduler.step(val_loss)

            # Save regular checkpoint
            self.save_checkpoint(epoch, phase)

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, phase, is_best=True)

    def train_epoch(self, epoch: int, phase: str, data_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        with tqdm(data_loader, desc=f'Epoch {epoch + 1} - {phase.capitalize()} Training', unit_scale=True,
                  colour='green') as pbar:
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    print(f"Skipping empty batch at index {batch_idx}")
                    continue
                images = batch['image'].to(self.device)
                gt = batch['gt'].to(self.device)

                self.optimizer.zero_grad()
                output = self.model(images)

                if phase == "segmentation":
                    loss = self.seg_loss(output['mask'], gt)
                else:  # inpainting phase
                    loss = self.inpaint_loss(output['inpainted_image'], gt)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

                self.global_step += 1

                # Log training progress
                if self.global_step % 500 == 0:
                    avg_loss = self.log_and_save_images(phase, epoch, batch_idx, total_loss, images, gt, output)
                    self.scheduler.step(avg_loss)

                if torch.isnan(loss):
                    loguru.logger.error(f'Epoch {epoch + 1} - {phase.capitalize()} Training, Loss: NaN: Exiting!')
                    return float('inf')  # Early exit on NaN loss

                pbar.set_description(
                    f'Epoch {epoch + 1} - {phase.capitalize()} Training, Loss: {total_loss / (batch_idx + 1):.6f}, LR: {self.optimizer.param_groups[0]["lr"]}')

                del images, output, loss
                torch.cuda.empty_cache()

        return total_loss / len(data_loader)

    def validate_epoch(self, epoch: int, phase: str, data_loader: DataLoader) -> float:
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(data_loader, desc=f'Epoch {epoch + 1} - {phase.capitalize()} Validation', unit_scale=True,
                      colour='blue') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    if batch is None:
                        print(f"Skipping empty batch at index {batch_idx}")
                        continue
                    images = batch['image'].to(self.device)
                    gt = batch['gt'].to(self.device)

                    output = self.model(images)
                    if phase == "segmentation":
                        loss = self.seg_loss(output['mask'], gt)
                    else:  # inpainting phase
                        loss = self.inpaint_loss(output['inpainted_image'], gt)

                    total_val_loss += loss.item()

                    if self.global_step % self.log_interval == 0:
                        _ = self.log_and_save_images(phase, epoch, batch_idx, total_val_loss, images, gt, output,
                                                     is_validation=True)

                    pbar.set_description(
                        f'Epoch {epoch + 1} - {phase.capitalize()} Validation, '
                        f'Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

                    del images, output, loss
                    torch.cuda.empty_cache()

        return total_val_loss / len(data_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the InPainTor model_local.')
    parser.add_argument('--coco_data_dir', type=str, default='/media/tiagociiic/easystore/COCO_dataset',
                        help='Path to COCO dataset dir.')
    parser.add_argument('--rord_data_dir', type=str, default='/media/tiagociiic/easystore/RORD_dataset',
                        help='Path to RORD dataset dir.')
    parser.add_argument('--seg_epochs', type=int, default=10, help='Number of epochs to train segmentation')
    parser.add_argument('--inpaint_epochs', type=int, default=10, help='Number of epochs to train inpainting')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer_local')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--model_name', type=str, default='InPainTor', help='Name of the model_local')
    parser.add_argument('--log_interval', type=int, default=1000, help='Log interval for training')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to the checkpoint to resume training from',
                        default=None)
    parser.add_argument('--selected_classes', type=int, nargs='+', default=[1, 72, 73, 77],
                        help='List of classes IDs for inpainting (default: person)')

    return parser.parse_args()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    args = parse_args()

    # COCO dataset for segmentation
    coco_train_dataset = COCOSegmentationDataset(root_dir=args.coco_data_dir, split='train', year='2017',
                                                 image_size=(args.image_size, args.image_size),
                                                 mask_size=(args.mask_size, args.mask_size),
                                                 selected_class_ids=args.selected_classes)
    coco_val_dataset = COCOSegmentationDataset(root_dir=args.coco_data_dir, split='val', year='2017',
                                               image_size=(args.image_size, args.image_size),
                                               mask_size=(args.mask_size, args.mask_size),
                                               selected_class_ids=args.selected_classes)

    # RORD dataset for inpainting
    rord_train_dataset = RORDInpaintingDataset(root_dir=args.rord_data_dir, split='train',
                                               image_size=(args.image_size, args.image_size))

    rord_val_dataset = RORDInpaintingDataset(root_dir=args.rord_data_dir, split='val',
                                             image_size=(args.image_size, args.image_size))

    loguru.logger.info(f'COCO Training samples: {len(coco_train_dataset)}, Validation samples: {len(coco_val_dataset)}')
    loguru.logger.info(f'RORD Training samples: {len(rord_train_dataset)}, Validation samples: {len(rord_val_dataset)}')

    coco_train_loader = DataLoader(coco_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    coco_val_loader = DataLoader(coco_val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    rord_train_loader = DataLoader(rord_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    rord_val_loader = DataLoader(rord_val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = InpainTor(selected_classes=args.selected_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=2)

    model = model.to(device)
    seg_loss = losses.SegmentationLoss().to(device)
    inpaint_loss = InpaintingLoss().to(device)

    trainer = Trainer(model_local=model,
                      seg_loss_local=seg_loss,
                      inpaint_loss_local=inpaint_loss,
                      optimizer_local=optimizer,
                      device_local=device,
                      seg_train_loader=coco_train_loader,
                      seg_val_loader=coco_val_loader,
                      inpaint_train_loader=rord_train_loader,
                      inpaint_val_loader=rord_val_loader,
                      model_name=args.model_name,
                      log_interval=args.log_interval,
                      scheduler_local=scheduler,
                      initial_lr=args.learning_rate)

    # Training stages
    trainer.train_segmentation(args.seg_epochs, args.resume_checkpoint)
    trainer.train_inpainting(args.inpaint_epochs, args.resume_checkpoint)

    # Plot training log
    plot_training_log(file_path=log_file)
