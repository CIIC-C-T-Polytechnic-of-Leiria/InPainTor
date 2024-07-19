"""
    train_v2.py

    This script is the main entry point for training the InPainTor model. It uses the Trainer class to train the model

    Usage:
        python train_v2.py --coco_data_dir path/to/coco --rord_data_dir path/to/rord --num_epochs_seg 10 --num_epochs_inpaint 10 --batch_size 2 --learning_rate 0.1 --image_size 512 --mask_size 256 --model_name inpaintor --log_interval 1000 --selected_classes 1 72 73 77

"""

import argparse
import importlib
import os
from datetime import datetime

import loguru
import torch
from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
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
from losses import InpaintingLoss, SegmentationFocalLoss
from visualizations import plot_training_log
from dataset import COCOSegmentationDataset, RORDInpaintingDataset
from torch.optim import Optimizer

loguru.logger.remove()
log_file = f"logs/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
loguru.logger.add(log_file, rotation="10 MB")


class Trainer:
    def __init__(self, model_: Module, seg_loss: callable, inpaint_loss: callable, optimizer_: Optimizer,
                 device_: torch.device, seg_train_loader: DataLoader, seg_val_loader: DataLoader,
                 inpaint_train_loader: DataLoader, inpaint_val_loader: DataLoader, model_name: str,
                 log_interval: int, scheduler_: ReduceLROnPlateau, initial_lr: float):
        self.model = model_
        self.seg_loss = seg_loss
        self.inpaint_loss = inpaint_loss
        self.optimizer = optimizer_
        self.device = device_
        self.seg_train_loader = seg_train_loader
        self.seg_val_loader = seg_val_loader
        self.inpaint_train_loader = inpaint_train_loader
        self.inpaint_val_loader = inpaint_val_loader
        self.model_name = model_name
        self.scheduler = scheduler_
        self.log_interval = log_interval
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.logger = loguru.logger.bind(trainer=True)
        self.initial_lr = initial_lr

    def save_checkpoint(self, epoch, phase, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }

        if is_best:
            filename = f'checkpoints/best_{self.model_name}_{phase}.pth'
        else:
            filename = f'checkpoints/{self.model_name}_{phase}_epoch{epoch}.pth'

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
            print(f"Checkpoint file {filename} does not exist.")
            return 0

        checkpoint = torch.load(filename)

        # Directly load the checkpoint into the model
        self.model.load_state_dict(checkpoint)

        # Check if other necessary states like optimizer, scheduler, and epoch are available
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

        print(f"Checkpoint loaded from {filename}. Resuming from epoch {start_epoch}.")
        return start_epoch

    def train_segmentation(self, num_epoch_seg: int, resume_from=None):
        self.model.freeze_generator()
        self.model.unfreeze_encoder_and_segmentor()
        self.train_phase(num_epoch_seg, "segmentation", self.seg_train_loader, self.seg_val_loader, resume_from)

        # self.model.save_encoder_and_segmentor("checkpoints/encoder_segmentor_checkpoint.pth")

    def train_inpainting(self, num_epoch_inpaint: int, resume_from=None):
        self.model.load_encoder_and_segmentor("checkpoints/encoder_segmentor_checkpoint.pth")
        self.model.freeze_encoder_and_segmentor()
        self.model.unfreeze_generator()

        # Reset the learning rate for inpainting
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

        # Reset the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.25, patience=2)

        self.logger.info(f"Reset learning rate to {self.initial_lr} for inpainting stage")

        self.train_phase(num_epoch_inpaint, "inpainting", self.inpaint_train_loader, self.inpaint_val_loader,
                         resume_from)

    def train_phase(self, num_epochs, phase, train_loader, val_loader, resume_from=None):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            if not start_epoch:
                print("Failed to load checkpoint. Starting from epoch 0.")
                start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_epoch(epoch, phase, train_loader)
            val_loss = self.validate_epoch(epoch, phase, val_loader)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            self.scheduler.step(val_loss)

            # Save regular checkpoint
            self.save_checkpoint(epoch, phase)

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
                if self.global_step % 500 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    self.logger.info(
                        f'{phase.capitalize()} - Epoch {epoch + 1}, Step {self.global_step}, '
                        f'Train Loss: {avg_loss:.6f}')
                    # save_train_images(self.global_step, images, gt if phase == "inpainting" else None, output,
                    #                   phase=phase)
                    """
                        Parameters:
                            step (int): The current training step.
                            inputs (torch.Tensor): The input images.
                            seg_target (torch.Tensor): The segmentation target.
                            inpaint_target (torch.Tensor): The inpainting target (can be None for segmentation phase).
                            outputs (dict): The model outputs.
                            phase (str): The current training phase ('segmentation' or 'inpainting').
                            is_validation (bool): Whether the images are from the validation set.
                            save_path (str): The directory to save the images.
                            selected_classes (list): List of selected class indices for visualization.
                            mean (list): Mean values used for normalization.
                            std (list): Standard deviation values used for normalization.
                        """
                    # save_train_images(step=self.global_step, inputs=images,
                    #                   seg_target=gt if phase == "inpainting" else None,
                    #                   inpaint_target=gt if phase == "inpainting" else None, outputs=output, phase=phase,
                    #                   is_validation=False, selected_classes=[1, 72, 73, 77])
                    self.scheduler.step(avg_loss)

                if torch.isnan(loss):
                    print("NaN loss detected!")
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
                        avg_val_loss = total_val_loss / (batch_idx + 1)
                        self.logger.info(
                            f'{phase.capitalize()} - Epoch {epoch + 1}, Step {self.global_step}, '
                            f'Val Loss: {avg_val_loss:.6f}')
                        # save_train_images(step=self.global_step, inputs=images,
                        #                   seg_target=gt if phase == "inpainting" else None,
                        #                   inpaint_target=gt if phase == "inpainting" else None, outputs=output,
                        #                   phase=phase,
                        #                   is_validation=True, selected_classes=[1, 72, 73, 77])

                    pbar.set_description(
                        f'Epoch {epoch + 1} - {phase.capitalize()} Validation, '
                        f'Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

                    del images, output, loss
                    torch.cuda.empty_cache()

        return total_val_loss / len(data_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the InPainTor model_.')
    parser.add_argument('--coco_data_dir', type=str, default='/media/tiagociiic/easystore/COCO_dataset',
                        help='Path to COCO dataset dir.')
    parser.add_argument('--rord_data_dir', type=str, default='/media/tiagociiic/easystore/RORD_dataset',
                        help='Path to RORD dataset dir.')
    parser.add_argument('--seg_epochs', type=int, default=10, help='Number of epochs to train segmentation')
    parser.add_argument('--inpaint_epochs', type=int, default=10, help='Number of epochs to train inpainting')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer_')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--model_name', type=str, default='InPainTor', help='Name of the model_')
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
                                               image_size=(args.image_size, args.image_size),
                                               transform=transforms.Compose([
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                               ])
                                               )

    rord_val_dataset = RORDInpaintingDataset(root_dir=args.rord_data_dir, split='val',
                                             image_size=(args.image_size, args.image_size),
                                             transform=transforms.Compose([
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                             ])
                                             )

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
    seg_loss = SegmentationFocalLoss().to(device)
    inpaint_loss = InpaintingLoss().to(device)

    trainer = Trainer(model_=model,
                      seg_loss=seg_loss,
                      inpaint_loss=inpaint_loss,
                      optimizer_=optimizer,
                      device_=device,
                      seg_train_loader=coco_train_loader,
                      seg_val_loader=coco_val_loader,
                      inpaint_train_loader=rord_train_loader,
                      inpaint_val_loader=rord_val_loader,
                      model_name=args.model_name,
                      log_interval=args.log_interval,
                      scheduler_=scheduler,
                      initial_lr=args.learning_rate)

    # Training stages
    trainer.train_segmentation(args.seg_epochs, args.resume_checkpoint)
    trainer.train_inpainting(args.inpaint_epochs, args.resume_checkpoint)

    # Plot training log
    plot_training_log(file_path=loguru.logger.handlers[0].baseFilename)
