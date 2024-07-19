"""
    train_v2.py

    This script is the main entry point for training the InPainTor model. It uses the Trainer class to train the model

    Usage:
        python train_v2.py --coco_data_dir path/to/coco --rord_data_dir path/to/rord --num_epochs_seg 10 --num_epochs_inpaint 10 --batch_size 2 --learning_rate 0.1 --image_size 512 --mask_size 256 --model_name inpainter --log_interval 1000 --selected_classes 1 72 73 77

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
from losses import SegmentationLoss, InpaintingLoss
from visualizations import save_train_images, plot_training_log
from dataset import COCOSegmentationDataset, RORDInpaintingDataset
from torch.optim import Optimizer

loguru.logger.remove()
log_file = f"logs/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
loguru.logger.add(log_file, rotation="10 MB")


class Trainer:
    def __init__(self, model_: Module, seg_loss: callable, inpaint_loss: callable, optimizer_: Optimizer,
                 device_: torch.device, seg_train_loader: DataLoader, seg_val_loader: DataLoader,
                 inpaint_train_loader: DataLoader, inpaint_val_loader: DataLoader, model_name: str,
                 log_interval: int, scheduler_: ReduceLROnPlateau):
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

    def save_best_model(self, phase: str) -> str:
        date = datetime.now().strftime('%Y-%m-%d')
        model_path = os.path.join("checkpoints", f'best_{self.model_name}_{phase}_{date}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def train_segmentation(self, num_epochs_seg: int):
        self.model.freeze_generator()
        self.model.unfreeze_encoder_and_segmentor()
        self.train_phase(num_epochs_seg, "segmentation", self.seg_train_loader, self.seg_val_loader)
        self.model.save_encoder_and_segmentor("checkpoints/encoder_segmentor_checkpoint.pth")

    def train_inpainting(self, num_epochs_inpaint: int):
        self.model.load_encoder_and_segmentor("checkpoints/encoder_segmentor_checkpoint.pth")
        self.model.freeze_encoder_and_segmentor()
        self.model.unfreeze_generator()
        self.train_phase(num_epochs_inpaint, "inpainting", self.inpaint_train_loader, self.inpaint_val_loader)

    def train_phase(self, num_epochs: int, phase: str, train_loader_: DataLoader, val_loader_: DataLoader):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, phase, train_loader_)
            val_loss = self.validate_epoch(epoch, phase, val_loader_)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, '
                             f'Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = self.save_best_model(phase)
                self.logger.info(f'Saving best {phase} model to {best_model_path}')

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
                masks = batch['mask'].to(self.device)
                self.optimizer.zero_grad()

                output = self.model(images)
                if phase == "segmentation":
                    loss = self.seg_loss(output['mask'], masks)
                else:  # inpainting phase
                    gt = batch['gt'].to(self.device)
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
                    save_train_images(self.global_step, images, masks, gt if phase == "inpainting" else None, output,
                                      phase=phase)
                    self.scheduler.step(avg_loss)

                if torch.isnan(loss):
                    print("NaN loss detected!")
                    return float('inf')  # Early exit on NaN loss

                pbar.set_description(
                    f'Epoch {epoch + 1} - {phase.capitalize()} Training, Loss: {total_loss / (batch_idx + 1):.6f}, LR: {self.optimizer.param_groups[0]["lr"]}')

                del images, masks, output, loss
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
                    masks = batch['mask'].to(self.device)

                    output = self.model(images)
                    if phase == "segmentation":
                        loss = self.seg_loss(output['mask'], masks)
                    else:  # inpainting phase
                        gt = batch['gt'].to(self.device)
                        loss = self.inpaint_loss(output['inpainted_image'], gt)

                    total_val_loss += loss.item()

                    if self.global_step % self.log_interval == 0:
                        avg_val_loss = total_val_loss / (batch_idx + 1)
                        self.logger.info(
                            f'{phase.capitalize()} - Epoch {epoch + 1}, Step {self.global_step}, '
                            f'Val Loss: {avg_val_loss:.6f}')
                        save_train_images(self.global_step, images, masks, gt if phase == "inpainting" else None,
                                          output, is_validation=True,
                                          phase=phase)

                    pbar.set_description(
                        f'Epoch {epoch + 1} - {phase.capitalize()} Validation, '
                        f'Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

                    del images, masks, output, loss
                    torch.cuda.empty_cache()

        return total_val_loss / len(data_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the InPainTor model_.')
    parser.add_argument('--coco_data_dir', type=str, default='/media/tiagociiic/easystore/COCO_dataset',
                        help='Path to COCO dataset dir.')
    parser.add_argument('--rord_data_dir', type=str, default='/media/tiagociiic/easystore/RORD_dataset',
                        help='Path to RORD dataset dir.')
    parser.add_argument('--num_epochs_seg', type=int, default=10, help='Number of epochs to train segmentation')
    parser.add_argument('--num_epochs_inpaint', type=int, default=10, help='Number of epochs to train inpainting')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer_')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--model_name', type=str, default='InPainTor', help='Name of the model_')
    parser.add_argument('--log_interval', type=int, default=1000, help='Log interval for training')
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
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=2, verbose=True)
    segmentation_loss = SegmentationLoss()
    inpainting_loss = InpaintingLoss()

    trainer = Trainer(model_=model,
                      seg_loss=segmentation_loss,
                      inpaint_loss=inpainting_loss,
                      optimizer_=optimizer,
                      device_=device,
                      seg_train_loader=coco_train_loader,
                      seg_val_loader=coco_val_loader,
                      inpaint_train_loader=rord_train_loader,
                      inpaint_val_loader=rord_val_loader,
                      model_name=args.model_name,
                      log_interval=args.log_interval,
                      scheduler_=scheduler)

    # Train segmentation
    trainer.train_segmentation(args.num_epochs_seg)

    # Train inpainting
    trainer.train_inpainting(args.num_epochs_inpaint)

    # Plot training log
    plot_training_log(file_path=loguru.logger.handlers[0].baseFilename)
