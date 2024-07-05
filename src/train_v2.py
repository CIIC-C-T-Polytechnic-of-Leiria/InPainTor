"""

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
from dataset import COCOSegmentationDataset

loguru.logger.remove()
log_file = f"logs/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
loguru.logger.add(log_file, rotation="10 MB")


class Trainer:
    def __init__(self, model_: Module, seg_loss: callable, inpaint_loss: callable, optimizer_: optim.Optimizer,
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
        self.model.unfreeze_segmenter()
        self.train_phase(num_epochs_seg, "segmentation", self.seg_train_loader, self.seg_val_loader)
        self.model.save_segmenter("checkpoints/segmenter_checkpoint.pth")

    def train_inpainting(self, num_epochs_inpaint: int):
        self.model.load_segmenter("checkpoints/segmenter_checkpoint.pth")
        self.model.freeze_segmenter()
        self.model.unfreeze_generator()
        self.train_phase(num_epochs_inpaint, "inpainting", self.inpaint_train_loader, self.inpaint_val_loader)

    def train_phase(self, num_epochs: int, phase: str, train_loader_: DataLoader, val_loader_: DataLoader):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, phase, train_loader_)
            val_loss = self.validate_epoch(epoch, phase, val_loader_)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = self.save_best_model(phase)
                self.logger.info(f'Saving best {phase} model_ to {best_model_path}')

    # TODO: Change and test the dataset classes
    # TODO: Choose the right classes for inpainting
    # TODO: Choose only 7 classes for segmentation: person, car, motorcycle, laptop, cell phone, book, television

    def train_epoch(self, epoch: int, phase: str, data_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        with tqdm(data_loader, desc=f'Epoch {epoch + 1} - {phase.capitalize()} Training', unit_scale=True,
                  colour='green') as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device) if phase == "segmentation" else batch['gt'].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.seg_loss(output['mask'], masks) if phase == "segmentation" else self.inpaint_loss(
                    output['inpainted_image'], masks)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                self.global_step += 1
                if self.global_step % self.log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    self.logger.info(
                        f'{phase.capitalize()} - Epoch {epoch + 1}, Step {self.global_step}, Train Loss: {avg_loss:.6f}')
                    save_train_images(self.global_step, images, masks, None, output, phase=phase)

                pbar.set_description(
                    f'Epoch {epoch + 1} - {phase.capitalize()} Training, Loss: {total_loss / (batch_idx + 1):.6f}')

        return total_loss / len(data_loader)

    def validate_epoch(self, epoch: int, phase: str, data_loader: DataLoader) -> float:
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(data_loader, desc=f'Epoch {epoch + 1} - {phase.capitalize()} Validation', unit_scale=True,
                      colour='blue') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device) if phase == "segmentation" else batch['gt'].to(self.device)
                    output = self.model(images)
                    loss = self.seg_loss(output['mask'], masks) if phase == "segmentation" else self.inpaint_loss(
                        output['inpainted_image'], masks)
                    total_val_loss += loss.item()

                    if self.global_step % self.log_interval == 0:
                        avg_val_loss = total_val_loss / (batch_idx + 1)
                        self.logger.info(
                            f'{phase.capitalize()} - Epoch {epoch + 1}, Step {self.global_step}, Val Loss: {avg_val_loss:.6f}')
                        save_train_images(self.global_step, images, masks, None, output, is_validation=True,
                                          phase=phase)

                    pbar.set_description(
                        f'Epoch {epoch + 1} - {phase.capitalize()} Validation, Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

        return total_val_loss / len(data_loader)


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train the InPainTor model_.')
    parser.add_argument('--data_dir', type=str, help='Path to dataset dir.')
    parser.add_argument('--num_epochs_seg', type=int, default=10, help='Number of epochs to train segmentation')
    parser.add_argument('--num_epochs_inpaint', type=int, default=10, help='Number of epochs to train inpainting')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer_')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--num_augment_operations', type=int, default=3, help='Number of data augmentation operations')
    parser.add_argument('--augment_magnitude', type=int, default=5, help='Magnitude of data augmentation operations')
    parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model_')
    parser.add_argument('--lambda_', type=float, default=0.99, help='Weight for the composite loss')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--log_interval', type=int, default=1000, help='Log interval for training')
    parser.add_argument('--find_lr', action='store_true', help='Run learning rate finder')
    parser.add_argument('--selected_classes', type=int, nargs='+', default=[0],
                        help='List of classes IDs for inpainting (default: person)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        train_dataset = dataset.COCOSegmentationDataset(root_dir=args.data_dir, split='debug', year='2017',
                                                        image_size=(args.image_size, args.image_size))
        val_dataset = COCOSegmentationDataset(root_dir=args.data_dir, split='debug', year='2017',
                                              image_size=(args.image_size, args.image_size))
        print("Debug mode enabled. Using debug dataset.")
    else:
        train_dataset = dataset.COCOSegmentationDataset(root_dir=args.data_dir, split='train', year='2017',
                                                        image_size=(args.image_size, args.image_size))
        val_dataset = COCOSegmentationDataset(root_dir=args.data_dir, split='val', year='2017',
                                              image_size=(args.image_size, args.image_size))

    loguru.logger.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = InpainTor(num_classes=80, selected_classes=[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=10)
    segmentation_loss = SegmentationLoss()
    inpainting_loss = InpaintingLoss()

    if args.find_lr:
        raise NotImplementedError("Learning rate finder not implemented yet.")
    else:
        trainer = Trainer(model_=model,
                          seg_loss=segmentation_loss,
                          inpaint_loss=inpainting_loss,
                          optimizer_=optimizer,
                          device_=device,
                          seg_train_loader=train_loader,
                          seg_val_loader=val_loader,
                          inpaint_train_loader=train_loader,
                          inpaint_val_loader=val_loader,
                          model_name=args.model_name,
                          log_interval=args.log_interval,
                          scheduler_=scheduler)

        # Train segmentation
        trainer.train_segmentation(args.num_epochs_seg)

        # Train inpainting
        trainer.train_inpainting(args.num_epochs_inpaint)

        # Plot training log
        plot_training_log(file_path=loguru.logger.handlers[0].baseFilename)
