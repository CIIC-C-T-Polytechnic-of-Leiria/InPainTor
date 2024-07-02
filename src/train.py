"""
 train.py

 Usage example:
    python src/train.py --data_dir=/media/tiagociiic/easystore/RORD --num_epochs=200 --batch_size=2 --image_size=512
 """

# TODO: refactor save_images method to be a class method of Trainer
# TODO: verify logging and file saving

import argparse
import importlib
import os
from datetime import datetime

import loguru
import torch
import torch.optim as optim
from torch.nn import Module, MSELoss, NLLLoss
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

from data_augmentation import RandAugment
from dataset import RORDDataset
from model import InpainTor
from visualizations import plot_training_log, save_train_images

loguru.logger.remove()  # Remove the default logger
log_file = f"logs/train{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}.log"
loguru.logger.add(log_file, rotation="10 MB")


class Trainer:
    def __init__(self, model_: Module,
                 loss: callable,
                 optimizer_: optim.Optimizer,
                 device_: torch.device,
                 training_loader: DataLoader,
                 valid_loader: DataLoader,
                 model_name: str,
                 log_interval: int,
                 scheduler_: ReduceLROnPlateau,
                 augment: bool = False,
                 num_augment_operations: int = 3,
                 augment_magnitude: int = 5,
                 lambda_: float = 0.5,
                 criterion_segment: callable = None,
                 criterion_inpaint: callable = None):
        self.lambda_ = lambda_
        # self.logger = loguru.logger
        self.logger = loguru.logger.bind(trainer=True)
        self.best_model_path = None
        self.best_val_loss = float('inf')
        self.model = model_
        self.loss = loss
        self.seg_loss = criterion_segment
        self.inpaint_loss = criterion_inpaint
        self.optim = optimizer_
        self.device = device_
        self.train_loader = training_loader
        self.val_loader = valid_loader
        self.model_name = model_name
        # self.log_train = ModelPerformanceTracker(model_name)
        self.augment = augment
        self.num_augment_operations = num_augment_operations
        self.scheduler = scheduler_
        self.augment_magnitude = augment_magnitude
        self.log_interval = log_interval
        self.global_step = 0
        self.rand_augment = RandAugment(num_operations=num_augment_operations,
                                        magnitude=augment_magnitude) if augment else None

    def save_best_model(self):
        date = datetime.now().strftime('%Y-%m-%d')
        model_path = os.path.join("checkpoints", f'best_{self.model_name}_{date}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            self.scheduler.step(val_loss)

            self.logger.info(
                f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.6f}, '
                f'Best Val Loss: {self.best_val_loss:.6f}, '
                f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_best_model()
                self.logger.info(f'Saving best model to {self.best_model_path}')

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        with tqdm(self.train_loader, desc=f'Epoch {epoch + 1:<2} - Training', unit_scale=True,
                  colour='green', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
            for batch_idx, batch in enumerate(pbar):
                inputs, seg_gt, inpaint_gt = [batch[k].to(self.device) for k in ['image', 'mask', 'gt']]
                self.optim.zero_grad()
                output = self.model(inputs)
                loss = self.loss(output, seg_gt, inpaint_gt)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

                self.global_step += 1
                if self.global_step % self.log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    self.logger.info(
                        f'Epoch {epoch + 1}, Step {self.global_step}, Train Loss: {avg_loss:.6f}, '
                        f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    )
                    save_train_images(self.global_step, inputs, seg_gt, inpaint_gt, output)

                pbar.set_description(f'Epoch {epoch + 1:<2} - Training, Loss: {total_loss / (batch_idx + 1):.6f}')

        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f'Epoch {epoch + 1:<2} - Validation', unit_scale=True,
                      colour='blue', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    inputs, seg_gt, inpaint_gt = [batch[k].to(self.device) for k in ['image', 'mask', 'gt']]
                    output = self.model(inputs)
                    loss = self.loss(output, seg_gt, inpaint_gt)
                    total_val_loss += loss.item()

                    if self.global_step % self.log_interval == 0:
                        avg_val_loss = total_val_loss / (batch_idx + 1)
                        self.logger.info(
                            f'Epoch {epoch + 1}, Step {self.global_step}, Val Loss: {avg_val_loss:.6f}, '
                            f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                        )
                        save_train_images(self.global_step, inputs, seg_gt, inpaint_gt, output, is_validation=True)

                    pbar.set_description(
                        f'Epoch {epoch + 1:<2} - Validation, Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

        return total_val_loss / len(self.val_loader)


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train the InPainTor model.')
    parser.add_argument('--data_dir', type=str, help='Path to dataset dir.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optim')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--num_augment_operations', type=int, default=3, help='Number of data augmentation operations')
    parser.add_argument('--augment_magnitude', type=int, default=5, help='Magnitude of data augmentation operations')
    parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model')
    parser.add_argument('--lambda_', type=float, default=0.99, help='Weight for the composite loss')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--log_interval', type=int, default=1000, help='Log interval for training')
    parser.add_argument('--find_lr', action='store_true', help='Run learning rate finder')
    parser.add_argument('--selected_classes', type=int, nargs='+', default=[0],
                        help='List of classes IDs for inpainting (default:: person)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        train_dataset = RORDDataset(root_dir=args.data_dir, split='debug',
                                    image_size=[args.image_size, args.image_size])
        val_dataset = RORDDataset(root_dir=args.data_dir, split='debug', image_size=[args.image_size, args.image_size])
        print("Debug mode enabled. Using debug dataset.")
    else:
        train_dataset = RORDDataset(root_dir=args.data_dir, split='train',
                                    image_size=[args.image_size, args.image_size])
        val_dataset = RORDDataset(root_dir=args.data_dir, split='val', image_size=[args.image_size, args.image_size])

    loguru.logger.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = InpainTor(num_classes=80, selected_classes=args.selected_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=10)

    criterion_segmentation = NLLLoss()
    criterion_inpainting = MSELoss()
    composite_loss = losses.CompositeLoss(criterion_segmentation, criterion_inpainting, args.lambda_)

    if args.find_lr:
        raise NotImplementedError("Learning rate finder not implemented yet.")
    else:
        trainer = Trainer(model_=model,
                          loss=composite_loss,
                          optimizer_=optimizer,
                          device_=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          log_interval=args.log_interval,
                          training_loader=train_loader,
                          valid_loader=val_loader,
                          model_name=args.model_name,
                          augment=args.augment,
                          num_augment_operations=args.num_augment_operations,
                          augment_magnitude=args.augment_magnitude,
                          lambda_=args.lambda_,
                          scheduler=scheduler,
                          criterion_segment=criterion_segmentation,
                          criterion_inpaint=criterion_inpainting)

        trainer.train(args.num_epochs)
        plot_training_log(file_path=log_file)
