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
                 augment: bool = False,
                 num_augment_operations: int = 3,
                 augment_magnitude: int = 5,
                 lambda_: float = 0.5,
                 criterion_segment: callable = None,
                 criterion_inpaint: callable = None):
        self.lambda_ = lambda_
        self.logger = loguru.logger
        self.lambda_ = lambda_
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
        self.augment_magnitude = augment_magnitude
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
            val_loss = self.validate_epoch(epoch, save_images=True)

            self.logger.info(
                f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.6f}, '
                f'Best Val Loss: {self.best_val_loss:.6f}, '
                f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_best_model()
                self.logger.info(f'Saving best model to {self.best_model_path}')

    def train_epoch(self, epoch: int, save_images: bool = False) -> float:
        self.model.train()
        total_loss = 0
        with tqdm(self.train_loader, desc=f'Epoch {epoch + 1:<2} - Training', unit_scale=True,
                  colour='green', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
            for batch in pbar:
                inputs, seg_gt, inpaint_gt = [batch[k].to(self.device) for k in ['image', 'mask', 'gt']]
                self.optim.zero_grad()
                output = self.model(inputs)
                loss = self.loss(output, seg_gt, inpaint_gt)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                pbar.set_description(f'Epoch {epoch + 1:<2} - Training, Loss: {total_loss / (pbar.n + 1):.6f}')
                if save_images:
                    save_train_images(epoch, inputs, seg_gt, inpaint_gt, output)
        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch: int, save_images: bool = False) -> float:
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f'Epoch {epoch + 1:<2} - Validation', unit_scale=True,
                      colour='blue', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
                for batch in pbar:
                    inputs, seg_gt, inpaint_gt = [batch[k].to(self.device) for k in ['image', 'mask', 'gt']]
                    output = self.model(inputs)
                    loss = self.loss(output, seg_gt, inpaint_gt)
                    total_val_loss += loss.item()
                    pbar.set_description(
                        f'Epoch {epoch + 1:<2} - Validation, Val Loss: {total_val_loss / (pbar.n + 1):.6f}')
                    if save_images:
                        save_train_images(epoch, inputs, seg_gt, inpaint_gt, output)
        return total_val_loss / len(self.val_loader)
    #
    # def train(self, num_epochs: int):
    #     for epoch in range(num_epochs):
    #         self.model.train()
    #         total_loss = 0
    #         with tqdm(self.train_loader, desc=f'Epoch {epoch + 1:<2} - Training', unit_scale=True,
    #                   colour='green', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
    #             for batch in pbar:
    #                 # inputs, seg_gt, inpaint_gt = [x.to(self.device) for x in batch.values()]
    #                 inputs, seg_gt, inpaint_gt = [batch[k].to(self.device) for k in ['image', 'mask', 'gt']]
    #                 self.optim.zero_grad()
    #                 output = self.model(inputs)
    #                 # args_loss = (output, seg_gt, inpaint_gt, self.seg_loss, self.inpaint_loss, self.lambda_)
    #                 # loss = composite_loss(*args_loss)
    #                 loss = self.loss(output, seg_gt, inpaint_gt)
    #                 loss.backward()
    #                 self.optim.step()
    #                 total_loss += loss.item()
    #                 pbar.set_description(f'Epoch {epoch + 1:<2} - Training, Loss: {total_loss / (pbar.n + 1):.6f}')
    #
    #         train_loss = total_loss / len(self.train_loader)
    #         self.model.eval()
    #
    #         # Validation loop
    #         total_val_loss = 0
    #         with tqdm(self.val_loader, desc=f'Epoch {epoch + 1:<2} - Validation', unit_scale=True,
    #                   colour='blue', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
    #             for batch in pbar:
    #                 # inputs, seg_target, inpaint_target = [x.to(self.device) for x in batch.values()]
    #                 inputs, seg_gt, inpaint_gt = [batch[k].to(self.device) for k in ['image', 'mask', 'gt']]
    #                 output = self.model(inputs)
    #                 # args_loss = (output, seg_gt, inpaint_gt, self.seg_loss, self.inpaint_loss, self.lambda_)
    #                 # loss = composite_loss(*args_loss)
    #                 loss = self.loss(output, seg_gt, inpaint_gt)
    #                 total_val_loss += loss.item()
    #                 pbar.set_description(
    #                     f'Epoch {epoch + 1:<2} - Validation, Val Loss: {total_val_loss / (pbar.n + 1):.6f}')
    #
    #         val_loss = total_val_loss / len(self.val_loader)
    #
    #         self.logger.info(
    #             f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.6f}, Best Val Loss: {self.best_val_loss:.6f}, Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    #         )
    #
    #         # Save the images to during training to check the model improvements
    #         # save_images(epoch, inputs, seg_gt, inpaint_gt, output)
    #
    #         if val_loss < self.best_val_loss:
    #             self.best_val_loss = val_loss
    #             self.best_model_path = self.save_best_model()
    #             self.logger.info(f'Saving best model to {self.best_model_path}')


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
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optim')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--num_augment_operations', type=int, default=3, help='Number of data augmentation operations')
    parser.add_argument('--augment_magnitude', type=int, default=5, help='Magnitude of data augmentation operations')
    parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model')
    parser.add_argument('--lambda_', type=float, default=0.99, help='Weight for the composite loss')
    parser.add_argument('--debug', action='store_false', help='Debug mode')
    parser.add_argument('--selected_classes', type=int, nargs='+', default=[0],
                        help='List of classes IDs for inpainting (default:: person)')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        train_dataset = RORDDataset(root_dir=args.data_dir, split='debug',
                                    image_size=[args.image_size, args.image_size])
        val_dataset = RORDDataset(root_dir=args.data_dir, split='debug', image_size=[args.image_size, args.image_size])
    else:
        train_dataset = RORDDataset(root_dir=args.data_dir, split='train',
                                    image_size=[args.image_size, args.image_size])
        val_dataset = RORDDataset(root_dir=args.data_dir, split='val', image_size=[args.image_size, args.image_size])

    loguru.logger.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model, criterion, and optimizer
    model = InpainTor(num_classes=80, selected_classes=args.selected_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Define the loss function
    criterion_segmentation = NLLLoss()  # Segmentation Losses: BCELoss() # IoULoss() #DiceLoss() # CrossEntropyLoss() #NLLLoss()
    criterion_inpainting = MSELoss()  # Inpainting Losses: MSELoss()
    composite_loss = losses.CompositeLoss(criterion_segmentation, criterion_inpainting, args.lambda_)

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
    plot_training_log(file_path=log_file, log_scale=True)
