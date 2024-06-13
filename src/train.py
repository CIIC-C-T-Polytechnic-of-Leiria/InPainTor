import argparse
import datetime
import importlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import inpaintor_model
import logger

# Reload the modules in case they have been modified
importlib.reload(dataset)
importlib.reload(inpaintor_model)
importlib.reload(logger)

from dataset import CamVidDataset
from inpaintor_model import InpainTor
from logger import Logger
from data_augmentation import RandAugment


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, device, train_loader,
                 val_loader, model_name, augment: bool = False, num_augment_operations: int = 3,
                 augment_magnitude: int = 5):
        self.best_model_path = None
        self.best_val_loss = float('inf')
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.logger = Logger(model_name)
        self.augment = augment
        self.num_augment_operations = num_augment_operations
        self.augment_magnitude = augment_magnitude
        self.rand_augment = RandAugment(num_operations=num_augment_operations,
                                        magnitude=augment_magnitude) if augment else None

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            with tqdm(self.train_loader, desc=f'Epoch {epoch + 1}',
                      bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for batch in pbar:
                    inputs, labels = batch['image'].to(self.device), batch['mask'].to(self.device)
                    if self.augment:
                        inputs, labels = self.rand_augment(inputs, labels)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.long())
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    pbar.set_postfix({'Loss': f'{total_loss / (pbar.n + 1):.3f}'}, refresh=False)

            # Calculate training loss
            train_loss = total_loss / len(self.train_loader)

            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs, labels = batch['image'].to(self.device), batch['mask'].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.long())
                    total_val_loss += loss.item()
            val_loss = total_val_loss / len(self.val_loader)

            print(
                f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            self.logger.log_metrics(epoch, train_loss, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_best_model(epoch)
                print(f'Saving best model to {self.best_model_path}')

    def save_best_model(self, epoch: int):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = os.path.join("checkpoints", f'best_model_{self.model_name}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Segmentor')
    parser.add_argument('--data_dir', type=str, default='data/CamVid', help='Path to dataset dir.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--num_augment_operations', type=int, default=3, help='Number of data augmentation operations')
    parser.add_argument('--augment_magnitude', type=int, default=5, help='Magnitude of data augmentation operations')
    parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model')

    args = parser.parse_args()
    # print("Parsed arguments:", args.__dict__)

    # Image and mask sizes
    image_size = (args.image_size, args.image_size)
    mask_size = (args.mask_size, args.mask_size)

    # Create data loaders
    train_dataset = CamVidDataset(root_dir=args.data_dir, split='train', image_size=image_size, mask_size=mask_size)
    val_dataset = CamVidDataset(root_dir=args.data_dir, split='val', image_size=image_size, mask_size=mask_size)
    print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model, criterion, and optimizer
    model = InpainTor(num_classes=40)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model in on device: {next(model.parameters()).device}, GPU: {torch.cuda.get_device_name(0)}")
    trainer = Trainer(model, criterion, optimizer, device, train_loader, val_loader, model_name='segmentor',
                      augment=args.augment, num_augment_operations=args.num_augment_operations,
                      augment_magnitude=args.augment_magnitude)
    print('Training the model...')
    trainer.train(args.num_epochs)
    print('Training completed.')
