"""
 train.py

 Usage example:
    python src/train.py --data_dir=/media/tiagociiic/easystore/RORD --num_epochs=200 --batch_size=2 --image_size=512
 """

import argparse
import importlib
import os
from datetime import datetime

import loguru
import torch
import torch.optim as optim
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

from dataset import RORDDataset
from visualizations import save_train_images

loguru.logger.remove()
log_file = f"logs/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
loguru.logger.add(log_file, rotation="10 MB")


# ------------CONTINAR AQUI----------------------------
class Trainer:
    def __init__(self, model_: Module,
                 seg_loss: callable,
                 inpaint_loss: callable,
                 optimizer_: optim.Optimizer,
                 device_: torch.device,
                 seg_train_loader: DataLoader,
                 seg_val_loader: DataLoader,
                 inpaint_train_loader: DataLoader,
                 inpaint_val_loader: DataLoader,
                 model_name: str,
                 log_interval: int,
                 scheduler_: ReduceLROnPlateau):
        self.logger = loguru.logger.bind(trainer=True)
        self.best_model_path = None
        self.best_val_loss = float('inf')
        self.model = model_
        self.seg_loss = seg_loss
        self.inpaint_loss = inpaint_loss
        self.optim = optimizer_
        self.device = device_
        self.seg_train_loader = seg_train_loader
        self.seg_val_loader = seg_val_loader
        self.inpaint_train_loader = inpaint_train_loader
        self.inpaint_val_loader = inpaint_val_loader
        self.model_name = model_name
        self.scheduler = scheduler_
        self.log_interval = log_interval
        self.global_step = 0

    def save_best_model(self, phase):
        date = datetime.now().strftime('%Y-%m-%d')
        model_path = os.path.join("checkpoints", f'best_{self.model_name}_{phase}_{date}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def train(self, num_epochs_seg: int, num_epochs_inpaint: int):
        # Phase 1: Train segmentation
        self.logger.info("Starting Phase 1: Segmentation Training")
        self.model.freeze_generator()
        self.train_phase(num_epochs_seg, "segmentation")
        self.model.save_segmenter("segmenter_checkpoint.pth")

        # Phase 2: Train inpainting
        self.logger.info("Starting Phase 2: Inpainting Training")
        self.model.load_segmenter("segmenter_checkpoint.pth")
        self.model.freeze_segmenter()
        self.model.unfreeze_generator()
        self.train_phase(num_epochs_inpaint, "inpainting")

        # Optional: Fine-tuning phase
        self.logger.info("Starting Fine-tuning Phase")
        self.model.unfreeze_all()
        self.train_phase(10, "fine_tuning")  # 10 epochs for fine-tuning, adjust as needed

    def train_phase(self, num_epochs: int, phase: str):
        for epoch in range(num_epochs):
            if phase == "segmentation":
                train_loss = self.train_epoch_seg(epoch)
                val_loss = self.validate_epoch_seg(epoch)
            else:  # inpainting or fine_tuning
                train_loss = self.train_epoch_inpaint(epoch)
                val_loss = self.validate_epoch_inpaint(epoch)

            self.scheduler.step(val_loss)

            self.logger.info(
                f'{phase.capitalize()} - Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.6f}, '
                f'Best Val Loss: {self.best_val_loss:.6f}, '
                f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_best_model(phase)
                self.logger.info(f'Saving best {phase} model_ to {self.best_model_path}')

    def train_epoch_seg(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        with tqdm(self.seg_train_loader, desc=f'Epoch {epoch + 1:<2} - Segmentation Training', unit_scale=True,
                  colour='green', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optim.zero_grad()
                output = self.model(images)
                loss = self.seg_loss(output['mask'], masks)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

                self.global_step += 1
                if self.global_step % self.log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    self.logger.info(
                        f'Segmentation - Epoch {epoch + 1}, Step {self.global_step}, Train Loss: {avg_loss:.6f}, '
                        f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    )
                    save_train_images(self.global_step, images, masks, None, output, phase="segmentation")

                pbar.set_description(
                    f'Epoch {epoch + 1:<2} - Segmentation Training, Loss: {total_loss / (batch_idx + 1):.6f}')

        return total_loss / len(self.seg_train_loader)

    def validate_epoch_seg(self, epoch: int) -> float:
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(self.seg_val_loader, desc=f'Epoch {epoch + 1:<2} - Segmentation Validation', unit_scale=True,
                      colour='blue', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images, masks = images.to(self.device), masks.to(self.device)
                    output = self.model(images)
                    loss = self.seg_loss(output['mask'], masks)
                    total_val_loss += loss.item()

                    if self.global_step % self.log_interval == 0:
                        avg_val_loss = total_val_loss / (batch_idx + 1)
                        self.logger.info(
                            f'Segmentation - Epoch {epoch + 1}, Step {self.global_step}, Val Loss: {avg_val_loss:.6f}, '
                            f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                        )
                        save_train_images(self.global_step, images, masks, None, output, is_validation=True,
                                          phase="segmentation")

                    pbar.set_description(
                        f'Epoch {epoch + 1:<2} - Segmentation Validation, Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

        return total_val_loss / len(self.seg_val_loader)

    def train_epoch_inpaint(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        with tqdm(self.inpaint_train_loader, desc=f'Epoch {epoch + 1:<2} - Inpainting Training', unit_scale=True,
                  colour='green', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
            for batch_idx, batch in enumerate(pbar):
                inputs, inpaint_gt = batch['image'].to(self.device), batch['gt'].to(self.device)
                self.optim.zero_grad()
                output = self.model(inputs)
                loss = self.inpaint_loss(output['inpainted_image'], inpaint_gt)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

                self.global_step += 1
                if self.global_step % self.log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    self.logger.info(
                        f'Inpainting - Epoch {epoch + 1}, Step {self.global_step}, Train Loss: {avg_loss:.6f}, '
                        f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    )
                    save_train_images(self.global_step, inputs, None, inpaint_gt, output, phase="inpainting")

                pbar.set_description(
                    f'Epoch {epoch + 1:<2} - Inpainting Training, Loss: {total_loss / (batch_idx + 1):.6f}')

        return total_loss / len(self.inpaint_train_loader)

    def validate_epoch_inpaint(self, epoch: int) -> float:
        self.model.eval()
        total_val_ = 0
        with torch.no_grad():
            with tqdm(self.inpaint_val_loader, desc=f'Epoch {epoch + 1:<2} - Inpainting Validation', unit_scale=True,
                      colour='blue', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    inputs, inpaint_gt = batch['image'].to(self.device), batch['gt'].to(self.device)
                    output = self.model(inputs)
                    loss = self.inpaint_loss(output['inpainted_image'], inpaint_gt)
                    total_val_loss += loss.item()

                    if self.global_step % self.log_interval == 0:
                        avg_val_loss = total_val_loss / (batch_idx + 1)
                        self.logger.info(
                            f'Inpainting - Epoch {epoch + 1}, Step {self.global_step}, Val Loss: {avg_val_loss:.6f}, '
                            f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                        )
                        save_train_images(self.global_step, inputs, None, inpaint_gt, output, is_validation=True,
                                          phase="inpainting")

                    pbar.set_description(
                        f'Epoch {epoch + 1:<2} - Inpainting Validation, Val Loss: {total_val_loss / (batch_idx + 1):.6f}')

        return total_val_loss / len(self.inpaint_val_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the InPainTor model_ in two phases.')
    parser.add_argument('--coco_dir', type=str, help='Path to COCO dataset dir')
    parser.add_argument('--rord_dir', type=str, help='Path to RORD dataset dir')
    parser.add_argument('--num_epochs_seg', type=int, default=50, help='Number of epochs to train segmentation')
    parser.add_argument('--num_epochs_inpaint', type=int, default=100, help='Number of epochs to train inpainting')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer_')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
    parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model_')
    parser.add_argument('--log_interval', type=int, default=1000, help='Log interval for training')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        coco_train_dataset = COCOSegmentationDataset(root_dir=args.coco_dir, split='debug',
                                                     image_size=[args.image_size, args.image_size])
        coco_val_dataset = COCOSegmentationDataset(root_dir=args.coco_dir, split='debug',
                                                   image_size=[args.image_size, args.image_size])
        rord_train_dataset = RORDDataset(root_dir=args.rord_dir, split='debug',
                                         image_size=[args.image_size, args.image_size])
        rord_val_dataset = RORDDataset(root_dir=args.rord_dir, split='debug',
                                       image_size=[args.image_size, args.image_size])
        print("Debug mode enabled. Using debug datasets.")
    else:
        coco_train_dataset = COCOSegmentationDataset(root_dir=args.coco_dir, split='train',
                                                     image_size=[args.image_size, args.image_size])
        coco_val_dataset = COCOSegmentationDataset(root_dir=args.coco_dir, split='val',
                                                   image_size=[args.image_size, args.image_size])
        rord_train_dataset = RORDDataset(root_dir=args.rord_dir, split='train',
                                         image_size=[args.image_size, args.image_size])
        rord_val_dataset = RORDDataset(root_dir=args.rord_dir, split='val',
                                       image_size=[args.image_size, args.image_size])

    loguru.logger.info(
        f'COCO Training samples: {len(coco_train_dataset)}, COCO Validation samples: {len(coco_val_dataset)}')
    loguru.logger.info(
        f'RORD Training samples: {len(rord_train_dataset)}, RORD Validation samples: {len(rord_val_dataset)}')

# ------------'

# class Trainer:
#     def __init__(self, model_: Module,
#                  loss: callable,
#                  optimizer_: optim.Optimizer,
#                  device_: torch.device_,
#                  training_loader: DataLoader,
#                  valid_loader: DataLoader,
#                  model_name: str,
#                  log_interval: int,
#                  scheduler_: ReduceLROnPlateau,
#                  augment: bool = False,
#                  num_augment_operations: int = 3,
#                  augment_magnitude: int = 5,
#                  lambda_: float = 0.5,
#                  criterion_segment: callable = None,
#                  criterion_inpaint: callable = None):
#         self.lambda_ = lambda_
#         # self.logger = loguru.logger
#         self.logger = loguru.logger.bind(trainer=True)
#         self.best_model_path = None
#         self.best_val_loss = float('inf')
#         self.model_ = model_
#         self.loss = loss
#         self.seg_loss = criterion_segment
#         self.inpaint_loss = criterion_inpaint
#         self.optim = optimizer_
#         self.device_ = device_
#         self.train_loader_ = training_loader
#         self.val_loader_ = valid_loader
#         self.model_name = model_name
#         # self.log_train = ModelPerformanceTracker(model_name)
#         self.augment = augment
#         self.num_augment_operations = num_augment_operations
#         self.scheduler_ = scheduler_
#         self.augment_magnitude = augment_magnitude
#         self.log_interval = log_interval
#         self.global_step = 0
#         self.rand_augment = RandAugment(num_operations=num_augment_operations,
#                                         magnitude=augment_magnitude) if augment else None
#
#     def save_best_model(self):
#         date = datetime.now().strftime('%Y-%m-%d')
#         model_path = os.path.join("checkpoints", f'best_{self.model_name}_{date}.pth')
#         torch.save(self.model_.state_dict(), model_path)
#         return model_path
#
#     def train(self, num_epochs: int):
#         for epoch in range(num_epochs):
#             train_loss = self.train_epoch(epoch)
#             val_loss = self.validate_epoch(epoch)
#
#             self.scheduler_.step(val_loss)
#
#             self.logger.info(
#                 f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.6f}, '
#                 f'Best Val Loss: {self.best_val_loss:.6f}, '
#                 f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
#             )
#
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 self.best_model_path = self.save_best_model()
#                 self.logger.info(f'Saving best model_ to {self.best_model_path}')
#
#     def train_epoch(self, epoch: int) -> float:
#         self.model_.train()
#         total_loss = 0
#         with tqdm(self.train_loader_, desc=f'Epoch {epoch + 1:<2} - Training', unit_scale=True,
#                   colour='green', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
#             for batch_idx, batch in enumerate(pbar):
#                 inputs, seg_gt, inpaint_gt = [batch[k].to(self.device_) for k in ['image', 'mask', 'gt']]
#                 self.optim.zero_grad()
#                 output = self.model_(inputs)
#                 loss = self.loss(output, seg_gt, inpaint_gt)
#                 loss.backward()
#                 self.optim.step()
#                 total_loss += loss.item()
#
#                 self.global_step += 1
#                 if self.global_step % self.log_interval == 0:
#                     avg_loss = total_loss / (batch_idx + 1)
#                     self.logger.info(
#                         f'Epoch {epoch + 1}, Step {self.global_step}, Train Loss: {avg_loss:.6f}, '
#                         f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
#                     )
#                     save_train_images(self.global_step, inputs, seg_gt, inpaint_gt, output)
#
#                 pbar.set_description(f'Epoch {epoch + 1:<2} - Training, Loss: {total_loss / (batch_idx + 1):.6f}')
#
#         return total_loss / len(self.train_loader_)
#
#     def validate_epoch(self, epoch: int) -> float:
#         self.model_.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             with tqdm(self.val_loader_, desc=f'Epoch {epoch + 1:<2} - Validation', unit_scale=True,
#                       colour='blue', bar_format='{l_bar}{bar:10}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]') as pbar:
#                 for batch_idx, batch in enumerate(pbar):
#                     inputs, seg_gt, inpaint_gt = [batch[k].to(self.device_) for k in ['image', 'mask', 'gt']]
#                     output = self.model_(inputs)
#                     loss = self.loss(output, seg_gt, inpaint_gt)
#                     total_val_loss += loss.item()
#
#                     if self.global_step % self.log_interval == 0:
#                         avg_val_loss = total_val_loss / (batch_idx + 1)
#                         self.logger.info(
#                             f'Epoch {epoch + 1}, Step {self.global_step}, Val Loss: {avg_val_loss:.6f}, '
#                             f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
#                         )
#                         save_train_images(self.global_step, inputs, seg_gt, inpaint_gt, output, is_validation=True)
#
#                     pbar.set_description(
#                         f'Epoch {epoch + 1:<2} - Validation, Val Loss: {total_val_loss / (batch_idx + 1):.6f}')
#
#         return total_val_loss / len(self.val_loader_)
#
#
# def parse_args() -> argparse.Namespace:
#     """
#     Parse the command line arguments.
#
#     Returns:
#         argparse.Namespace: The parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description='Train the InPainTor model_.')
#     parser.add_argument('--data_dir', type=str, help='Path to dataset dir.')
#     parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
#     parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
#     parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optim')
#     parser.add_argument('--image_size', type=int, default=512, help='Size of the input images, assumed to be square')
#     parser.add_argument('--mask_size', type=int, default=256, help='Size of the masks, assumed to be square')
#     parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
#     parser.add_argument('--num_augment_operations', type=int, default=3, help='Number of data augmentation operations')
#     parser.add_argument('--augment_magnitude', type=int, default=5, help='Magnitude of data augmentation operations')
#     parser.add_argument('--model_name', type=str, default='InpainTor', help='Name of the model_')
#     parser.add_argument('--lambda_', type=float, default=0.99, help='Weight for the composite loss')
#     parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
#     parser.add_argument('--log_interval', type=int, default=1000, help='Log interval for training')
#     parser.add_argument('--find_lr', action='store_true', help='Run learning rate finder')
#     parser.add_argument('--selected_classes', type=int, nargs='+', default=[0],
#                         help='List of classes IDs for inpainting (default:: person)')
#
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = parse_args()
#
#     if args.debug:
#         train_dataset = RORDDataset(root_dir=args.data_dir, split='debug',
#                                     image_size=[args.image_size, args.image_size])
#         val_dataset = RORDDataset(root_dir=args.data_dir, split='debug', image_size=[args.image_size, args.image_size])
#         print("Debug mode enabled. Using debug dataset.")
#     else:
#         train_dataset = RORDDataset(root_dir=args.data_dir, split='train',
#                                     image_size=[args.image_size, args.image_size])
#         val_dataset = RORDDataset(root_dir=args.data_dir, split='val', image_size=[args.image_size, args.image_size])
#
#     loguru.logger.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
#     train_loader_ = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     val_loader_ = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
#
#     model_ = InpainTor(num_classes=80, selected_classes=args.selected_classes)
#     device_ = torch.device_('cuda' if torch.cuda.is_available() else 'cpu')
#     model_ = model_.to(device_)
#     optimizer_ = optim.AdamW(model_.parameters(), lr=args.learning_rate)
#     scheduler_ = ReduceLROnPlateau(optimizer_, mode='min', factor=0.25, patience=10)
#
#     criterion_segmentation = NLLLoss()
#     criterion_inpainting = MSELoss()
#     composite_loss = losses.CompositeLoss(criterion_segmentation, criterion_inpainting, args.lambda_)
#
#     if args.find_lr:
#         raise NotImplementedError("Learning rate finder not implemented yet.")
#     else:
#         trainer = Trainer(model_=model_,
#                           loss=composite_loss,
#                           optimizer_=optimizer_,
#                           device_=torch.device_('cuda' if torch.cuda.is_available() else 'cpu'),
#                           log_interval=args.log_interval,
#                           training_loader=train_loader_,
#                           valid_loader=val_loader_,
#                           model_name=args.model_name,
#                           augment=args.augment,
#                           num_augment_operations=args.num_augment_operations,
#                           augment_magnitude=args.augment_magnitude,
#                           lambda_=args.lambda_,
#                           scheduler_=scheduler_,
#                           criterion_segment=criterion_segmentation,
#                           criterion_inpaint=criterion_inpainting)
#
#         trainer.train(args.num_epochs)
#         plot_training_log(file_path=log_file)
