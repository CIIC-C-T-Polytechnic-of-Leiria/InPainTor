"""
Module defining custom loss functions for deep learning tasks.

This module includes implementations for various loss functions commonly used in segmentation and inpainting tasks:

- `DiceLoss`: Computes the Dice loss, a measure of overlap between predicted and target binary masks.
- `SegmentationLoss`: Combines Binary Cross Entropy (BCE) loss and Dice loss for segmentation tasks.
- `FocalLoss`: Applies Focal Loss to address class imbalance by focusing on hard-to-classify examples.
- `SegmentationFocalLoss`: Combines Focal Loss with optional normalization of target values for segmentation tasks.
- `InpaintingLoss`: Uses L1 loss for image inpainting tasks, with a placeholder for potential Perceptual Loss integration.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice_score = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)

        return 1 - dice_score.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight=1, dice_weight=0.0):
        super(SegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Garantir que target esteja no formato float e normalizado para [0, 1]
        if target.dtype != torch.float32:
            target = target.float()
        if target.max() > 1:
            target = target / 255.0

        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_factor * focal_weight * bce_loss

        return focal_loss.mean()


class SegmentationFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(SegmentationFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dtype != torch.float32:
            target = target.float()
        if target.max() > 1:
            target = target / 255.0

        return self.focal_loss(pred, target)


# ----------------------------
# Softmax Implementation (for reference)
# class SegmentationLoss(nn.Module):
#     def __init__(self, ce_weight=1, dice_weight=0.0):
#         super(SegmentationLoss, self).__init__()
#         self.ce_weight = ce_weight
#         self.dice_weight = dice_weight
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dice_loss = DiceLoss()
#
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Verificar se target é um tensor de índices
#         if target.dim() == 4:
#             target = target.argmax(dim=1)
#         elif target.dim() == 3:
#             target = target.long()
#         if torch.any(target < 0) or torch.any(target >= pred.shape[1]):
#             raise ValueError("Values in target are out of expected range")
#
#         ce_loss = self.ce_loss(pred, target)
#         dice_loss = self.dice_loss(pred, target)
#         return self.ce_weight * ce_loss + self.dice_weight * dice_loss
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Verificar se target é um tensor de índices
#         if target.dim() == 4:
#             target = torch.argmax(target, dim=1)
#         if torch.any(target < 0) or torch.any(target >= pred.shape[1]):
#             raise ValueError("Values in target are out of expected range")
#
#         pred = F.softmax(pred, dim=1)
#         target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
#
#         intersection = (pred * target_one_hot).sum(dim=(2, 3))
#         union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
#
#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
#         return 1 - dice.mean()
#

# ----------------------------

class InpaintingLoss(nn.Module):
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1):
        super(InpaintingLoss, self).__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.mse_loss = nn.MSELoss()
        # TODO: test with PerceptualLoss
        # self.perceptual_loss = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse_loss(pred, target)
        # TODO: test with PerceptualLoss
        # perceptual_loss = self.perceptual_loss(pred, target)
        total_loss = self.mse_weight * mse_loss
        # total_loss += self.perceptual_weight * perceptual_loss

        return total_loss


# TODO: Add PerceptualLoss to the InpaintingLoss
class InpaintingLoss_v2(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super(InpaintingLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        # TODO: test with PerceptualLoss
        # self.perceptual_loss = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1_loss(pred, target)
        # TODO: test with PerceptualLoss
        # perceptual_loss = self.perceptual_loss(pred, target)
        total_loss = self.l1_weight * l1_loss
        # total_loss += self.perceptual_weight * perceptual_loss

        return total_loss
