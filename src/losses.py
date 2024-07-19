import torch
import torch.nn as nn


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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class SegmentationFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(SegmentationLoss, self).__init__()
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

# class SegmentationLoss(nn.Module):
#     def __init__(self, ce_weight=0.5, dice_weight=0.5):
#         super(SegmentationLoss, self).__init__()
#         self.ce_weight = ce_weight
#         self.dice_weight = dice_weight
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dice_loss = DiceLoss()
#
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             # print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
#             if torch.any(target < 0) or torch.any(target >= pred.shape[1]):
#                 raise ValueError("Values in target are out of expected range")
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
#         with torch.no_grad():
#             # print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
#             if torch.any(target < 0) or torch.any(target >= pred.shape[1]):
#                 raise ValueError("Values in target are out of expected range")
#
#         pred = F.softmax(pred, dim=1)
#         target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
#
#         intersection = (pred * target_one_hot).sum(dim=(2, 3))
#         union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
#
#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
#         return 1 - dice.mean()


# TODO: Add PerceptualLoss to the InpaintingLoss
class InpaintingLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super(InpaintingLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply mask to focus loss on inpainted region
        l1_loss = self.l1_loss(pred * mask, target * mask)
        return self.l1_weight * l1_loss

#
# class SegmentationLoss(nn.Module):
#     def __init__(self, loss_type='dice', **kwargs):
#         super(SegmentationLoss, self).__init__()
#         self.loss_type = loss_type
#         if loss_type == 'dice':
#             self.loss_fn = self.dice_loss
#         elif loss_type == 'iou':
#             self.loss_fn = self.iou_loss
#         elif loss_type == 'focal':
#             self.alpha = kwargs.get('alpha', 0.25)
#             self.gamma = kwargs.get('gamma', 2.0)
#             self.loss_fn = self.focal_loss
#         elif loss_type == 'tversky':
#             self.alpha = kwargs.get('alpha', 0.5)
#             self.beta = kwargs.get('beta', 0.5)
#             self.loss_fn = self.tversky_loss
#         elif loss_type == 'binary_cross_entropy':
#             self.loss_fn = self.binary_cross_entropy_loss
#         else:
#             raise ValueError("Invalid loss type.")
#
#     def forward(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
#         return self.loss_fn(seg_output, seg_target)
#
#     @staticmethod
#     def dice_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
#         seg_output = torch.sigmoid(seg_output)
#         smooth = 1.0
#         intersection = (seg_output * seg_target).sum(dim=(1, 2, 3))
#         union = seg_output.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3))
#         dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
#         return dice_loss.mean()
#
#     @staticmethod
#     def iou_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
#         seg_output = torch.sigmoid(seg_output)
#         intersection = (seg_output * seg_target).sum(dim=(1, 2, 3))
#         union = seg_output.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3)) - intersection
#         iou_loss = 1 - (intersection + 1) / (union + 1)
#         return iou_loss.mean()
#
#     def focal_loss(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
#         seg_output = torch.sigmoid(seg_output)
#         pt = seg_output * seg_target + (1 - seg_output) * (1 - seg_target)
#         focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)
#         return focal_loss.mean()
#
#     def tversky_loss(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
#         seg_output = torch.sigmoid(seg_output)
#         intersection = (seg_output * seg_target).sum(dim=(1, 2, 3))
#         fps = (seg_output * (1 - seg_target)).sum(dim=(1, 2, 3))
#         fns = ((1 - seg_output) * seg_target).sum(dim=(1, 2, 3))
#         tversky_loss = 1 - (intersection + 1) / (intersection + self.alpha * fps + self.beta * fns + 1)
#         return tversky_loss.mean()
#
#     @staticmethod
#     def binary_cross_entropy_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
#         return F.binary_cross_entropy_with_logits(seg_output, seg_target)

#
# class CompoundSegmentationLoss(nn.Module):
#     """
#     Compound Segmentation Loss: Combination of Dice Loss, Focal Loss, and Boundary Loss.
#
#     Usage
#         criterion = CompoundSegmentationLoss(alpha=0.5, beta=0.25, gamma=2.0)
#         loss = criterion(pred, target)
#     """
#
#     def __init__(self, alpha=0.5, beta=0.25, gamma=2.0):
#         super(CompoundSegmentationLoss, self).__init__()
#         self.alpha = alpha  # weight for Dice Loss
#         self.beta = beta  # weight for Focal Loss
#         self.gamma = gamma  # focal parameter
#
#     def forward(self, pred, target):
#         # Ensure pred and target have the same shape
#         assert pred.size() == target.size(), f"Prediction shape {pred.size()} doesn't match target shape {target.size()}"
#         pred = torch.sigmoid(pred)
#         dice_loss = self.dice_loss(pred, target)
#         focal_loss = self.focal_loss(pred, target)
#         boundary_loss = self.boundary_loss(pred, target)
#         total_loss = self.alpha * dice_loss + self.beta * focal_loss + (1 - self.alpha - self.beta) * boundary_loss
#         return total_loss
#
#     @staticmethod
#     def dice_loss(pred, target):
#         smooth = 1.0
#         intersection = (pred * target).sum(dim=(2, 3))
#         union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
#         dice = (2. * intersection + smooth) / (union + smooth)
#         return 1 - dice.mean()
#
#     def focal_loss(self, pred, target):
#         bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = (1 - pt) ** self.gamma * bce_loss
#         return focal_loss.mean()
#
#     @staticmethod
#     def boundary_loss(pred, target):
#         # Compute gradients
#         pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
#         pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
#         target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
#         target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
#
#         # Compute boundary loss
#         dx_loss = F.mse_loss(pred_dx, target_dx)
#         dy_loss = F.mse_loss(pred_dy, target_dy)
#         return (dx_loss + dy_loss) / 2.0
#
#
# class CompositeLoss:
#     """
#     Composite Loss: Combines segmentation and inpainting losses.
#
#     Usage:
#         criterion = CompositeLoss(seg_loss, inpaint_loss, lambda_)
#         loss = criterion(output, seg_gt, inpaint_gt)
#     """
#
#     def __init__(self, seg_loss: callable, inpaint_loss: callable, lambda_: float):
#         self.seg_loss = seg_loss
#         self.inpaint_loss = inpaint_loss
#         self.lambda_ = lambda_
#
#     def __call__(self, output: dict, seg_gt: torch.Tensor, inpaint_gt: torch.Tensor) -> torch.Tensor:
#         seg_output = output['mask']
#         inpaint_output = output['inpainted_image']
#         seg_loss_value = self.seg_loss(seg_output, seg_gt)
#         inpaint_loss_value = self.inpaint_loss(inpaint_output, inpaint_gt)
#         return self.lambda_ * seg_loss_value + (1 - self.lambda_) * inpaint_loss_value
