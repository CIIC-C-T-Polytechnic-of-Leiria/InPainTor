import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, loss_type='dice', **kwargs):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'dice':
            self.loss_fn = self.dice_loss
        elif loss_type == 'iou':
            self.loss_fn = self.iou_loss
        elif loss_type == 'focal':
            self.alpha = kwargs.get('alpha', 0.25)
            self.gamma = kwargs.get('gamma', 2.0)
            self.loss_fn = self.focal_loss
        elif loss_type == 'tversky':
            self.alpha = kwargs.get('alpha', 0.5)
            self.beta = kwargs.get('beta', 0.5)
            self.loss_fn = self.tversky_loss
        elif loss_type == 'binary_cross_entropy':
            self.loss_fn = self.binary_cross_entropy_loss
        else:
            raise ValueError("Invalid loss type.")

    def forward(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(seg_output, seg_target)

    @staticmethod
    def dice_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        seg_output = torch.sigmoid(seg_output)
        smooth = 1.0
        intersection = (seg_output * seg_target).sum(dim=(1, 2, 3))
        union = seg_output.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
        return dice_loss.mean()

    @staticmethod
    def iou_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        seg_output = torch.sigmoid(seg_output)
        intersection = (seg_output * seg_target).sum(dim=(1, 2, 3))
        union = seg_output.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3)) - intersection
        iou_loss = 1 - (intersection + 1) / (union + 1)
        return iou_loss.mean()

    def focal_loss(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        seg_output = torch.sigmoid(seg_output)
        pt = seg_output * seg_target + (1 - seg_output) * (1 - seg_target)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)
        return focal_loss.mean()

    def tversky_loss(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        seg_output = torch.sigmoid(seg_output)
        intersection = (seg_output * seg_target).sum(dim=(1, 2, 3))
        fps = (seg_output * (1 - seg_target)).sum(dim=(1, 2, 3))
        fns = ((1 - seg_output) * seg_target).sum(dim=(1, 2, 3))
        tversky_loss = 1 - (intersection + 1) / (intersection + self.alpha * fps + self.beta * fns + 1)
        return tversky_loss.mean()

    @staticmethod
    def binary_cross_entropy_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(seg_output, seg_target)


class CompoundSegmentationLoss(nn.Module):
    """
    Compound Segmentation Loss: Combination of Dice Loss, Focal Loss, and Boundary Loss.

    Usage
        criterion = CompoundSegmentationLoss(alpha=0.5, beta=0.25, gamma=2.0)
        loss = criterion(pred, target)
    """

    def __init__(self, alpha=0.5, beta=0.25, gamma=2.0):
        super(CompoundSegmentationLoss, self).__init__()
        self.alpha = alpha  # weight for Dice Loss
        self.beta = beta  # weight for Focal Loss
        self.gamma = gamma  # focal parameter

    def forward(self, pred, target):
        # Ensure pred and target have the same shape
        assert pred.size() == target.size(), f"Prediction shape {pred.size()} doesn't match target shape {target.size()}"
        pred = torch.sigmoid(pred)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        boundary_loss = self.boundary_loss(pred, target)
        total_loss = self.alpha * dice_loss + self.beta * focal_loss + (1 - self.alpha - self.beta) * boundary_loss
        return total_loss

    @staticmethod
    def dice_loss(pred, target):
        smooth = 1.0
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

    @staticmethod
    def boundary_loss(pred, target):
        # Compute gradients
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Compute boundary loss
        dx_loss = F.mse_loss(pred_dx, target_dx)
        dy_loss = F.mse_loss(pred_dy, target_dy)
        return (dx_loss + dy_loss) / 2.0


class CompositeLoss:
    """
    Composite Loss: Combines segmentation and inpainting losses.

    Usage:
        criterion = CompositeLoss(seg_loss, inpaint_loss, lambda_)
        loss = criterion(output, seg_gt, inpaint_gt)
    """

    def __init__(self, seg_loss: callable, inpaint_loss: callable, lambda_: float):
        self.seg_loss = seg_loss
        self.inpaint_loss = inpaint_loss
        self.lambda_ = lambda_

    def __call__(self, output: dict, seg_gt: torch.Tensor, inpaint_gt: torch.Tensor) -> torch.Tensor:
        seg_output = output['mask']
        inpaint_output = output['inpainted_image']
        seg_loss_value = self.seg_loss(seg_output, seg_gt)
        inpaint_loss_value = self.inpaint_loss(inpaint_output, inpaint_gt)
        return self.lambda_ * seg_loss_value + (1 - self.lambda_) * inpaint_loss_value

# def composite_loss(
#         outputs: dict,
#         seg_target: torch.Tensor,
#         inpaint_target: torch.Tensor,
#         segmentation_loss: callable,
#         inpaint_loss: callable,
#         lambda_: float
# ) -> torch.Tensor:
#     """
#     Computes the composite loss for segmentation and inpainting tasks.
#
#     Args:
#         outputs (dict): Dictionary containing the model's output tensors.
#         seg_target (torch.Tensor): Segmentation target tensor.
#         inpaint_target (torch.Tensor): Inpainting target tensor.
#         segmentation_loss (callable): Segmentation loss function.
#         inpaint_loss (callable): Inpainting loss function.
#         lambda_ (float): Weight for the composite loss.
#
#     Returns:
#         torch.Tensor: Composite loss tensor.
#     """
#     try:
#         seg_output, inpaint_output = outputs['mask'], outputs['inpainted_image']
#         # TODO: Depois corrigir a linha em baixo para funcionar com batch_size > 1
#         seg_output = seg_output
#         # seg_target_class = torch.argmax(seg_target, dim=0)
#         seg_target_class = seg_target
#
#         print(f"seg_output.shape: {seg_output.shape}, seg_target_class.shape: {seg_target_class.shape}")
#         print(
#             f"seg_target_class max: {seg_target_class.max()}, seg_target_class min: {seg_target_class.min()}, mean: {seg_target_class.mean()}")
#
#         print(f"seg_output max: {seg_output.max()}, seg_output min: {seg_output.min()}, mean: {seg_output.mean()}")
#
#         segmentation_loss_val = segmentation_loss(seg_output.float(), seg_target_class.float())
#         print(f"segmentation_loss_val: {segmentation_loss_val}")
#         inpainting_loss_val = inpaint_loss(inpaint_output, inpaint_target)
#         loss = lambda_ * segmentation_loss_val + (1 - lambda_) * inpainting_loss_val
#         return loss
#     except KeyError as e:
#         loguru.logger.error("Outputs dictionary must contain 'ask' and 'inpainted_image' keys.")
#         raise ValueError("Outputs dictionary must contain 'ask' and 'inpainted_image' keys.") from e
#     except TypeError as e:
#         loguru.logger.error(
#             "Invalid input types. Check the types of outputs, seg_target, inpaint_target, segmentation_loss, "
#             "and inpaint_loss."
#         )
#         raise ValueError(
#             "Invalid input types. Check the types of outputs, seg_target, inpaint_target, segmentation_loss, "
#             "and inpaint_loss."
#         ) from e
#     except RuntimeError as e:
#         loguru.logger.error("Error computing the composite loss. Check the input tensors and loss functions.")
#         raise ValueError("Error computing the composite loss. Check the input tensors and loss functions.") from e
