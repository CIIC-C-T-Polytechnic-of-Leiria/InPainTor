import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """
    A custom loss function for segmentation tasks.

    This class supports five types of loss functions: 'dice', 'iou', 'focal', 'tversky', and 'binary_cross_entropy'.
    The type of loss function to use is specified during initialization.

    Args:
        loss_type (str, optional): The type of loss function to use.
            Must be one of 'dice', 'iou', 'focal', 'tversky', or 'binary_cross_entropy'. Defaults to 'dice'.
        **kwargs: Additional parameters for the 'focal' and 'tversky' loss functions.
            For 'focal', 'alpha' and 'gamma' can be specified.
            For 'tversky', 'alpha' and 'beta' can be specified.

    Raises:
        ValueError: If an invalid loss type is provided.

    Example:
        >>> criterion = SegmentationLoss(loss_type='dice')
        >>> loss = criterion(seg_output, seg_target)
    """

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
            raise ValueError(
                "Invalid loss type. Must be one of 'dice', 'iou', 'focal', 'tversky', or 'binary_cross_entropy'.")

    def forward(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(seg_output, seg_target)

    @staticmethod
    def dice_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        assert seg_output.dim() == 4 and seg_target.dim() == 4, "Unexpected tensor dimensions"
        smooth = 1.0
        if seg_output.shape[1] == 1:
            seg_output = torch.sigmoid(seg_output)
            intersection = (seg_output * seg_target).sum(dim=(2, 3))  # sum over H and W dimensions
            union = seg_output.sum(dim=(2, 3)) + seg_target.sum(dim=(2, 3))  # sum over H and W dimensions
            dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
            return dice_loss.mean()
        else:
            seg_output = torch.softmax(seg_output, dim=1)
            intersection = (seg_output * seg_target).sum(dim=(2, 3))  # sum over H and W dimensions
            union = seg_output.sum(dim=(2, 3)) + seg_target.sum(dim=(2, 3))  # sum over H and W dimensions
            dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
            return dice_loss.mean(dim=1)  # average over classes

    @staticmethod
    def iou_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        assert seg_output.dim() == 4 and seg_target.dim() == 4, "Unexpected tensor dimensions"
        if seg_output.shape[1] == 1:
            seg_output = torch.sigmoid(seg_output)
            intersection = (seg_output * seg_target).sum()
            union = seg_output.sum() + seg_target.sum() - intersection
            iou_loss = 1 - (intersection + 1) / (union + 1)
            return iou_loss
        else:
            seg_output = torch.softmax(seg_output, dim=1)
            intersection = (seg_output * seg_target).sum(dim=(2, 3))  # sum over H and W dimensions
            union = seg_output.sum(dim=(2, 3)) + seg_target.sum(dim=(2, 3))  # sum over H and W dimensions
            iou_loss = 1 - (intersection + 1) / (union + 1)
            return iou_loss.mean(dim=1)  # average over classes

    def focal_loss(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-6  # Add epsilon to prevent log(0)
        if seg_output.shape[1] == 1:
            seg_output = torch.sigmoid(seg_output)
            pt = seg_output * seg_target + (1 - seg_output) * (1 - seg_target)
            focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + epsilon)
            return focal_loss.mean()
        else:
            seg_output = torch.softmax(seg_output, dim=1)
            pt = seg_output * seg_target + (1 - seg_output) * (1 - seg_target)
            focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + epsilon)
            return focal_loss.mean(dim=1)  # average over classes

    def tversky_loss(self, seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        if seg_output.shape[1] == 1:
            seg_output = torch.sigmoid(seg_output)
            intersection = (seg_output * seg_target).sum()
            fps = (seg_output * (1 - seg_target)).sum()
            fns = ((1 - seg_output) * seg_target).sum()
            tversky_loss = 1 - (intersection + 1) / (intersection + self.alpha * fps + self.beta * fns + 1)
            return tversky_loss
        else:
            seg_output = torch.softmax(seg_output, dim=1)
            intersection = (seg_output * seg_target).sum(dim=(2, 3))  # sum over H and W dimensions
            fps = (seg_output * (1 - seg_target)).sum(dim=(2, 3))  # sum over H and W dimensions
            fns = ((1 - seg_output) * seg_target).sum(dim=(2, 3))  # sum over H and W dimensions
            tversky_loss = 1 - (intersection + 1) / (intersection + self.alpha * fps + self.beta * fns + 1)
            return tversky_loss.mean(dim=1)  # average over classes

    @staticmethod
    def binary_cross_entropy_loss(seg_output: torch.Tensor, seg_target: torch.Tensor) -> torch.Tensor:
        assert seg_output.dim() == 4 and seg_target.dim() == 4, "Unexpected tensor dimensions"
        if seg_output.shape[1] == 1:
            seg_output = torch.sigmoid(seg_output)
            bce_loss = F.binary_cross_entropy(seg_output, seg_target)
            return bce_loss
        else:
            seg_output = torch.softmax(seg_output, dim=1)
            bce_loss = F.binary_cross_entropy(seg_output, seg_target)
            return bce_loss.mean(dim=1)
