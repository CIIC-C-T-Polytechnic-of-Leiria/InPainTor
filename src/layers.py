"""
Module containing custom PyTorch layers for deep learning architectures.

This module includes implementations for various neural network layers and blocks:

- `calculate_same_padding`: Computes padding to maintain the input size after a convolution operation.
- `Sigmoid`: Applies the sigmoid activation function.
- `LogSoftmax`: Applies the log softmax activation function along a specified dimension.
- `Softmax`: Applies the softmax activation function along a specified dimension.
- `Conv1x1`: Performs 1x1 convolutions followed by Batch Normalization and an optional activation function.
- `SepConvBlock`: Implements a separable convolution block with optional skip connections and pooling.
- `SepConvTranspBlock`: Implements a separable transposed convolution block.
- `AttentionBlock`: Applies an attention mechanism to input features, enhancing feature representation.
- `ClassesToMask_v2`: Converts class logits to binary masks based on specified class IDs and an optional threshold.

Each class and function in this module is designed to facilitate the construction and experimentation
with neural network models, providing functionalities such as activation functions,
convolutions, separable convolutions, attention mechanisms, and class-to-mask transformations.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_same_padding(input_size: int, kernel_size: int, stride: int):
    """Calculate the padding needed to keep the input size the same after a convolution."""
    output_size = (input_size + stride - 1) // stride
    total_padding = max(0, (output_size - 1) * stride + kernel_size - input_size)
    padding = total_padding // 2

    return padding


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    @staticmethod
    def forward(x):
        return F.sigmoid(x)


class LogSoftmax(nn.Module):
    def __init__(self, dim=1):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.log_softmax(x, dim=self.dim)


class Softmax(nn.Module):
    def __init__(self, dim=1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False, activation=None):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

        # He initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: int,
                 stride: int = 1,
                 pool_kernel_size: int = 2,
                 kernel_size: int = 3,
                 return_skip: bool = False
                 ):

        super(SepConvBlock, self).__init__()
        self.return_skip = return_skip

        padding = calculate_same_padding(input_size, kernel_size, stride)

        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)  # TODO: try with LayerNorm
        self.pool = nn.MaxPool2d(pool_kernel_size, stride=2)

        # He Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep_conv(x)
        x = self.activation(x)
        x = self.bn(x)
        x_pool = self.pool(x)

        if self.return_skip:
            return x_pool, x
        else:
            return x_pool


class SepConvTranspBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2):
        super(SepConvTranspBlock, self).__init__()

        self.sep_transp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1,
                               output_padding=1,
                               groups=in_channels,
                               bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        )
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)  # TODO: try with LayerNorm

        # He Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"SepConvTransposeBlock, Before Conv: {x.shape}")
        x = self.sep_transp_conv(x)
        # print(f"SepConvTransposeBlock, After Conv: {x.shape}")
        x = self.activation(x)
        # print(f"SepConvTransposeBlock, Before BN. {x.shape}")
        x = self.bn(x)
        # print(f"SepConvTransposeBlock, After BN.:{x.shape}")

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, attention_dim: int):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels=attention_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels=attention_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1)
        self.bn_query = nn.BatchNorm2d(attention_dim)
        self.bn_key = nn.BatchNorm2d(attention_dim)
        self.bn_value = nn.BatchNorm2d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

        # He initialization
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, segmentation_features) -> torch.Tensor:
        query = self.bn_query(self.query_conv(segmentation_features))
        key = self.bn_key(self.key_conv(segmentation_features))
        value = self.bn_value(self.value_conv(segmentation_features))
        # Dot product attention
        attention_weights = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(key.shape[-1]).float())
        # print(f"attention_weights data type: {attention_weights.dtype}")
        # Normalize attention weights to sum to 1
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Apply attention weights to the value
        filtered_features = attention_weights * value
        return filtered_features


class ClassesToMask_v2(nn.Module):
    def __init__(self, class_ids: List[int], threshold: float = 0.25, use_threshold: bool = True):
        super(ClassesToMask_v2, self).__init__()
        self.num_classes = len(class_ids)
        self.class_ids = class_ids
        self.threshold = threshold
        self.use_threshold = use_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"ClassesToMask_v2 input shape: {x.shape}")

        # Ensure class_ids are within bounds
        valid_class_ids = [i for i in self.class_ids if i < x.shape[1]]
        if len(valid_class_ids) == 0:
            raise ValueError(f"No valid class_ids. Input has {x.shape[1]} channels, but class_ids are {self.class_ids}")

        # Select only the channels corresponding to the valid class_ids
        x = x[:, valid_class_ids, :, :]

        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(x)

        # Get the maximum probability across all classes
        max_probs, _ = torch.max(probs, dim=1, keepdim=True)

        if self.use_threshold:
            # Apply threshold to create binary mask
            mask = (max_probs > self.threshold).float()
        else:
            # Use probability values as mask
            mask = max_probs

        return mask


class AveragePool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(AveragePool2d, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x) -> torch.Tensor:
        return self.avg_pool(x)


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation='gelu', same_dims=False):
        super(Conv2D, self).__init__()
        if same_dims:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation

        # He Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'relu':
            x = F.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, num_channels):
        super(SpatialAttention, self).__init__()
        self.query_linear = nn.Linear(num_channels, num_channels)
        self.key_linear = nn.Linear(num_channels, num_channels)
        self.value_linear = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        # Compute attention scores
        query = F.gelu(self.query_linear(x))
        key = F.gelu(self.key_linear(x))
        value = F.gelu(self.value_linear(x))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Compute output
        output = torch.matmul(attention_scores, value)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        # Compute channel attention scores
        x_avg_pool = self.avg_pool(x).squeeze(-1).squeeze(-1)
        attention_scores = F.gelu(self.fc(x_avg_pool))
        attention_scores = F.sigmoid(attention_scores)

        # Compute output
        output = x * attention_scores.unsqueeze(-1).unsqueeze(-1)
        return output
