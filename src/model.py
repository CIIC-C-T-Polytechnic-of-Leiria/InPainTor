"""
InpaintTor Model Definition

This script defines the `InpaintTor` model, which is an architecture combining shared encoding, segmentation decoding, and generative decoding blocks. The model is designed for tasks involving inpainting and segmentation. The components of the model include:

1. **SharedEncoder**: Encodes input images into a series of feature maps.
2. **SegmentorDecoder**: Decodes encoded features into segmentation masks.
3. **GenerativeDecoder**: Uses segmentation information to generate inpainted images.

Classes:
    - SharedEncoder: A neural network encoder with several convolutional blocks for feature extraction.
    - SegmentorDecoder: A neural network decoder that generates segmentation masks from encoded features.
    - GenerativeDecoder: A neural network decoder that performs image generation using segmentation masks and encoded features.
    - InpaintTor: The main model class that integrates the encoder, segmentation decoder, and generative decoder. It also includes methods for freezing/unfreezing parts of the network and saving/loading model states.

Usage:
    To use the `InpaintTor` model for training or inference, follow the steps below:

    1. **Instantiate the Model**:
        ```python
        from your_module import InpaintTor

        # Create an instance of the InpaintTor model
        model = InpaintTor(selected_classes=[0, 1, 2], base_chs=32)
        ```

    2. **Forward Pass**:
        ```python
        import torch

        # Create a dummy input tensor with the shape (batch_size, channels, height, width)
        input_tensor = torch.randn(1, 3, 512, 512)

        # Perform a forward pass through the model
        outputs = model(input_tensor)

        # Access the inpainted image and mask
        inpainted_image = outputs['inpainted_image']
        mask = outputs['mask']
        ```

    3. **Freezing and Unfreezing Parts of the Model**:
        ```python
        # Freeze the generator part of the model
        model.freeze_part("generator")

        # Unfreeze the generator part of the model
        model.unfreeze_part("generator")
        ```

    4. **Saving and Loading Model State**:
        ```python
        # Save the model state
        state_dict = model.get_state_dict("full")
        torch.save(state_dict, "model.pth")

        # Load the model state
        loaded_state_dict = torch.load("model.pth")
        model.load_state_dict(loaded_state_dict, "full")
        ```

Components:
    - **SharedEncoder**: Consists of multiple `SepConvBlock` layers for progressively encoding the input image.
    - **SegmentorDecoder**: Uses `SepConvTranspBlock` and `Conv1x1` layers to decode encoded features into segmentation masks.
    - **GenerativeDecoder**: Combines several `SepConvTranspBlock` and `Conv2D` layers, along with attention blocks, to generate the final inpainted image.
    - **InpaintTor**: Integrates the encoder, segmentation decoder, and generative decoder, and includes methods for managing model parameters and states.

Dependencies:
    - PyTorch (`torch`, `torchvision`)
    - Custom layers (`AttentionBlock`, `AveragePool2d`, `ClassesToMask_v2`, `SepConvBlock`, `SepConvTranspBlock`, `Conv1x1`, `Conv2D`, `Sigmoid`)

Note:
    - TODO: Consider replacing `Conv2D` with `ResidualConv2D` for potential improvements.
    - TODO: Explore using the ENet model as a base for the segmentation decoder.

Exceptions:
    - ValueError: Raised for invalid part specifications in methods for freezing/unfreezing and state dict operations.

"""

import os
from typing import List, Dict, Any

from torch import cat
from torch.nn import Module
from torchvision.utils import save_image

from layers import AttentionBlock, AveragePool2d
from layers import ClassesToMask_v2
from layers import SepConvBlock, SepConvTranspBlock, Conv1x1, Conv2D, Sigmoid


class SharedEncoder(Module):
    def __init__(self, input_size: int = 512, base_chs: int = 16):
        super(SharedEncoder, self).__init__()
        self.conv_block1 = SepConvBlock(in_channels=3, out_channels=base_chs * 2, input_size=input_size)
        self.conv_block2 = SepConvBlock(in_channels=base_chs * 2, out_channels=base_chs * 4, input_size=input_size // 2)
        self.conv_block3 = SepConvBlock(in_channels=base_chs * 4, out_channels=base_chs * 8, input_size=input_size // 4)
        self.conv_block4 = SepConvBlock(in_channels=base_chs * 8, out_channels=base_chs * 16,
                                        input_size=input_size // 8)

    def forward(self, input_tensor):
        enc1 = self.conv_block1(input_tensor)
        enc2 = self.conv_block2(enc1)
        enc3 = self.conv_block3(enc2)
        enc4 = self.conv_block4(enc3)
        return enc1, enc2, enc3, enc4


class SegmentorDecoder(Module):
    def __init__(self, in_channels: int, num_classes: int, base_chs: int = 16):
        super(SegmentorDecoder, self).__init__()
        self.conv_transp_block1 = SepConvTranspBlock(in_channels, out_channels=base_chs * 16)
        self.conv_transp_block2 = SepConvTranspBlock(in_channels=base_chs * 24, out_channels=base_chs * 8)
        self.conv_transp_block3 = SepConvTranspBlock(in_channels=base_chs * 12, out_channels=base_chs * 4)
        self.conv1x1 = Conv1x1(in_channels=base_chs * 4, out_channels=num_classes, stride=1, activation=None)
        # self.log_softmax = LogSoftmax(dim=1)
        # self.log_softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()

    def forward(self, enc4, enc3, enc2):
        seg1 = self.conv_transp_block1(enc4)
        seg_cat1 = cat([seg1, enc3], dim=1)
        seg2 = self.conv_transp_block2(seg_cat1)
        seg_cat2 = cat([seg2, enc2], dim=1)
        seg3 = self.conv_transp_block3(seg_cat2)
        seg_out = self.conv1x1(seg3)
        # classes_out = self.log_softmax(seg_out)
        classes_out = self.sigmoid(seg_out)
        return classes_out, seg2, seg3


class GenerativeDecoder(Module):
    def __init__(self, selected_classes: List[int], base_chs: int = 16):
        super(GenerativeDecoder, self).__init__()
        # self.class_selector = ClassesToMask(num_classes=num_classes, class_ids=selected_classes, use_threshold=True)
        self.class_selector = ClassesToMask_v2(class_ids=selected_classes, use_threshold=True)
        self.conv_transp_block1 = SepConvTranspBlock(in_channels=base_chs * 16, out_channels=base_chs * 16)
        self.conv_transp_block2 = SepConvTranspBlock(in_channels=base_chs * 16, out_channels=base_chs * 8)
        self.conv_transp_block3 = SepConvTranspBlock(in_channels=base_chs * 16, out_channels=base_chs * 8)
        self.conv1 = Conv2D(in_channels=base_chs * 12, out_channels=base_chs * 4, stride=1, same_dims=True)
        self.conv_transp_block5 = SepConvTranspBlock(in_channels=(base_chs * 4) + 3, out_channels=64)
        self.conv2 = Conv2D(in_channels=base_chs * 4, out_channels=base_chs, stride=1, same_dims=True)
        self.conv3 = Conv2D(in_channels=base_chs, out_channels=base_chs, stride=1, same_dims=True)
        self.conv1x1 = Conv1x1(in_channels=base_chs, out_channels=3, stride=1, activation="sigmoid")
        self.average_pool = AveragePool2d(kernel_size=2, stride=2)
        self.attention_block1 = AttentionBlock(in_channels=base_chs * 8, attention_dim=base_chs * 8)
        self.attention_block2 = AttentionBlock(in_channels=base_chs * 4, attention_dim=base_chs * 4)

        self.counter = 0
        self.save_path = "outputs/debug"
        os.makedirs(self.save_path, exist_ok=True)

    def save_mask(self, tensor, step, name):
        save_file = os.path.join(self.save_path, f'{name}_step_{step}_sample_0.png')
        save_image(tensor[0], save_file)

    def forward(self, input_image, enc4, classes_out, seg2, seg3):
        self.counter += 1
        input_small = self.average_pool(input_image)

        # print(f"GenerativeDecoder - input_image shape: {input_image.shape}")
        # print(f"GenerativeDecoder - classes_out shape: {classes_out.shape}")
        # print(f"GenerativeDecoder - enc4 shape: {enc4.shape}")
        # print(f"GenerativeDecoder - seg2 shape: {seg2.shape}")
        # print(f"GenerativeDecoder - seg3 shape: {seg3.shape}")

        masked_out = self.class_selector(classes_out)

        # print(f"GenerativeDecoder - masked_out shape: {masked_out.shape}")

        # masked_input = input_small * masked_out
        masked_input = input_small * masked_out

        # save the masked input in the debug folder in a png file

        # print(f"GenerativeDecoder - masked_input shape: {masked_input.shape}")

        if self.counter % 100 == 0:
            self.save_mask(masked_input, self.counter, name="masked_input")

        attention1 = self.attention_block1(seg2)
        attention2 = self.attention_block2(seg3)
        gen1 = self.conv_transp_block1(enc4)
        gen2 = self.conv_transp_block2(gen1)
        gen_cat1 = cat([attention1, gen2], dim=1)
        gen3 = self.conv_transp_block3(gen_cat1)
        gen_cat2 = cat([attention2, gen3], dim=1)
        gen4 = self.conv1(gen_cat2)
        gen_cat3 = cat([masked_input, gen4], dim=1)
        gen5 = self.conv_transp_block5(gen_cat3)
        gen6 = self.conv2(gen5)
        gen7 = self.conv3(gen6)
        gen_out = self.conv1x1(gen7)

        return gen_out


class InpainTor(Module):
    def __init__(self, selected_classes: List[int] = [0], base_chs: int = 16):
        super().__init__()
        self.shared_encoder = SharedEncoder()
        self.segment_decoder = SegmentorDecoder(in_channels=base_chs * 16, num_classes=len(selected_classes))
        self.generative_decoder = GenerativeDecoder(base_chs=base_chs, selected_classes=selected_classes)

    def forward(self, x):
        _, enc2, enc3, enc4 = self.shared_encoder(x)
        masked_out, seg2, seg3 = self.segment_decoder(enc4, enc3, enc2)
        out_gen = self.generative_decoder(x, enc4, masked_out, seg2, seg3)
        return {'mask': masked_out, 'inpainted_image': out_gen}

    def freeze_part(self, part: str):
        if part == "generator":
            for param in self.generative_decoder.parameters():
                param.requires_grad = False
        elif part == "encoder_segmentor":
            for param in self.shared_encoder.parameters():
                param.requires_grad = False
            for param in self.segment_decoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Invalid part specified. Use 'generator' or 'encoder_segmentor'.")

    def unfreeze_part(self, part: str):
        if part == "generator":
            for param in self.generative_decoder.parameters():
                param.requires_grad = True
        elif part == "encoder_segmentor":
            for param in self.shared_encoder.parameters():
                param.requires_grad = True
            for param in self.segment_decoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid part specified. Use 'generator' or 'encoder_segmentor'.")

    def get_state_dict(self, part: str = "full"):
        if part == "full":
            return self.state_dict()
        elif part == "encoder_segmentor":
            return {
                'shared_encoder': self.shared_encoder.state_dict(),
                'segment_decoder': self.segment_decoder.state_dict()
            }
        elif part == "generator":
            return self.generative_decoder.state_dict()
        else:
            raise ValueError("Invalid part specified. Use 'full', 'encoder_segmentor', or 'generator'.")

    def load_state_dict(self, state_dict: Dict[str, Any], part: str = "full"):
        if part == "full":
            super().load_state_dict(state_dict)
        elif part == "encoder_segmentor":
            self.shared_encoder.load_state_dict(state_dict['shared_encoder'])
            self.segment_decoder.load_state_dict(state_dict['segment_decoder'])
        elif part == "generator":
            self.generative_decoder.load_state_dict(state_dict)
        else:
            raise ValueError("Invalid part specified. Use 'full', 'encoder_segmentor', or 'generator'.")
