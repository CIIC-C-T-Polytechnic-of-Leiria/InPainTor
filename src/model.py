import os
from typing import List

import torch
from torch import cat
from torch.nn import Module
from torchvision.utils import save_image

from layers import AttentionBlock, AveragePool2d, LogSoftmax, ClassesToMask
from layers import SepConvBlock, SepConvTranspBlock, Conv1x1, Conv2D


# TODO: Substituir Conv2D por ResidualConv2D? - Todo later...

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
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, enc4, enc3, enc2):
        seg1 = self.conv_transp_block1(enc4)
        seg_cat1 = cat([seg1, enc3], dim=1)
        seg2 = self.conv_transp_block2(seg_cat1)
        seg_cat2 = cat([seg2, enc2], dim=1)
        seg3 = self.conv_transp_block3(seg_cat2)
        seg_out = self.conv1x1(seg3)
        classes_out = self.log_softmax(seg_out)
        return classes_out, seg2, seg3


class GenerativeDecoder(Module):
    def __init__(self, selected_classes: List[int], base_chs: int = 16, num_classes: int = 80):
        super(GenerativeDecoder, self).__init__()
        self.class_selector = ClassesToMask(num_classes=num_classes, class_ids=selected_classes, use_threshold=True)
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
        masked_out = self.class_selector(classes_out)
        masked_input = input_small * masked_out

        if self.counter % 10 == 0:
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
    def __init__(self, num_classes: int = 80, selected_classes: List[int] = [0], base_chs: int = 16):
        super().__init__()
        self.shared_encoder = SharedEncoder()
        self.segment_decoder = SegmentorDecoder(in_channels=base_chs * 16, num_classes=num_classes)
        self.generative_decoder = GenerativeDecoder(base_chs=base_chs, num_classes=num_classes,
                                                    selected_classes=selected_classes)

    def forward(self, x):
        _, enc2, enc3, enc4 = self.shared_encoder(x)
        masked_out, seg2, seg3 = self.segment_decoder(enc4, enc3, enc2)
        out_gen = self.generative_decoder(x, enc4, masked_out, seg2, seg3)
        return {'mask': masked_out, 'inpainted_image': out_gen}

    def freeze_segmentor(self):
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        for param in self.segment_decoder.parameters():
            param.requires_grad = False

    def unfreeze_segmentor(self):
        for param in self.shared_encoder.parameters():
            param.requires_grad = True
        for param in self.segment_decoder.parameters():
            param.requires_grad = True

    def freeze_generator(self):
        for param in self.generative_decoder.parameters():
            param.requires_grad = False

    def unfreeze_generator(self):
        for param in self.generative_decoder.parameters():
            param.requires_grad = True

    def save_segmentor(self, path):
        torch.save({
            'shared_encoder': self.shared_encoder.state_dict(),
            'segment_decoder': self.segment_decoder.state_dict(),
        }, path)

    def load_segmentor(self, path):
        checkpoint = torch.load(path)
        self.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        self.segment_decoder.load_state_dict(checkpoint['segment_decoder'])

    def save_generator(self, path):
        torch.save(self.generative_decoder.state_dict(), path)

    def load_generator(self, path):
        self.generative_decoder.load_state_dict(torch.load(path))

    def save_full_model(self, path):
        torch.save(self.state_dict(), path)

    def load_full_model(self, path):
        self.load_state_dict(torch.load(path))

# class InpainTor(Module):
#     """
#     InpainTor: A Generative model_ for segmentation based inpainting.
#
#     Args:
#         num_classes (int): Number of classes in the dataset (defaults to COCO dataset classes: 80).
#         selected_classes (List[int]): List of selected classes for segmentation (defaults to [0]: person).
#         base_chs (int): Base number of channels for the model_ (defaults to 16).
#
#     Returns:
#         dict: Dictionary containing the mask and inpainted image.
#     """
#
#     def __init__(self, num_classes: int = 80, selected_classes: List[int] = [0], base_chs: int = 16):
#         super().__init__()  # Call the parent class's __init__ method
#         self.shared_encoder = SharedEncoder()
#         self.segment_decoder = SegmentorDecoder(in_channels=base_chs * 16, num_classes=num_classes)
#         # selected_classes=selected_classes
#         self.generative_decoder = GenerativeDecoder(base_chs=base_chs, num_classes=num_classes,
#                                                     selected_classes=selected_classes)
#
#     def forward(self, x):
#         _, enc2, enc3, enc4 = self.shared_encoder(x)
#         masked_out, seg2, seg3 = self.segment_decoder(enc4, enc3, enc2)
#         out_gen = self.generative_decoder(x, enc4, masked_out, seg2, seg3)
#         return {'mask': masked_out, 'inpainted_image': out_gen}
