import os
from typing import List

from torch import cat
from torch.nn import Module
from torchvision.utils import save_image

from layers import AttentionBlock, AveragePool2d, LogSoftmax, ClassesToMask
from layers import SepConvBlock, SepConvTranspBlock, Conv1x1, Conv2D


# TODO: Mecanismo para seleção de classe após a segmentação - In Progress...
# TODO: Consultar Livro Generative Models - In Progress...
# TODO: Substituir Conv2D por ResidualConv2D? - Todo later...

class SharedEncoder(Module):
    """
    Shared Encoder: 4 successive SepConvBlocks. The second, third, and fourth blocks have a skip connections.
    """

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
        # print(
        #     f"Input: {input_tensor.shape}, Enc1: {enc1.shape}, Enc2: {enc2.shape}, Enc3: {enc3.shape}, Enc4: {enc4.shape}")
        return enc1, enc2, enc3, enc4


class SegmentorDecoder(Module):
    def __init__(self, in_channels: int, num_classes: int, base_chs: int = 16):
        super(SegmentorDecoder, self).__init__()
        self.conv_transp_block1 = SepConvTranspBlock(in_channels, out_channels=base_chs * 16)
        self.conv_transp_block2 = SepConvTranspBlock(in_channels=base_chs * 24, out_channels=base_chs * 8)
        self.conv_transp_block3 = SepConvTranspBlock(in_channels=base_chs * 12, out_channels=base_chs * 4)
        self.conv1x1 = Conv1x1(in_channels=base_chs * 4, out_channels=num_classes, stride=1, activation=None)

        self.log_softmax = LogSoftmax(dim=1)  # LogSoftmax for segmentation output, use NLLLoss for loss calculation
        # self.softmax = Softmax(dim=1)  # Softmax for segmentation output, use CrossEntropyLoss for loss calculation
        # self.class_selector = ClassesToMask(num_classes=num_classes, class_ids=selected_classes)

    def forward(self, enc4, enc3, enc2):
        seg1 = self.conv_transp_block1(enc4)
        # print(f"Seg1: {seg1.shape}, Enc3: {enc3.shape}, Enc4: {enc4.shape}")
        seg_cat1 = cat([seg1, enc3], dim=1)  # seg1: (64, 64, 512), enc3: (64, 64, 256)
        seg2 = self.conv_transp_block2(seg_cat1)
        seg_cat2 = cat([seg2, enc2], dim=1)
        seg3 = self.conv_transp_block3(seg_cat2)
        seg_out = self.conv1x1(seg3)
        classes_out = self.log_softmax(seg_out)
        # classes_out = self.class_selector(seg_out)
        # print(f"\nSeg_out shape: {seg_out.shape}, Masked_out shape: {classes_out.shape}")
        # print(f"\nSeg_out: {seg_out}, \nMasked_out: {classes_out}")

        return classes_out, seg2, seg3


class GenerativeDecoder(Module):
    def __init__(self, selected_classes: List[int], base_chs: int = 16, num_classes: int = 80, ):
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

        # -- TODO DEBUG: save masked_input to disk ----------------
        self.counter = 0
        self.save_path = "outputs/debug"
        os.makedirs(self.save_path, exist_ok=True)

    def save_mask(self, tensor, step, name):
        batch_size = tensor.size(0)
        save_file = os.path.join(self.save_path, f'{name}_step_{step}_sample_{0}.png')
        save_image(tensor[0], save_file)

        # for i in range(batch_size):
        #     save_file = os.path.join(self.save_path, f'{name}_step_{step}_sample_{i}.png')
        #     save_image(tensor[i], save_file)
        # --------------------------------------------------------------

    def forward(self, input_image, enc4, classes_out, seg2, seg3):
        self.counter += 1

        input_small = self.average_pool(input_image)
        # print(f"classes_out: {classes_out}")
        masked_out = self.class_selector(classes_out)

        # print(f"\nBEFORE: input_small.shape: {input_small.shape}", f"masked_out.shape: {masked_out.shape}")
        masked_input = input_small * masked_out

        # -- TODO: DEBUG: save masked_input to disk ---------------
        if self.counter % 10 == 0:
            self.save_mask(masked_input, self.counter, name="masked_input")
            # self.save_mask(masked_out, self.counter, name="masked_out")
            # print(f"\nmasked_out max: {masked_out.max():.3f}, masked_out min: {masked_out.min():.3f}")
            # print(f"masked_input MAX: {masked_input.max():.3f}, masked_input MIN: {masked_input.min():.3f}")
            # print(f"masked_out mean: {masked_out.mean():.3f}, masked_input mean: {masked_input.mean():.3f}")
        # --------------------------------------------------------------

        # print(f"AFTER: masked_input.shape: {masked_input.shape}")
        attention1 = self.attention_block1(seg2)
        attention2 = self.attention_block2(seg3)
        # print(f"enc4.shape: {enc4.shape}")

        gen1 = self.conv_transp_block1(enc4)
        gen2 = self.conv_transp_block2(gen1)
        gen_cat1 = cat([attention1, gen2], dim=1)
        # print(f"attention1.shape: {attention1.shape}, gen2.shape: {gen2.shape}, gen_cat1.shape: {gen_cat1.shape}")
        gen3 = self.conv_transp_block3(gen_cat1)
        gen_cat2 = cat([attention2, gen3], dim=1)
        # print(f"gen_cat2.shape: {gen_cat2.shape}")
        # gen4 = self.conv_transp_block4(gen_cat2)
        gen4 = self.conv1(gen_cat2)
        # print(f"gen4.shape: {gen4.shape}, masked_input.shape: {masked_input.shape}")
        gen_cat3 = cat([masked_input, gen4], dim=1)
        gen5 = self.conv_transp_block5(gen_cat3)
        # print(f"gen5.shape: {gen5.shape}")
        # gen6 = self.conv_transp_block6(gen5)
        gen6 = self.conv2(gen5)
        gen7 = self.conv3(gen6)
        # print(f"gen7.shape: {gen7.shape}")
        gen_out = self.conv1x1(gen7)

        return gen_out


class InpainTor(Module):
    """
    InpainTor: A Generative model for segmentation based inpainting.

    Args:
        num_classes (int): Number of classes in the dataset (defaults to COCO dataset classes: 80).
        selected_classes (List[int]): List of selected classes for segmentation (defaults to [0]: person).
        base_chs (int): Base number of channels for the model (defaults to 16).

    Returns:
        dict: Dictionary containing the mask and inpainted image.
    """

    def __init__(self, num_classes: int = 80, selected_classes: List[int] = [0], base_chs: int = 16):
        super().__init__()  # Call the parent class's __init__ method
        self.shared_encoder = SharedEncoder()
        self.segment_decoder = SegmentorDecoder(in_channels=base_chs * 16, num_classes=num_classes)
        # selected_classes=selected_classes
        self.generative_decoder = GenerativeDecoder(base_chs=base_chs, num_classes=num_classes,
                                                    selected_classes=selected_classes)

    def forward(self, x):
        _, enc2, enc3, enc4 = self.shared_encoder(x)
        masked_out, seg2, seg3 = self.segment_decoder(enc4, enc3, enc2)
        out_gen = self.generative_decoder(x, enc4, masked_out, seg2, seg3)
        return {'mask': masked_out, 'inpainted_image': out_gen}
