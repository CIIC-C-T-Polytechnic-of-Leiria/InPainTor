from typing import List

from torch import cat
from torch.nn import Module

from layers import SepConvBlock, SepConvTranspBlock, Conv1x1, AttentionBlock, AveragePool2d, ClassesToMask, Conv2D


# TODO: PROBLEMA no summary: erro de dimensões. Verificar se o erro está na implementação ou no summary!!
# TODO: Mecanismo para seleção de classe após a segmentação - In Progress...
# TODO: Substituir Conv2D por ResidualConv2D?
# TODO: Rever ligações de AttentionBlock: são estas as ligações que devem ser feitas?
# TODO: Consultar Livro Generative Models

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
        # print(f"Enc1: {enc1.shape}, Enc2: {enc2.shape}, Enc3: {enc3.shape}, Enc4: {enc4.shape}")
        return enc1, enc2, enc3, enc4


class SegmentorDecoder(Module):
    def __init__(self, in_channels: int, num_classes: int = 40, selected_classes: List[int] = [1, 2, 3],
                 base_chs: int = 16):
        super(SegmentorDecoder, self).__init__()
        self.conv_transp_block1 = SepConvTranspBlock(in_channels, out_channels=base_chs * 16)
        self.conv_transp_block2 = SepConvTranspBlock(in_channels=base_chs * 24, out_channels=base_chs * 8)
        self.conv_transp_block3 = SepConvTranspBlock(in_channels=base_chs * 12, out_channels=base_chs * 4)
        self.conv1x1 = Conv1x1(in_channels=base_chs * 4, out_channels=num_classes, stride=1, activation=None)
        self.class_selector = ClassesToMask(num_classes=num_classes, class_ids=selected_classes)

    def forward(self, enc4, enc3, enc2):
        seg1 = self.conv_transp_block1(enc4)
        seg_cat1 = cat([seg1, enc3], dim=1)  # seg1: (64, 64, 512), enc3: (64, 64, 256)
        seg2 = self.conv_transp_block2(seg_cat1)
        seg_cat2 = cat([seg2, enc2], dim=1)
        seg3 = self.conv_transp_block3(seg_cat2)
        seg_out = self.conv1x1(seg3)
        masked_out = self.class_selector(seg_out)
        return masked_out, seg2, seg3


class GenerativeDecoder(Module):
    def __init__(self, base_chs: int = 16):
        super(GenerativeDecoder, self).__init__()
        self.conv_transp_block1 = SepConvTranspBlock(in_channels=base_chs * 16, out_channels=base_chs * 16)
        self.conv_transp_block2 = SepConvTranspBlock(in_channels=base_chs * 16, out_channels=base_chs * 8)
        self.conv_transp_block3 = SepConvTranspBlock(in_channels=base_chs * 16, out_channels=base_chs * 8)
        # self.conv_transp_block4 = SepConvTranspBlock(in_channels=384, out_channels=128, kernel_size=3, stride=2)
        self.conv1 = Conv2D(in_channels=base_chs * 12, out_channels=base_chs * 4, stride=1, same_dims=True)
        self.conv_transp_block5 = SepConvTranspBlock(in_channels=(base_chs * 4) + 3, out_channels=64)
        # self.conv_transp_block6 = SepConvTranspBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = Conv2D(in_channels=base_chs * 4, out_channels=base_chs, stride=1, same_dims=True)
        self.conv3 = Conv2D(in_channels=base_chs, out_channels=base_chs, stride=1, same_dims=True)
        self.conv1x1 = Conv1x1(in_channels=base_chs, out_channels=3, stride=1, activation=None)
        self.average_pool = AveragePool2d(kernel_size=2, stride=2)
        self.attention_block1 = AttentionBlock(in_channels=base_chs * 8, attention_dim=base_chs * 8)
        self.attention_block2 = AttentionBlock(in_channels=base_chs * 4, attention_dim=base_chs * 4)

    def forward(self, input_image, enc4, masked_out, seg2, seg3):
        input_small = self.average_pool(input_image)
        masked_input = input_small * masked_out.unsqueeze(1)
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
    def __init__(self, num_classes: int = 40, selected_classes: List[int] = [1, 2, 3], base_chs: int = 16):
        super(InpainTor, self).__init__()
        self.shared_encoder = SharedEncoder()
        self.segment_decoder = SegmentorDecoder(in_channels=base_chs * 16, num_classes=num_classes,
                                                selected_classes=selected_classes)
        self.generative_decoder = GenerativeDecoder()

    def forward(self, x):
        _, enc2, enc3, enc4 = self.shared_encoder(x)
        masked_out, seg2, seg3 = self.segment_decoder(enc4, enc3, enc2)
        # print(f"\n seg_out.shape: {seg_out.shape}") input_image, enc4, masked_out, seg2, seg3):
        out_gen = self.generative_decoder(x, enc4, masked_out, seg2, seg3)

        return out_gen
