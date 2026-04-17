import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=3):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        encoder_blocks = []
        decoder_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                encoder_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(encoder_blocks)
        self.bridge = Bridge(2048, 2048)
        decoder_blocks.append(DecoderBlock(2048, 1024))
        decoder_blocks.append(DecoderBlock(1024, 512))
        decoder_blocks.append(DecoderBlock(512, 256))
        decoder_blocks.append(DecoderBlock(in_channels=128 + 64, out_channels=128,
                                           up_conv_in_channels=256, up_conv_out_channels=128))
        decoder_blocks.append(DecoderBlock(in_channels=64 + 3, out_channels=64,
                                           up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(decoder_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = dict()
        skip_connections["layer_0"] = x
        x = self.input_block(x)
        skip_connections["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            skip_connections[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, skip_connections[key])

        x = self.out(x)
        del skip_connections
        return x
