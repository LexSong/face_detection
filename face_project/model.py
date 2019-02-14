import torch.nn as nn


def Conv2dLReLU(in_, out):
    return nn.Sequential(
        nn.Conv2d(in_, out, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


def Downscale(in_, out):
    return nn.Sequential(
        Conv2dLReLU(in_, in_),
        Conv2dLReLU(in_, out),
        nn.AvgPool2d(2),
    )


def Upscale(in_, out):
    return nn.Sequential(
        Conv2dLReLU(in_, out),
        Conv2dLReLU(out, out * 4),
        nn.PixelShuffle(2),
    )


def Encoder_4x4(in_, out):
    return nn.Sequential(
        Conv2dLReLU(in_, in_),
        nn.Conv2d(in_, out, kernel_size=4),
        nn.LeakyReLU(0.2, inplace=True),
    )


def Decoder_4x4(in_, out):
    return nn.Sequential(
        nn.Conv2d(in_, out * 16, kernel_size=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.PixelShuffle(4),
        Conv2dLReLU(out, out),
    )


def FromRGB(n):
    return nn.Sequential(
        nn.Conv2d(3, n, kernel_size=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


def ToRGB(n):
    return nn.Conv2d(n, 3, kernel_size=1)


class ProgressiveAutoencoder(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.max_layer_num = len(depths) - 1

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(Encoder_4x4(depths[1], depths[0]))
        for inner, outer in zip(depths[1:], depths[2:]):
            self.encoder_layers.append(Downscale(outer, inner))

        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(Decoder_4x4(depths[0], depths[1]))
        self.decoder_layers.append(Upscale(depths[1], depths[2]))

        self.decoder_A_layers = nn.ModuleList()
        self.decoder_B_layers = nn.ModuleList()
        for inner, outer in zip(depths[2:], depths[3:]):
            self.decoder_A_layers.append(Upscale(inner, outer))
            self.decoder_B_layers.append(Upscale(inner, outer))

        self.from_rgb = nn.ModuleList([FromRGB(x) for x in depths[1:]])
        self.to_rgb = nn.ModuleList([ToRGB(x) for x in depths[1:]])

    def forward(self, x, layer_num=None, label='A'):
        if layer_num is None:
            layer_num = self.max_layer_num

        encoder_layers = list(reversed(self.encoder_layers[:layer_num]))

        decoder_layers = list(self.decoder_layers)
        if label == 'A':
            decoder_layers += list(self.decoder_A_layers)
        elif label == 'B':
            decoder_layers += list(self.decoder_B_layers)
        else:
            raise ValueError(f"Unknown autoencoder label: {label}")
        decoder_layers = decoder_layers[:layer_num]

        layers = encoder_layers + decoder_layers

        x = self.from_rgb[layer_num - 1](x)
        for layer in layers:
            x = layer(x)
        x = self.to_rgb[layer_num - 1](x)
        return x
