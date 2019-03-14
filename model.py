import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedBlockBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False):
        super(GatedBlockBN, self).__init__()
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.BatchNorm2d(c_out)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = torch.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class GatedBlockIN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False):
        super(GatedBlockIN, self).__init__()
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.InstanceNorm2d(c_out)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.InstanceNorm2d(c_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = torch.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class GatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False):
        super(GatedBlock, self).__init__()
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        return out

class ReluBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False):
        super(ReluBlock, self).__init__()
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

# class Encoder(nn.Module):
    # def __init__(self, conv_dim=16, c_in=1, num_repeat=3, block_type='normal'):
        # super(Encoder, self).__init__()
        # block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN}[block_type]
        # layers = []
        # layers.append(block(c_in=c_in, c_out=conv_dim, kernel_size=3, stride=1, padding=1, deconv=False))

        # for i in range(num_repeat-1):
            # layers.append(block(c_in=conv_dim, c_out=conv_dim*2, kernel_size=4, stride=2, padding=1, deconv=False))
            # conv_dim *= 2

        # self.main = nn.Sequential(*layers)

    # def forward(self, x):
        # h = self.main(x)
        # return h

# class CarrierDecoder(nn.Module):
    # def __init__(self, conv_dim, num_repeat=3, block_type='normal'):
        # super(CarrierDecoder, self).__init__()
        # block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN}[block_type]
        # layers = []

        # for i in range(num_repeat-1):
            # layers.append(block(c_in=int(conv_dim), c_out=int(conv_dim/2), kernel_size=4, stride=2, padding=1, deconv=True))
            # conv_dim /= 2

        # layers.append(block(c_in=int(conv_dim), c_out=1, kernel_size=4, stride=1, padding=1, deconv=True))
        # self.main = nn.Sequential(*layers)

    # def forward(self, x):
        # h = self.main(x)
        # return h

class Encoder(nn.Module):
    def __init__(self, conv_dim=16, c_in=1, num_repeat=3, block_type='normal'):
        super(Encoder, self).__init__()
        block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN, 'relu': ReluBlock}[block_type]
        layers = []
        layers.append(block(c_in=c_in, c_out=conv_dim, kernel_size=3, stride=1, padding=1, deconv=False))

        for i in range(num_repeat-1):
            layers.append(block(c_in=conv_dim, c_out=conv_dim*2, kernel_size=3, stride=1, padding=1, deconv=False))
            conv_dim *= 2

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class CarrierDecoder(nn.Module):
    def __init__(self, conv_dim, num_repeat=3, block_type='normal'):
        super(CarrierDecoder, self).__init__()
        block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN, 'relu': ReluBlock}[block_type]
        layers = []

        for i in range(num_repeat-1):
            layers.append(block(c_in=int(conv_dim), c_out=int(conv_dim/2), kernel_size=3, stride=1, padding=1, deconv=True))
            conv_dim /= 2

        layers.append(block(c_in=int(conv_dim), c_out=1, kernel_size=3, stride=1, padding=1, deconv=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class MsgDecoder(nn.Module):
    def __init__(self, conv_dim, num_repeat=4, filter_increase_factor=4, block_type='normal'):
        super(MsgDecoder, self).__init__()
        block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN, 'relu': ReluBlock}[block_type]
        assert num_repeat % 2 == 0
        half_num_repeat = int(num_repeat/2)
        layers = []

        # increase filters
        for i in range(half_num_repeat):
            layers.append(block(c_in=conv_dim, c_out=conv_dim*filter_increase_factor, kernel_size=3, stride=1, padding=1, deconv=False))
            conv_dim *= filter_increase_factor

        # decrease filters
        for i in range(half_num_repeat):
            layers.append(block(c_in=int(conv_dim), c_out=int(conv_dim/filter_increase_factor), kernel_size=3, stride=1, padding=1, deconv=False))
            conv_dim /= filter_increase_factor

        # layers.append(block(c_in=int(conv_dim), c_out=1, kernel_size=3, stride=1, padding=1, deconv=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                GatedBlockBN(1,16,3,1,1),
                GatedBlockBN(16,32,3,1,1),
                GatedBlockBN(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x
