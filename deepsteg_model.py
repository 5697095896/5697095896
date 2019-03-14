import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GatedBlock, GatedBlockBN, GatedBlockIN, ReluBlock

class Encoder(nn.Module):
    def __init__(self, conv_dim=1, block_type='normal'):
        super(Encoder, self).__init__()
        block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN, 'relu': ReluBlock}[block_type]

        self.main = nn.Sequential(
                block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)
                )

    def forward(self, x):
        h = self.main(x)
        return h

class CarrierDecoder(nn.Module):
    def __init__(self, conv_dim, block_type='normal'):
        super(CarrierDecoder, self).__init__()
        block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN, 'relu': ReluBlock}[block_type]
        layers = []

        self.main = nn.Sequential(
                block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=1, kernel_size=1, stride=1, padding=0, deconv=False)
                )

    def forward(self, x):
        h = self.main(x)
        return h

class MsgDecoder(nn.Module):
    def __init__(self, conv_dim=1, block_type='normal'):
        super(MsgDecoder, self).__init__()
        block = {'normal': GatedBlock, 'bn': GatedBlockBN, 'in': GatedBlockIN, 'relu': ReluBlock}[block_type]
        layers = []

        self.main = nn.Sequential(
                block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=1, kernel_size=3, stride=1, padding=1, deconv=False)
                )

    def forward(self, x):
        h = self.main(x)
        return h
