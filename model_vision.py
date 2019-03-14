import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_3x3 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                )
        self.conv_5x5 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                )
        self.conv_7x7 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                )
        self.conv_3x3_end = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU()
                )
        self.conv_5x5_end = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU()
                )
        self.conv_7x7_end = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU()
                )

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)  # B,50,F,T
        conv_5x5 = self.conv_5x5(x)  # B,50,F,T
        conv_7x7 = self.conv_7x7(x)  # B,50,F,T
        concat = torch.cat((conv_3x3, conv_5x5, conv_7x7), dim=1)  # B,150,F,T
        conv_3x3_end = self.conv_3x3_end(concat)  # B,50,F,T
        conv_5x5_end = self.conv_5x5_end(concat)  # B,50,F,T
        conv_7x7_end = self.conv_7x7_end(concat)  # B,50,F,T
        concat_final = torch.cat((conv_3x3_end, conv_5x5_end, conv_7x7_end), dim=1)  # B,150,F,T
        return concat_final

class Decoder(nn.Module):
    def __init__(self, conv_dim):
        super(Decoder, self).__init__()
        self.conv_3x3 = nn.Sequential(
                nn.Conv2d(in_channels=conv_dim, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                )
        self.conv_5x5 = nn.Sequential(
                nn.Conv2d(in_channels=conv_dim, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                )
        self.conv_7x7 = nn.Sequential(
                nn.Conv2d(in_channels=conv_dim, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU(),
                )
        self.conv_3x3_end = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=50, kernel_size=3, stride=1, padding=1), nn.ReLU()
                )
        self.conv_5x5_end = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=50, kernel_size=5, stride=1, padding=2), nn.ReLU()
                )
        self.conv_7x7_end = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=50, kernel_size=7, stride=1, padding=3), nn.ReLU()
                )
        self.conv_1x1_final = nn.Conv2d(in_channels=150, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)  # B,50,F,T
        conv_5x5 = self.conv_5x5(x)  # B,50,F,T
        conv_7x7 = self.conv_7x7(x)  # B,50,F,T
        concat = torch.cat((conv_3x3, conv_5x5, conv_7x7), dim=1)  # B,150,F,T
        conv_3x3_end = self.conv_3x3_end(concat)  # B,50,F,T
        conv_5x5_end = self.conv_5x5_end(concat)  # B,50,F,T
        conv_7x7_end = self.conv_7x7_end(concat)  # B,50,F,T
        concat_final = torch.cat((conv_3x3_end, conv_5x5_end, conv_7x7_end), dim=1)  # B,150,F,T
        output = self.conv_1x1_final(concat_final)  # B,1,F,T
        return output

