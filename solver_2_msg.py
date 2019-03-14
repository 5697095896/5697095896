import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists, join
from os import makedirs
from gcommand_loader_2_msg import GCommandLoader
from convert_ceps import convert
from model import Encoder, CarrierDecoder, MsgDecoder
#from model_vision import Encoder, Decoder
import wandb
from solver import Solver

class Solver2Msg(Solver):
    def __init__(self, config):
        super(Solver2Msg, self).__init__(config)

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        carrier, msg = carrier.to(self.device), msg.to(self.device)
        carrier_loss = self.reconstruction_loss(carrier_reconst, carrier, type=self.loss_type)
        msg_loss = self.reconstruction_loss(msg_reconst, msg, type=self.loss_type)
        loss = self.lambda_carrier_loss * carrier_loss + self.lambda_msg_loss * msg_loss

        # logging
        losses_log = {}
        losses_log['carrier_loss'] = carrier_loss.item()
        losses_log['msg_loss'] = msg_loss.item()
        losses_log['overall_loss'] = loss.item()

        return loss, losses_log

    def forward(self, carrier, msg, add_noise=False):
        assert type(carrier) == torch.Tensor and type(msg) == torch.Tensor
        carrier, msg = carrier.to(self.device), msg.to(self.device)
        carrier_enc = self.enc_c(carrier)
        msg_enc = self.enc_m(msg)
        merged_enc = torch.cat((carrier_enc, msg_enc), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)
        if add_noise:
            noise = carrier_reconst.data.new(carrier_reconst.size()).normal_(0.0, 0.1)
            carrier_reconst += noise
        msg_reconst = self.dec_m(carrier_reconst)
        return carrier_reconst, msg_reconst
