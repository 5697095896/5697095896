import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists, join
from os import makedirs
from convert_ceps import convert
from model import Encoder, CarrierDecoder, MsgDecoder
#from model_vision import Encoder, Decoder
import wandb
from solver import Solver
from collections import defaultdict

class SolverNMsgChainRNN(Solver):
    def __init__(self, config):
        super(SolverNMsgChainRNN, self).__init__(config)

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        n_messages = len(msg)
        losses_log = defaultdict(int)
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        carrier_reconst = [carrier] + carrier_reconst
        all_carrier_loss = 0
        all_msg_loss = 0
        for i in range(n_messages):
            carrier_loss = self.reconstruction_loss(carrier_reconst[i+1], carrier_reconst[i], type=self.loss_type)
            msg_loss = self.reconstruction_loss(msg_reconst[i], msg[i], type=self.loss_type)
            all_carrier_loss += carrier_loss
            all_msg_loss += msg_loss
            losses_log[f'carrier_{i+1}to{i}_loss'] = carrier_loss.item()
            losses_log[f'msg_{i}_loss'] = msg_loss.item()
        losses_log['carrier_loss'] = all_carrier_loss.item()
        losses_log['msg_loss'] = all_msg_loss.item()
        loss = self.lambda_carrier_loss * all_carrier_loss + self.lambda_msg_loss * all_msg_loss

        return loss, losses_log

    def forward(self, carrier, msg):
        assert type(carrier) == torch.Tensor and type(msg) == list
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        carrier_reconst_list = []
        msg_reconst_list = []
        carrier_reconst = carrier
        for i,msg in enumerate(msg):
            carrier_enc = self.enc_c(carrier_reconst)
            msg_enc = self.enc_m(msg)
            merged_enc = torch.cat((carrier_enc, msg_enc), dim=1)  # concat encodings on features axis
            carrier_reconst = self.dec_c(merged_enc)
            msg_reconst = self.dec_m(carrier_reconst)
            carrier_reconst_list.append(carrier_reconst)
            msg_reconst_list.append(msg_reconst)
        return carrier_reconst_list, msg_reconst_list

    def decode(self, carrier, n_msg=1):
        msgs = []
        for i in range(n_msg):
            msg = self.dec_m(carrier)
            msgs.append(msg)

            carrier_enc = self.enc_c(carrier_reconst)
            msg_enc = self.enc_m(msg)
            merged_enc = torch.cat((carrier_enc, msg_enc), dim=1)  # concat encodings on features axis
            carrier = self.dec_c(merged_enc)

        return msgs
