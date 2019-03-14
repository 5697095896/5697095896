import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists, join
from os import makedirs
from convert_ceps import convert
# from model import Encoder, CarrierDecoder, MsgDecoder
from deepsteg_model import Encoder, CarrierDecoder, MsgDecoder
#from model_vision import Encoder, Decoder
import wandb
from solver import Solver
from collections import defaultdict

class SolverNMsgChain(Solver):
    def __init__(self, config):
        super(SolverNMsgChain, self).__init__(config)
        self.dec_c_conv_dim = 1 + (self.enc_conv_dim * (2 ** (self.enc_num_repeat-1)))

        # ------ create models ------
        self.enc_c = Encoder(block_type=self.block_type)
        self.dec_c = CarrierDecoder(conv_dim=self.dec_c_conv_dim,
                                    block_type=self.block_type)
        self.dec_m = MsgDecoder(conv_dim=self.dec_m_conv_dim,
                                 block_type=self.block_type)

        # ------ make parallel ------
        self.enc_c = nn.DataParallel(self.enc_c)
        self.dec_c = nn.DataParallel(self.dec_c)
        self.dec_m = nn.DataParallel(self.dec_m)

        # ------ create optimizers ------
        self.dec_m_opt = torch.optim.Adam(self.dec_m.parameters(), lr=self.enc_c_lr)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=self.enc_c_lr)
        self.dec_c_opt = torch.optim.Adam(self.dec_c.parameters(), lr=self.dec_c_lr)
        self.dec_m_opt = torch.optim.Adam(self.dec_m.parameters(), lr=self.dec_m_lr)

        # ------ send to cuda ------
        self.enc_c.to(self.device)
        self.dec_c.to(self.device)
        self.dec_m.to(self.device)

        if self.load_ckpt_dir:
            self.load_models(self.load_ckpt_dir)

        print(self.enc_c)
        print(self.dec_c)
        print(self.dec_m)

    def load_models(self, ckpt_dir):
        self.enc_c.load_state_dict(torch.load(join(ckpt_dir, "enc_c.ckpt")))
        self.enc_m.load_state_dict(torch.load(join(ckpt_dir, "enc_m.ckpt")))
        self.dec_c.load_state_dict(torch.load(join(ckpt_dir, "dec_c.ckpt")))
        self.dec_m.load_state_dict(torch.load(join(ckpt_dir, "dec_m.ckpt")))
        print("==> loaded models")

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        losses_log = defaultdict(int)
        losses_log['avg_msg_loss'] = []

        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        carrier_loss = self.reconstruction_loss(carrier_reconst, carrier, type=self.loss_type)
        msg_loss = self.reconstruction_loss(msg_reconst, msg[0], type=self.loss_type)
        loss = self.lambda_carrier_loss * carrier_loss + self.lambda_msg_loss * msg_loss
        losses_log['chain_at_1'] = msg_loss.item()
        losses_log['avg_msg_loss'].append(msg_loss.item())

        ###
        three_msg = msg[:3]
        c = self.encode(carrier, three_msg)
        m = self.decode(c, n_msg=len(three_msg))
        chain_at_3 = self.reconstruction_loss(m[-1], msg[0])
        losses_log['chain_at_3'] = chain_at_3.item()
        losses_log['avg_msg_loss'].append(chain_at_3.item())
        loss += chain_at_3

        five_msg = msg[:5]
        c = self.encode(carrier, five_msg)
        m = self.decode(c, n_msg=len(five_msg))
        chain_at_5 = self.reconstruction_loss(m[-1], msg[0])
        losses_log['chain_at_5'] = chain_at_5.item()
        losses_log['avg_msg_loss'].append(chain_at_5.item())
        loss += chain_at_5
        ###

        # logging
        losses_log['carrier_loss'] = carrier_loss.item()
        losses_log['avg_msg_loss'] = sum(losses_log['avg_msg_loss']) / len(losses_log['avg_msg_loss'])

        return loss, losses_log

    def forward(self, carrier, carrier_phase, msg):
        assert type(carrier) == torch.Tensor and type(msg) == list
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        msg = msg[0]

        carrier_enc = self.enc_c(carrier)
        merged_enc = torch.cat((carrier_enc, msg), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)

        self.stft.to(self.device)
        y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
        carrier_reconst_tag, _ = self.stft.transform(y.squeeze(1))
        carrier_reconst_tag = carrier_reconst_tag[:,:,:self.max_len].unsqueeze(1)
        self.stft.to('cpu')

        msg_reconst = self.dec_m(carrier_reconst_tag)
        return carrier_reconst, msg_reconst

    def encode(self, carrier, msg):
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]

        carrier_enc = self.enc_c(carrier)
        merged_enc = torch.cat((carrier_enc, msg[0]), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)

        self.stft.to(self.device)
        y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
        carrier_reconst, _ = self.stft.transform(y.squeeze(1))
        carrier_reconst= carrier_reconst[:,:,:self.max_len].unsqueeze(1)
        self.stft.to('cpu')

        for i, msg in enumerate(msg[1:]):
            carrier_reconst, msg = msg, carrier_reconst
            carrier_enc = self.enc_c(carrier_reconst)
            merged_enc = torch.cat((carrier_enc, msg), dim=1)  # concat encodings on features axis
            carrier_reconst = self.dec_c(merged_enc)

            self.stft.to(self.device)
            y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
            carrier_reconst, _ = self.stft.transform(y.squeeze(1))
            carrier_reconst = carrier_reconst[:,:,:self.max_len].unsqueeze(1)
            self.stft.to('cpu')

        return carrier_reconst

    def decode(self, carrier, n_msg=1):
        msgs = []
        carrier = carrier.to(self.device)
        for i in range(n_msg):
            msg = self.dec_m(carrier)
            msgs.append(msg)
            carrier = msg

        return msgs
