import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists, join
from os import makedirs
from convert_ceps import convert
# from model import Encoder, CarrierDecoder, MsgDecoder
from deepsteg_model import Encoder, CarrierDecoder, MsgDecoder
from model_vision import Encoder, Decoder
import wandb
from solver import Solver
from collections import defaultdict
from utils import get_stft_error
import librosa
from hparams import *
import time

class SolverNMsgMultipleDecoders(Solver):
    def __init__(self, config):
        super(SolverNMsgMultipleDecoders, self).__init__(config)
        print("==> running multiple decoders solver!")
        self.dec_c_conv_dim = (self.n_messages+1) * (self.enc_conv_dim * (2 ** (self.enc_num_repeat-1)))

        # ------ create models ------
        self.dec_c = CarrierDecoder(conv_dim=self.dec_c_conv_dim,
                                    block_type=self.block_type)
        self.dec_m = [MsgDecoder(conv_dim=self.dec_m_conv_dim,
                                 block_type=self.block_type) for _ in range(self.n_messages)]

        # ------ make parallel ------
        self.dec_c = nn.DataParallel(self.dec_c)
        self.dec_m = [nn.DataParallel(m) for m in self.dec_m]

        # ------ create optimizers ------
        self.dec_c_opt = torch.optim.Adam(self.dec_c.parameters(), lr=self.dec_c_lr)
        self.dec_m_params = []
        for i in range(len(self.dec_m)):
            self.dec_m_params += list(self.dec_m[i].parameters())
        self.dec_m_opt = torch.optim.Adam(self.dec_m_params, lr=self.dec_m_lr)

        # ------ send to cuda ------
        self.dec_c.to(self.device)
        self.dec_m = [m.to(self.device) for m in self.dec_m]

        if self.load_ckpt_dir:
            self.load_models(self.load_ckpt_dir)

        print(self.enc_c)
        print(self.dec_c)
        print(self.dec_m)

    def save_models(self, suffix=''):
        print(f"==> saving model to: {self.ckpt_dir}\n==> suffix: {suffix}")
        makedirs(join(self.ckpt_dir, suffix), exist_ok=True)
        torch.save(self.enc_c.state_dict(), join(self.ckpt_dir, suffix, "enc_c.ckpt"))
        torch.save(self.enc_m.state_dict(), join(self.ckpt_dir, suffix, "enc_m.ckpt"))
        torch.save(self.dec_c.state_dict(), join(self.ckpt_dir, suffix, "dec_c.ckpt"))
        for i,m in enumerate(self.dec_m):
            torch.save(m.state_dict(), join(self.ckpt_dir, suffix, f"dec_m_{i}.ckpt"))

    def load_models(self, ckpt_dir):
        self.enc_c.load_state_dict(torch.load(join(ckpt_dir, "enc_c.ckpt")))
        self.enc_m.load_state_dict(torch.load(join(ckpt_dir, "enc_m.ckpt")))
        self.dec_c.load_state_dict(torch.load(join(ckpt_dir, "dec_c.ckpt")))
        for i,m in enumerate(self.dec_m):
            m.load_state_dict(torch.load(join(ckpt_dir, f"dec_m_{i}.ckpt")))
        print("==> loaded models")

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        n_messages = len(msg)
        losses_log = defaultdict(int)
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        all_msg_loss = 0
        carrier_loss = self.reconstruction_loss(carrier_reconst, carrier, type=self.loss_type)
        for i in range(n_messages):
            msg_loss = self.reconstruction_loss(msg_reconst[i], msg[i], type=self.loss_type)
            all_msg_loss += msg_loss
        losses_log['carrier_loss'] = carrier_loss.item()
        losses_log['avg_msg_loss'] = all_msg_loss.item() / self.n_messages
        loss = self.lambda_carrier_loss * carrier_loss + self.lambda_msg_loss * all_msg_loss

        return loss, losses_log

    def forward(self, carrier, carrier_phase, msg):
        assert type(carrier) == torch.Tensor and type(msg) == list
        carrier, carrier_phase, msg = carrier.to(self.device), carrier_phase.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        msg_reconst_list = []

        # create embedded carrier
        carrier_enc = self.enc_c(carrier)  # encode the carrier
        msg_enc = [self.enc_m(m) for m in msg]  # encode each msg_i
        msg_enc = torch.cat(msg_enc, dim=1)  # concat all msg_i into single tensor
        merged_enc = torch.cat((carrier_enc, msg_enc), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)  # decode carrier [B,1,161,101]

        if self.carrier_detach != -1 and self.cur_iter > self.carrier_detach:
            carrier_reconst = carrier_reconst.detach()

        # add stft noise to carrier
        if (self.add_stft_noise != -1 and self.cur_iter > self.add_stft_noise) or self.mode == 'test':
            self.stft.to(self.device)
            y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
            carrier_reconst_tag, _ = self.stft.transform(y.squeeze(1))
            carrier_reconst_tag = carrier_reconst_tag[:,:,:self.max_len].unsqueeze(1)
            self.stft.to('cpu')
        else:
            carrier_reconst_tag = carrier_reconst

        # decode messages from carrier
        for i in range(len(msg)):  # decode each msg_i using decoder_m_i
            msg_reconst = self.dec_m[i](carrier_reconst_tag)
            msg_reconst_list.append(msg_reconst)

        return carrier_reconst, msg_reconst_list
