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
from utils import get_stft_error
import librosa
from hparams import *

class SolverNMsgCond(Solver):
    def __init__(self, config):
        super(SolverNMsgCond, self).__init__(config)
        print("==> running conditional solver!")
        self.dec_c_conv_dim = (self.n_messages+1) * (self.enc_conv_dim * (2 ** (self.enc_num_repeat-1)))

        # ------ create models ------
        self.enc_c = Encoder(conv_dim=1)
        self.enc_m = Encoder(conv_dim=1 + self.n_messages)
        self.dec_c = CarrierDecoder(conv_dim=self.dec_c_conv_dim,
                                    block_type=self.block_type)
        self.dec_m = MsgDecoder(conv_dim=self.dec_m_conv_dim + self.n_messages,
                                block_type=self.block_type)

        # ------ make parallel ------
        self.enc_c = nn.DataParallel(self.enc_c)
        self.enc_m = nn.DataParallel(self.enc_m)
        self.dec_m = nn.DataParallel(self.dec_m)
        self.dec_c = nn.DataParallel(self.dec_c)

        # ------ create optimizers ------
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=self.enc_c_lr)
        self.enc_m_opt = torch.optim.Adam(self.enc_m.parameters(), lr=self.enc_m_lr)
        self.dec_m_opt = torch.optim.Adam(self.dec_m.parameters(), lr=self.dec_m_lr)
        self.dec_c_opt = torch.optim.Adam(self.dec_c.parameters(), lr=self.dec_c_lr)

        # ------ send to cuda ------
        self.enc_c.to(self.device)
        self.enc_m.to(self.device)
        self.dec_m.to(self.device)
        self.dec_c.to(self.device)

        if self.load_ckpt_dir:
            self.load_models(self.load_ckpt_dir)

        print(self.enc_m)
        print(self.dec_c)
        print(self.dec_m)

    def save_models(self, suffix=''):
        print(f"==> saving model to: {self.ckpt_dir}\n==> suffix: {suffix}")
        makedirs(join(self.ckpt_dir, suffix), exist_ok=True)
        torch.save(self.enc_c.state_dict(), join(self.ckpt_dir, suffix, "enc_c.ckpt"))
        torch.save(self.enc_m.state_dict(), join(self.ckpt_dir, suffix, "enc_m.ckpt"))
        torch.save(self.dec_c.state_dict(), join(self.ckpt_dir, suffix, "dec_c.ckpt"))
        torch.save(self.dec_m.state_dict(), join(self.ckpt_dir, suffix, "dec_m.ckpt"))

    def load_models(self, ckpt_dir):
        self.enc_c.load_state_dict(torch.load(join(ckpt_dir, "enc_c.ckpt")))
        self.enc_m.load_state_dict(torch.load(join(ckpt_dir, "enc_m.ckpt")))
        self.dec_c.load_state_dict(torch.load(join(ckpt_dir, "dec_c.ckpt")))
        self.dec_m.load_state_dict(torch.load(join(ckpt_dir, "dec_m.ckpt")))
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
        batch_size = carrier.shape[0]
        carrier, carrier_phase, msg = carrier.to(self.device), carrier_phase.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        msg_encoded_list = []
        msg_reconst_list = []

        # encode carrier
        carrier_enc = self.enc_c(carrier)

        # encoder mesasges
        for i in range(self.n_messages):
            # create one-hot vectors for msg index
            cond = torch.tensor(()).new_full((batch_size,), i)
            cond = self.label2onehot(cond, self.n_messages).to(self.device)
            # concat conditioning vectors to input
            msg_i = self.concat_cond(msg[i], cond)
            msg_encoded_list.append(self.enc_m(msg_i))

        # merge encodings and reconstruct carrier
        msg_enc = torch.cat(msg_encoded_list, dim=1)
        merged_enc = torch.cat((carrier_enc, msg_enc), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)

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
        for i in range(self.n_messages):
            cond = torch.tensor(()).new_full((batch_size,), i)
            cond = self.label2onehot(cond, self.n_messages).to(self.device)
            msg_reconst = self.dec_m(self.concat_cond(carrier_reconst_tag, cond))
            msg_reconst_list.append(msg_reconst)

        return carrier_reconst, msg_reconst_list

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def concat_cond(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        return torch.cat([x, c], dim=1)
