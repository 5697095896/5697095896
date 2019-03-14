import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists, join
from os import makedirs
from dataloader import GCommandLoader, TimitLoader, YohoLoader
from convert_ceps import convert
from model import Encoder, CarrierDecoder, MsgDecoder
from model import Discriminator
from tqdm import tqdm
import wandb
from tensorboardX import SummaryWriter
from hparams import *
from stft.stft import STFT
import shutil
import json
import traceback


class Solver(object):
    def __init__(self, config):
        print(config)
        self.config = config
        # optimization hyperparams
        self.enc_c_lr = config.enc_c_lr
        self.enc_m_lr = config.enc_m_lr
        self.dec_c_lr = config.dec_c_lr
        self.dec_m_lr = config.dec_m_lr
        self.lambda_carrier_loss = config.lambda_carrier_loss
        self.lambda_msg_loss = config.lambda_msg_loss

        # training config
        self.num_iters = config.num_iters
        self.cur_iter = 0
        self.loss_type = config.loss_type
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.test_path = config.test_path
        self.batch_size = config.batch_size
        self.n_pairs = config.n_pairs
        self.n_messages = config.n_messages
        self.model_type = config.model_type
        self.dataset= config.dataset
        self.max_len = {'yoho': 101, 'timit': 101, 'gcommand': 101}[self.dataset]
        self.trim_start = {'yoho': int(1.5*8000), 'timit': int(0.2*16000), 'gcommand': 0}[self.dataset]
        self.num_samples = {'yoho': 16000, 'timit': 16000, 'gcommand': 16000}[self.dataset]
        self.carrier_detach = config.carrier_detach
        self.add_stft_noise = config.add_stft_noise
        self.adv = config.adv
        self.flip_msg = config.flip_msg
        self.block_type = config.block_type

        # model dimensions
        self.enc_conv_dim = 16
        self.enc_num_repeat = 3
        self.dec_c_conv_dim = self.enc_conv_dim * (2 ** self.enc_num_repeat)
        self.dec_c_num_repeat = self.enc_num_repeat
        self.dec_m_conv_dim = 1
        self.dec_m_num_repeat = 8 #4 #8

        self.num_workers = config.num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wandb = config.wandb
        self.tensorboard = config.tensorboard
        self.run_dir = config.run_dir
        self.ckpt_dir = join(self.run_dir, 'ckpt')
        self.load_ckpt_dir = config.load_ckpt
        self.save_model_every = config.save_model_every
        self.samples_dir = join(self.run_dir, 'samples')
        self.sample_every = config.sample_every
        self.plot_weights_every = config.plot_weights_every
        self.carrier_losses_npy = np.array([])
        self.avg_msg_losses_npy = np.array([])

        # logging
        self.print_every = 10
        if self.wandb:
            wandb.init()
            wandb.config.update(config, allow_val_change=True)
        if self.tensorboard:
            self.tensorboard = SummaryWriter(log_dir=self.run_dir, comment=self.run_dir.split('/')[-1])
        else: self.wandb = None

        self.create_dirs()
        self.load_data()
        self.build_models()
        torch.manual_seed(10)

        self.stft = STFT(N_FFT, HOP_LENGTH)
        self.stft.num_samples = self.num_samples

    def log_losses(self, losses):
        for loss_name, loss_value in losses.items():
            if self.tensorboard:
                self.tensorboard.add_scalar(loss_name, loss_value, self.cur_iter)
            if self.wandb:
                wandb.log({loss_name: loss_value}, step=self.cur_iter)
        self.carrier_losses_npy = np.append(self.carrier_losses_npy, [losses['carrier_loss']])
        self.avg_msg_losses_npy = np.append(self.avg_msg_losses_npy, [losses['avg_msg_loss']])
        np.save(join(self.run_dir, 'carrier_losses'), self.carrier_losses_npy)
        np.save(join(self.run_dir, 'avg_msg_losses'), self.avg_msg_losses_npy)

    def create_dirs(self):
        if exists(self.run_dir):
            shutil.rmtree(self.run_dir, ignore_errors=True)
        makedirs(self.run_dir, exist_ok=True)
        makedirs(self.ckpt_dir, exist_ok=True)
        makedirs(join(self.ckpt_dir, 'best_c_recon'), exist_ok=True)
        makedirs(join(self.ckpt_dir, 'best_m_recon'), exist_ok=True)
        makedirs(self.samples_dir, exist_ok=True)
        with open(join(self.run_dir, 'hparams.txt'), 'w') as f:
            f.write(json.dumps(vars(self.config)))
        print("==> created dirs")

    def load_data(self):
        loader = {'gcommand': GCommandLoader,
                  'yoho': YohoLoader,
                  'timit': TimitLoader}[self.dataset]

        train = loader(self.train_path, n_messages=self.n_messages, n_pairs=self.n_pairs, max_len=self.max_len, flip_msg=self.flip_msg)
        self.train_loader = torch.utils.data.DataLoader(train,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.num_workers)
        val = loader(self.val_path, n_messages=self.n_messages, n_pairs=1000, max_len=self.max_len, flip_msg=self.flip_msg)
        self.val_loader = torch.utils.data.DataLoader(val,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.num_workers)
        test = loader(self.test_path, n_messages=self.n_messages, n_pairs=1000, max_len=self.max_len, flip_msg=self.flip_msg)
        self.test_loader = torch.utils.data.DataLoader(test,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=0)
        print(f"==> loaded train ({len(train)}), val ({len(val)}), test ({len(test)})")

    def build_models(self):
        self.enc_c = Encoder(conv_dim=self.enc_conv_dim, num_repeat=self.enc_num_repeat, block_type=self.block_type)
        self.enc_m = Encoder(conv_dim=self.enc_conv_dim, num_repeat=self.enc_num_repeat, block_type=self.block_type)
        self.dec_c = CarrierDecoder(conv_dim=self.dec_c_conv_dim, num_repeat=self.dec_c_num_repeat, block_type=self.block_type)
        self.dec_m = MsgDecoder(conv_dim=self.dec_m_conv_dim, num_repeat=self.dec_m_num_repeat, block_type=self.block_type)

        #self.enc_c = Encoder()
        #self.enc_m = Encoder()
        #self.dec_c = Decoder(conv_dim=300)
        #self.dec_m = Decoder(conv_dim=1)

        self.enc_c = nn.DataParallel(self.enc_c)
        self.enc_m = nn.DataParallel(self.enc_m)
        self.dec_c = nn.DataParallel(self.dec_c)
        self.dec_m = nn.DataParallel(self.dec_m)

        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), self.enc_c_lr)
        self.enc_m_opt = torch.optim.Adam(self.enc_m.parameters(), self.enc_m_lr)
        self.dec_c_opt = torch.optim.Adam(self.dec_c.parameters(), self.dec_c_lr)
        self.dec_m_opt = torch.optim.Adam(self.dec_m.parameters(), self.dec_m_lr)

        self.enc_c.to(self.device)
        self.enc_m.to(self.device)
        self.dec_c.to(self.device)
        self.dec_m.to(self.device)

        self.discriminator = Discriminator()
        self.discriminator = nn.DataParallel(self.discriminator)
        self.discriminator.to(self.device)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

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

    def step(self):
        self.enc_c_opt.step()
        self.enc_m_opt.step()
        self.dec_c_opt.step()
        self.dec_m_opt.step()

    def reset_grad(self):
        self.enc_c_opt.zero_grad()
        self.enc_m_opt.zero_grad()
        self.dec_c_opt.zero_grad()
        self.dec_m_opt.zero_grad()

    def reconstruction_loss(self, input, target, type='mse'):
        if type == 'mse':
            loss = F.mse_loss(input, target)
        elif type == 'abs':
            loss = F.l1_loss(input, target)
        elif type == 'mix':
            loss = 0.5 * F.mse_loss(input, target) + 0.5 * F.l1_loss(input, target)
        elif type == 'sched':
            if self.cur_iter < 1000:
                loss = F.mse_loss(input, target)
            else:
                loss = F.l1_loss(input, target)
        else:
            print("==> unsupported loss function! reverting to MSE...")
            loss = F.mse_loss(input, target)
        return loss

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        # IMPLEMENT THIS IN SUB-CLASS
        pass

    def forward(self, carrier, msg):
        # IMPLEMENT THIS IN SUB-CLASS
        pass

    def decode(self, carrier):
        # IMPLEMENT THIS IN SUB-CLASS
        pass

    def train(self):
        self.mode = 'train'
        self.enc_c.train()
        self.enc_m.train()
        self.dec_c.train()
        self.discriminator.train()
        if type(self.dec_m) == list:
            for dec in self.dec_m:
                dec.train()
        else:
            self.dec_m.train()
        # start of training loop
        print("==> start training...")
        for i in range(self.num_iters):
            self.cur_iter += 1

            try:
                carrier, carrier_phase, msg = next(data_iter)
            except Exception as e:
                print(f"==> Error: {e}")
                traceback.print_exc()
                data_iter = iter(self.train_loader)
                carrier, carrier_phase, msg = next(data_iter)

            batch_size, _, h, w = carrier.shape

            # feedforward and suffer loss
            carrier_reconst, msg_reconst = self.forward(carrier, carrier_phase, msg)
            if torch.isnan(carrier_reconst).sum().item() > 0: continue
            loss, losses_log = self.incur_loss(carrier, carrier_reconst, msg, msg_reconst)

            if self.adv:
                g_target_label_encoded = torch.full((batch_size, 1), 1, device=self.device)
                d_on_encoded_for_enc = self.discriminator(carrier_reconst)
                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                loss += g_loss_adv

            self.reset_grad()
            loss.backward()
            self.step()

            # ------ train discriminator ------
            if self.adv:
                self.discriminator_opt.zero_grad()
                d_target_label_cover = torch.full((batch_size, 1), 1, device=self.device)
                d_on_cover = self.discriminator(carrier)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)
                d_loss_on_cover.backward()

                d_target_label_encoded = torch.full((batch_size, 1), 0, device=self.device)
                d_on_encoded = self.discriminator(carrier_reconst.detach())
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                d_loss_on_encoded.backward()
                self.discriminator_opt.step()

                losses_log['d_real'] = d_loss_on_cover
                losses_log['d_fake'] = d_loss_on_encoded
                losses_log['g_fake'] = g_loss_adv

            # log stuff
            if i % self.print_every == 0:
                log = f"[{i}/{self.num_iters}]"
                for loss_name, loss_value in losses_log.items():
                    log += f", {loss_name}: {loss_value:.4f}"
                print(log)

            self.log_losses(losses_log)

            # save models
            if self.save_model_every and (i+1) % self.save_model_every == 0:
                self.save_models(suffix=str(i+1))

            if self.sample_every and (i+1) % self.sample_every == 0:
                self.sample_examples()

        print("==> finished training!")

    def test(self, data='test'):
        self.mode = 'test'
        self.enc_c.eval()
        self.enc_m.eval()
        self.dec_c.eval()
        if type(self.dec_m) == list:
            for dec in self.dec_m:
                dec.eval()
        else:
            self.dec_m.eval()
        with torch.no_grad():
            avg_carrier_loss, avg_msg_loss = 0, 0

            data = self.test_loader if data == 'test' else self.val_loader
            # start of training loop
            print("==> start testing...")
            for carrier, carrier_phase, msg in tqdm(data):
                # feedforward and suffer loss
                carrier_reconst, msg_reconst = self.forward(carrier, carrier_phase, msg)
                loss, losses_log = self.incur_loss(carrier, carrier_reconst, msg, msg_reconst)
                avg_carrier_loss += losses_log['carrier_loss']
                avg_msg_loss += losses_log['avg_msg_loss']

            print("==> finished testing!")
            print(f"==> carrier loss: {avg_carrier_loss/len(data)}")
            print(f"==> message loss: {avg_msg_loss/len(data)}")

    def sample_examples(self, n_examples=20):
        print(f"==> generating {n_examples} examples...")
        examples_dir = self.samples_dir + "_final"
        makedirs(examples_dir, exist_ok=True)
        for i in tqdm(range(n_examples)):
            examples_subdir = join(examples_dir, f'{i}')
            makedirs(examples_subdir, exist_ok=True)
            carrier_path, msg_path = self.train_loader.dataset.spect_pairs[i]
            convert(self, carrier_path, msg_path, examples_subdir, i, self.max_len, self.trim_start, self.flip_msg)
        print("==> done")
