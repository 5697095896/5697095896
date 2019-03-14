import soundfile
import librosa
import torch
import numpy as np
import argparse
from os.path import join, basename
from os import makedirs
from hparams import *
from stft.stft import STFT
from dataloader import spect_loader
import soundfile as sf

def wav_to_magphase(wav, n_fft, win_length, hop_length, window):
    D = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    spect, phase = librosa.magphase(D)
    return spect, phase

def pad(spect, max_len):
    pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
    spect = np.hstack((spect, pad))
    return spect

def convert(solver, carrier_wav_path, msg_wav_paths, trg_dir, epoch, max_len, trim_start, flip_msg):
    solver.mode = 'test'

    _, sr = sf.read(carrier_wav_path)
    carrier_basename = basename(carrier_wav_path).split(".")[0]
    msg_basenames = [basename(msg_wav_path).split(".")[0] for msg_wav_path in msg_wav_paths]

    spect_carrier, phase_carrier = spect_loader(carrier_wav_path, max_len, trim_start, return_phase=True, inject_noise=True)
    spect_carrier, phase_carrier = spect_carrier.unsqueeze(0), phase_carrier.unsqueeze(0)
    magphase_msg = [spect_loader(path, max_len, trim_start, return_phase=True, inject_noise=False) for path in msg_wav_paths]
    spects_msg, phases_msg = [D[0].unsqueeze(0) for D in magphase_msg], [D[1].unsqueeze(0) for D in magphase_msg]

    if flip_msg:
        spects_msg= [torch.flip(m, [2,3]) for m in spects_msg]

    spect_carrier = spect_carrier.to('cuda')
    spects_msg = [spect_msg.to('cuda') for spect_msg in spects_msg]
    spect_carrier_reconst, spects_msg_reconst = solver.forward(spect_carrier, phase_carrier, spects_msg)
    spect_carrier_reconst = spect_carrier_reconst.cpu().squeeze(0)
    spects_msg_reconst = [spect_msg_reconst.cpu().squeeze(0) for spect_msg_reconst in spects_msg_reconst]

    if flip_msg:
        spects_msg_reconst = [torch.flip(m, [1,2]) for m in spects_msg_reconst]

    stft = STFT(N_FFT, HOP_LENGTH)
    out_carrier = stft.inverse(spect_carrier_reconst, phase_carrier.squeeze(0)).squeeze(0).squeeze(0).detach().numpy()
    orig_out_carrier = stft.inverse(spect_carrier.cpu().squeeze(0), phase_carrier.squeeze(0)).squeeze(0).squeeze(0).detach().numpy()

    if flip_msg:
        spects_msg= [torch.flip(m, [2,3]) for m in spects_msg]

    outs_msg = [stft.inverse(spect_msg_reconst, phase_msg.squeeze(0)).squeeze(0).squeeze(0).detach().numpy() for spect_msg_reconst, phase_msg in zip(spects_msg_reconst, phases_msg)]
    orig_outs_msg = [stft.inverse(spect_msg.cpu().squeeze(0), phase_msg.squeeze(0)).squeeze(0).squeeze(0).detach().numpy() for spect_msg, phase_msg in zip(spects_msg, phases_msg)]

    librosa.output.write_wav(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_embedded.wav'), out_carrier, sr)
    librosa.output.write_wav(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_orig.wav'), orig_out_carrier, sr)
    for i in range(len(outs_msg)):
        librosa.output.write_wav(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered.wav'), outs_msg[i], sr)
        librosa.output.write_wav(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_orig.wav'), orig_outs_msg[i], sr)
    solver.mode = 'train'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--g', type=str, help='---')
    parser.add_argument('--c', type=int, help='<`5:---`>')
    parser.add_argument('--c_dim', type=int, help='<`5:---`>')
    parser.add_argument('--src', type=str, help='<`5:---`>')
    parser.add_argument('--trg', type=str, help='<`5:---`>')
    args = parser.parse_args()

    G = Generator(num_speakers=args.c_dim, repeat_num=5).to('cuda')
    G.load_state_dict(torch.load(args.g, map_location=lambda storage, loc: storage))
    c = torch.zeros(1, args.c_dim)
    c[0,args.c] = 1
    c = c.to('cuda')
    convert(G, args.src, args.trg, c, "cuda", 16000)
