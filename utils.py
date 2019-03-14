import torch
import librosa
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import time
from hparams import *


def griffin_lim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window)

    return inverse

def get_stft_error(mag_batch, phase_batch=None, device='cuda'):
    result = []

    if type(mag_batch) == torch.Tensor:
        mag_batch = mag_batch.detach().cpu().numpy()

    for i, mag in enumerate(mag_batch):
        mag = mag[0]

        if not phase_batch:
            y = griffin_lim(mag, n_fft=N_FFT, hop_length=HOP_LENGTH)
        else:
            phase = phase_batch[i]
            y = librosa.istft(mag*phase, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

        new_mag, new_phase = librosa.magphase(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
        result.append(new_mag - mag)

    result = np.stack(result)
    result = torch.from_numpy(result).unsqueeze(1).to(device)
    return result

def get_stft_error(mag_batch, phase_batch=None, device='cuda'):
    start_time = time.time()
    def get_noise(mag, phase=None):
        if phase is None:
            y = griffin_lim(mag, n_fft=N_FFT, hop_length=HOP_LENGTH)
        else:
            phase_real, phase_imag = phase
            phase_real, phase_imag = phase_real.cpu().numpy(), phase_imag.cpu().numpy()
            phase = phase_real + 1j*phase_imag
            y = librosa.istft(mag*phase, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

        new_mag, new_phase = librosa.magphase(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
        return new_mag - mag

    if phase_batch is not None:
        phase_real, phase_imag = phase_batch
        phase_real = phase_real.squeeze(1)
        phase_imag = phase_imag.squeeze(1)

    mag_batch = mag_batch.squeeze(1)
    if type(mag_batch) == torch.Tensor:
        mag_batch = mag_batch.detach().cpu().numpy()

    results = Parallel(n_jobs=mag_batch.shape[0])(delayed(get_noise)(mag, (pr,pi)) for i, (mag, pr, pi) in enumerate(zip(mag_batch, phase_real, phase_imag)))
    results = np.stack(results)
    results = torch.from_numpy(results).unsqueeze(1).to(device)
    # print("--- %s seconds ---" % (time.time() - start_time))
    return results
