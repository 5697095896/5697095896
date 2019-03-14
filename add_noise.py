from os.path import join
import os
import numpy as np
import soundfile as sf
import random
from boltons import fileutils
import argparse

def inject_noise_sample(data_path, noise_path, noise_level):
    data, sr = sf.read(data_path)
    noise, sr = sf.read(noise_path)
    noise_len = len(noise)
    data_len = len(data)

    if noise_len > data_len:
         diff = noise_len-data_len
         noise_start = random.randint(0, diff - 1)
         noise_end = noise_start + data_len
         noise = noise[noise_start:noise_end]

    noise_len = len(noise)
    data_len = len(data)

    start = int(np.random.rand() * (data_len - noise_len))
    end = int(start + noise_len)
    noise_energy = np.sqrt(noise.dot(noise) / noise.size)
    data_energy = np.sqrt(data.dot(data) / data.size)
    data[start:end] += noise_level * noise* data_energy / noise_energy
    return data, sr

def inject_noise_sample_write(data_path, noise_path, target_path, noise_level):
    data, sr = inject_noise_sample(data_path, noise_path, noise_level)
    sf.write(target_path, data, sr)

def inject_noise_folder(wav_folder, noise_levels, n_items):
    if type(noise_levels) == float:
        noise_levels = [noise_levels]
    trg_dir = join(wav_folder, 'out')
    os.makedirs(trg_dir, exist_ok=True)
    wavs = list(fileutils.iter_find_files(wav_folder, "*.wav"))
    for noise_level in noise_levels:
        for i in range(n_items):
            w1, w2 = random.sample(wavs, 2)
            inject_noise_sample(w1, w2, join(trg_dir, f"{i}_{noise_level}_noise.wav"), noise_level)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wav_folder', type=str, help='')
    parser.add_argument('--noise_levels', type=str, help='')
    parser.add_argument('--n_items', type=int, help='')
    args = parser.parse_args()
    args.noise_levels = list(map(float, args.noise_levels.split(',')))
    inject_noise_folder(args.wav_folder, args.noise_levels, args.n_items)
