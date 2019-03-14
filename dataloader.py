from boltons import fileutils
import os
import os.path
from collections import defaultdict
import librosa
import numpy as np
import torch
import torch.utils.data as data
import random
import soundfile
from hparams import *
from stft.stft import STFT
from add_noise import inject_noise_sample

def is_audio_file(filename):
    AUDIO_EXTENSIONS = ['.wav', '.WAV']
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def spect_loader(path, max_len, trim_start, return_phase=False, inject_noise=False):
    if inject_noise:
        y, sr = inject_noise_sample(path, '/data/felix/resturant_mono.wav', .5)
    else:
        y, sr = soundfile.read(path)
    y = y[trim_start:]

    n_fft = N_FFT
    win_length = N_FFT
    hop_length = int(sr * WINDOW_STRIDE)

    # STFT
    stft = STFT(N_FFT, HOP_LENGTH)
    y = torch.FloatTensor(y).unsqueeze(0)
    spect, phase = stft.transform(y)
    spect, phase = spect.squeeze(0).cpu().numpy(), phase.squeeze(0).cpu().numpy()

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
        phase = np.hstack((phase, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
        phase = phase[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    phase = np.resize(phase, (1, phase.shape[0], phase.shape[1]))
    spect = torch.FloatTensor(spect)
    phase = torch.FloatTensor(phase)

    if return_phase:
        return spect, phase
    return spect

class BaseLoader(data.Dataset):
    def __init__(self, root, n_messages=1, n_pairs=100000, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, trim_start=0, flip_msg=False):
        self.spect_pairs = self.make_pairs_dataset(root, n_messages, n_pairs)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.trim_start = int(trim_start)
        self.flip_msg = flip_msg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        carrier_file, msg_files = self.spect_pairs[index]
        carrier_spect, carrier_phase = self.loader(carrier_file, self.max_len, self.trim_start, return_phase=True, inject_noise=True)
        msg_spects = [self.loader(msg_file, self.max_len, self.trim_start, return_phase=False, inject_noise=False) for msg_file in msg_files]

        if self.transform is not None:
            carrier_spect = self.transform(carrier_spect)
            carrier_phase= self.transform(carrier_phase)
            msg_spects = [self.transform(msg_spect) for msg_spect in msg_spects]

        if self.flip_msg:
            msg_spects = [torch.flip(m, [1,2]) for m in msg_spects]

        return carrier_spect, carrier_phase, msg_spects

    def __len__(self):
        return len(self.spect_pairs)

class YohoLoader(BaseLoader):
    def __init__(self, root, n_messages=1, n_pairs=100000, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, trim_start=1.5*8000, flip_msg=False):
        super(YohoLoader, self).__init__(root,
                                          n_messages,
                                          n_pairs,
                                          transform,
                                          target_transform,
                                          window_size,
                                          window_stride,
                                          window_type,
                                          normalize,
                                          max_len,
                                          trim_start,
                                          flip_msg)

    def make_pairs_dataset(self, path, n_hidden_messages, n_pairs):
        pairs = []
        files_by_speaker = defaultdict(list)
        unfiltered_wav_files = list(fileutils.iter_find_files(path, "*.wav"))
        wav_files = []
        for wav in unfiltered_wav_files:
            try:
                if soundfile.read(wav)[0].shape[0] > 3*8000: wav_files.append(wav)
            except:
                pass

        for wav in wav_files:
            speaker = int(wav.split('/')[6])
            files_by_speaker[speaker].append(wav)

        for i in range(n_pairs):
            speaker = random.sample(files_by_speaker.keys(), 1)[0]
            sampled_files = random.sample(files_by_speaker[speaker], 1+n_hidden_messages)
            carrier_file, hidden_message_files = sampled_files[0], sampled_files[1:]
            pairs.append((carrier_file, hidden_message_files))

        return pairs

class TimitLoader(BaseLoader):
    def __init__(self, root, n_messages=1, n_pairs=100000, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, trim_start=0.1*16000, flip_msg=False):
        super(TimitLoader, self).__init__(root,
                                          n_messages,
                                          n_pairs,
                                          transform,
                                          target_transform,
                                          window_size,
                                          window_stride,
                                          window_type,
                                          normalize,
                                          max_len,
                                          trim_start,
                                          flip_msg)

    def make_pairs_dataset(self, path, n_hidden_messages, n_pairs):
        pairs = []
        wav_files = list(fileutils.iter_find_files(path, "*.flac"))
        fn_files = list(fileutils.iter_find_files(path, "*.fn"))
        assert len(fn_files)==len(wav_files)

        def get_id(path):
            with open(path,'r') as f:
                return f.readlines()[0].split('/')[2][1:]
        fn_files = list(map(get_id, fn_files))

        files_by_speaker = defaultdict(list)
        for speaker, wav in zip(fn_files, wav_files):
            files_by_speaker[speaker].append(wav)

        for i in range(n_pairs):
            speaker = random.sample(files_by_speaker.keys(), 1)[0]
            sampled_files = random.sample(files_by_speaker[speaker], 1+n_hidden_messages)
            carrier_file, hidden_message_files = sampled_files[0], sampled_files[1:]
            pairs.append((carrier_file, hidden_message_files))

        return pairs

class GCommandLoader(BaseLoader):
    def __init__(self, root, n_messages=1, n_pairs=100000, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, trim_start=0.0, flip_msg=False):
        super(GCommandLoader, self).__init__(root,
                                          n_messages,
                                          n_pairs,
                                          transform,
                                          target_transform,
                                          window_size,
                                          window_stride,
                                          window_type,
                                          normalize,
                                          max_len,
                                          trim_start,
                                          flip_msg)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        spects = []
        idx_by_class = defaultdict(list)
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_audio_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        idx_by_class[class_to_idx[target]].append(path)
                        spects.append(item)
        return spects, idx_by_class

    def make_pairs_dataset(self, path, n_hidden_messages, n_pairs):
        classes, class_to_idx = self.find_classes(path)
        spects, idx_by_class = self.make_dataset(path, class_to_idx)

        pairs = []
        classes = idx_by_class.keys()
        for i in range(n_pairs):
            samples_classes = random.sample(classes, 1+n_hidden_messages)
            # sample 2 different classes
            carrier_class = samples_classes[0]
            hidden_message_classes = samples_classes[1:]

            # sample 2 different files from the above classes
            carrier_file = random.sample(idx_by_class[carrier_class], 1)[0]
            hidden_message_files = [random.sample(idx_by_class[c], 1)[0] for c in hidden_message_classes]

            pairs.append((carrier_file, hidden_message_files))
        return pairs
