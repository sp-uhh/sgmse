#!/usr/env/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import os
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve, stft
import glob
import itertools, operator

def obtain_noise_file(noise_dir, i_sample, channels, dataset, sample_rate, len_speech):

    nb_samples = len(os.listdir(noise_dir))

    if dataset == "wham":
        noise_sample, noise_sr = sf.read(os.path.join(noise_dir, os.listdir(noise_dir)[i_sample%nb_samples]))

        if noise_sr != sample_rate: #Resample
            noise_sample = resample(noise_sample, noise_sr, sample_rate)
        if channels == 1:
            noise_sample = noise_sample[0, :].unsqueeze(0)
            
    elif dataset == "chime":
        noise_types = ["CAF", "PED", "STR", "BUS"]
        noise_type = noise_types[np.random.randint(len(noise_types))]
        noise_candidates = glob.glob(os.path.join(noise_dir, f"*_{noise_type}.CH1.wav"))
        noise_sample_basename = noise_candidates[np.random.randint(len(noise_candidates))][: -8]
        noise_sample_ch1, noise_sr = sf.read(noise_sample_basename + ".CH1.wav")

        if noise_sr != sample_rate: #Resample
            noise_sample_ch1 = resample(noise_sample_ch1, noise_sr, sample_rate)
            
        start = np.random.randint(noise_sample_ch1.shape[-1]-len_speech)
        noise_sample = np.stack([ sf.read(noise_sample_basename + f".CH{i_ch+1}.wav")[0].squeeze()[start: start+len_speech]
            for i_ch in range(channels) ])

        if noise_sr != sample_rate: #Resample
            noise_sample_resampled = np.stack([ resample(noise_sample_ch, noise_sr, sample_rate) for noise_sample_ch in noise_sample ])
            noise_sample = noise_sample_resampled

    elif dataset == "qut":
        raise NotImplementedError

    return noise_sample, noise_sr
