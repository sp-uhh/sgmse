import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch

import matplotlib.pyplot as plt, matplotlib as mpl
import ipywidgets as widgets

from sgmse.data_module import SpecsDataModule
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel

from utils import energy_ratios

from IPython.display import display, Audio

# File directories
clean_dir = '/export/home/jrichter/data/VoiceBank/valid/clean/'
noisy_dir = '/export/home/jrichter/data/VoiceBank/valid/noisy/'

# Model checkpoint
#checkpoint_file = '/export/home/jrichter/repos/score-speech/sgmse_logs/SGMSE/sweet-resonance-97/epoch=325-step=39445.ckpt'
#checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/1r5f7rwb/checkpoints/epoch=44-step=32579.ckpt'
checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/2haceomy/checkpoints/epoch=295-step=214303.ckpt'

# Load score model 
model = ScoreModel.load_from_checkpoint(
    checkpoint_file, base_dir='/export/home/jrichter/data/VoiceBank/',
    batch_size=16, num_workers=0, kwargs=dict(gpu=False)
)
model.eval(no_ema=False)
model.cuda()

noisy_files = glob.glob('{}/*.wav'.format(noisy_dir))

noisy_file = noisy_dir + 'p257_432.wav'
noisy_file = noisy_dir + 'p232_164.wav'

# Settings
sr = 16000
snr = 0.33
N = 50
corrector_steps = 1

filename = noisy_file.split('/')[-1]
clean_file = '{}/{}'.format(clean_dir, filename)

# Load wavs
x, _ = load(clean_file)
y, _ = load(noisy_file) 
T_orig = x.size(1)   

# Normalize per utterance
norm_factor = x.abs().max()
x = x / norm_factor
y = y / norm_factor

# Prepare DNN input
Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
X = torch.unsqueeze(model._forward_transform(model._stft(x.cuda())), 0)


T = Y.size(3)
div = int(T / 64)
num_pad = 64*(div+1) - T
pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))

Y = pad2d(Y)
X = pad2d(X)

#Y = Y[:,:,:256,:256]
#X = X[:,:,:256,:256]

# Reverse sampling
sampler = model.get_pc_sampler(
    'reverse_diffusion', 'ald', Y.cuda(), N=N, 
    corrector_steps=corrector_steps, snr=snr)
sample, reverse_samples, xt1, full_n_steps = sampler()

# Backward transform in time domain
x_hat = model.to_audio(sample.squeeze(), T_orig)


x_hat = model.to_audio(sample.squeeze())
x_hat_len = x_hat.size(0)
x = x[:,:x_hat_len]
y = y[:,:x_hat_len]

# Linear combination with noisy audio
mixed_audios = 0.8*x_hat.cpu() + 0.2*y
recon = x_hat.cpu()

x_hat = x_hat.squeeze().cpu().numpy()
x = x.squeeze().cpu().numpy()
y = y.squeeze().cpu().numpy()
n = y - x
reverse_samples.insert(0, xt1)

reverse_audios = []
for reverse_sample in reverse_samples:
    reverse_audios.append(model.to_audio(reverse_sample.squeeze().cpu(), T_orig))