import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch

from sgmse.data_module import SpecsDataModule
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel

from utils import pad_spec

# File directories
clean_dir = '/export/home/jrichter/data/WSJ0+CHiME3/test/clean'
noisy_dir = '/export/home/jrichter/data/WSJ0+CHiME3/test/noisy'
sgmse_dir = '/export/home/jrichter/repos/sgmse/enhanced/3hpyr94h_348/'

#clean_dir = '/export/home/jrichter/data/VoiceBank/valid/clean/'
#noisy_dir = '/export/home/jrichter/data/VoiceBank/valid/noisy/'


# Model checkpoint
#checkpoint_file = '/export/home/jrichter/repos/score-speech/sgmse_logs/SGMSE/sweet-resonance-97/epoch=325-step=39445.ckpt'
#checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/2haceomy/checkpoints/epoch=295-step=214303.ckpt'
#checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/2haceomy/checkpoints/epoch=401-step=291047.ckpt'
checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/3hpyr94h/checkpoints/epoch=348-step=139599.ckpt'

# Settings
sr = 16000
snr = 0.33
N = 50
corrector_steps = 1


# Load score model 
model = ScoreModel.load_from_checkpoint(
    checkpoint_file, base_dir='/export/home/jrichter/data/VoiceBank/',
    batch_size=16, num_workers=0, kwargs=dict(gpu=False)
)
model.eval(no_ema=False)
model.cuda()

noisy_files = glob.glob('{}/*.wav'.format(noisy_dir))

for noisy_file in tqdm(noisy_files):
    filename = noisy_file.split('/')[-1]
    clean_file = '{}/{}'.format(clean_dir, filename)
    diffuse_file = '{}/{}'.format(sgmse_dir, filename)
    
    # Load wav
    y, _ = load(noisy_file) 
    T_orig = y.size(1)   

    # Normalize
    norm_factor = y.abs().max()
    y = y / norm_factor
    
    # Prepare DNN input
    Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
    Y = pad_spec(Y)
    
    # Reverse sampling
    sampler = model.get_pc_sampler(
        'reverse_diffusion', 'ald', Y.cuda(), N=N, 
        corrector_steps=corrector_steps, snr=snr)
    sample = sampler()
    
    # Backward transform in time domain
    x_hat = model.to_audio(sample.squeeze(), T_orig)

    # Renormalize
    x_hat = x_hat * norm_factor

    # Write enhanced wav file
    write(sgmse_dir+filename, x_hat.cpu().numpy(), 16000)
    
    