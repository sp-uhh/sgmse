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

# File directories
#clean_dir = '/export/home/jrichter/data/WSJ0+CHiME3/test/clean'
#noisy_dir = '/export/home/jrichter/data/WSJ0+CHiME3/test/noisy'
sgmse_dir = '/export/home/jrichter/repos/sgmse/enhanced/wklkh13a_322/'

clean_dir = '/export/home/jrichter/data/VoiceBank/valid/clean/'
noisy_dir = '/export/home/jrichter/data/VoiceBank/valid/noisy/'


# Model checkpoint
#checkpoint_file = '/export/home/jrichter/repos/score-speech/sgmse_logs/SGMSE/sweet-resonance-97/epoch=325-step=39445.ckpt'
#checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/2haceomy/checkpoints/epoch=295-step=214303.ckpt'
#checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/2haceomy/checkpoints/epoch=401-step=291047.ckpt'
checkpoint_file = '/export/home/jrichter/repos/sgmse/logs/sgmse/wklkh13a/checkpoints/epoch=322-step=116925.ckpt'

# Settings
sr = 16000
snr = 0.33
N = 50
corrector_steps = 1

def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)

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
    Y = pad_spec(Y)
    
    # Reverse sampling
    sampler = model.get_pc_sampler(
        'reverse_diffusion', 'ald', Y.cuda(), N=N, 
        corrector_steps=corrector_steps, snr=snr)
    sample, reverse_samples, xt1, full_n_steps = sampler()
    
    # Backward transform in time domain
    x_hat = model.to_audio(sample.squeeze(), T_orig)
    
    # Linear combination with noisy audio
    mixed_audios = 0.8*x_hat.cpu() + 0.2*y
    recon = x_hat.cpu()

    # Write enhanced wav file
    write(sgmse_dir+'mixed/'+filename, mixed_audios.numpy()[0,:], 16000)
    write(sgmse_dir+filename, recon.numpy(), 16000)
    
    