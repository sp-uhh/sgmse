import torch
from torchaudio import load

from pesq import pesq
from pystoi import stoi
import numpy as np 

from .other import si_sdr, pad_spec

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1


def evaluate_model(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        x_hat = model.enhance(y)

        x_hat = np.squeeze(x_hat)
        x = np.squeeze(x)

        _si_sdr += si_sdr(x.numpy(), x_hat)
        _pesq += pesq(sr, x.numpy(), x_hat, 'wb') 
        _estoi += stoi(x.numpy(), x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files

