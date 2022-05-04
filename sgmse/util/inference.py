import torch
from torchaudio import load
from utils import si_sdr
from pesq import pesq
from tqdm import tqdm


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

def evaluate_model(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)

    _pesq = 0
    _si_sdr = 0

    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
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
        Y = pad_spec(Y)

        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', 'ald', Y.cuda(), N=N, 
            corrector_steps=corrector_steps, snr=snr)

        sample, _, _, _ = sampler()

        x_hat = model.to_audio(sample.squeeze(), T_orig)

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files

