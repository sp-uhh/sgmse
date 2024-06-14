import glob
import torch
from os import makedirs
from os.path import join, dirname
from argparse import ArgumentParser
from soundfile import write
from torchaudio import load
from tqdm import tqdm
from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--format", type=str, default='default', help='Format of the directory structure. Use "default" for the default format and "ears" for the EARS format.')
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model 
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    # Check format
    if args.format == 'default':
        noisy_files = sorted(glob.glob(join(noisy_dir, '*.wav')))
        sr = 16000
        pad_mode = "zero_pad"
    elif args.format == 'ears':
        noisy_files = sorted(glob.glob(join(noisy_dir, '**', '*.wav')))
        sr = 48000
        pad_mode = "reflection"
    else:
        raise ValueError('Unknown format')

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        filename = noisy_file.replace(noisy_dir, "")[1:] # Remove the first character which is a slash
        
        # Load wav
        y, _ = load(noisy_file) 
        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y, mode=pad_mode)
        
        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', corrector_cls, Y.cuda(), N=N, 
            corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        makedirs(dirname(join(target_dir, filename)), exist_ok=True)
        write(join(target_dir, filename), x_hat.cpu().numpy(), sr)
