import glob
from argparse import ArgumentParser
import os
from os.path import join

import torch
from torchaudio import load
from soundfile import write
from tqdm import tqdm
import numpy as np

from sgmse.model import ScoreModel, DiscriminativeModel
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
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    checkpoint_file = args.ckpt

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    sr = 16000

    # Load score model 
    model_cls = ScoreModel if not args.discriminatively else DiscriminativeModel
    model = model_cls.load_from_checkpoint(
        args.ckpt, batch_size=1, num_workers=0)
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        y, sr = load(noisy_file) 
        assert sr == 16000, "Pretrained models worked wth sampling rate of 16000"
        x_hat = model.enhance(y, corrector=args.corrector, N=args.N, corrector_steps=args.corrector_steps, snr=args.snr)
        
        # Write enhanced wav file
        write(f'{args.out_dir}/{os.path.basename(noisy_file)}', x_hat, 16000)