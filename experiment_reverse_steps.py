import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from torchaudio import load
import torch
import pandas as pd
from argparse import ArgumentParser

from sgmse.data_module import SpecsDataModule
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel
from os.path import join

from utils import pad_spec, ensure_dir, si_sdr


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help='Directory containing the data set')
    parser.add_argument("--ckpt", type=str,  required=True, help='Path to model checkpoint.')
    parser.add_argument("--snr", type=float,  default=0.33, help='snr parameter')
    parser.add_argument("--start", type=int,  default=1, help='Minimum number of reverse steps')
    parser.add_argument("--end", type=int,  default=100, help='Maximum number of reverse steps')
    args = parser.parse_args()

    checkpoint_file = args.ckpt

    clean_dir = join(args.dir, "clean")
    noisy_dir = join(args.dir, "noisy")

    # Settings
    sr = 16000
    snr = args.snr
    # N = 50
    corrector_steps = 1

    # Load score model 
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, base_dir='/export/home/jrichter/data/wsj0_chime3/',
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()

    data = {"reverse_steps": [], "pesq": [], "estoi": [], "si_sdr": []}
    df = pd.DataFrame(data)

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    for reverse_steps in range(args.start, args.end+1):
        print(f"reverse steps: {reverse_steps}/{args.end}")

        _pesq = []; _estoi = []; _si_sdr = []
        for noisy_file in tqdm(noisy_files):
            filename = noisy_file.split('/')[-1]
            
            # Load wav
            x, _ = load(join(clean_dir, filename)) 
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
                'reverse_diffusion', 'ald', Y.cuda(), N=reverse_steps, 
                corrector_steps=corrector_steps, snr=snr)
            sample = sampler()
            
            # Backward transform in time domain
            x_hat = model.to_audio(sample.squeeze(), T_orig)

            # Renormalize
            x_hat = x_hat * norm_factor

            x = x.squeeze().cpu().numpy()
            x_hat = x_hat.squeeze().cpu().numpy()

            _pesq.append(pesq(sr, x, x_hat, 'wb'))
            _estoi.append(stoi(x, x_hat, sr, extended=True))
            _si_sdr.append(si_sdr(x, x_hat))
            
            # Write enhanced wav file
        data["reverse_steps"].append(reverse_steps)
        data["pesq"].append(np.array(_pesq).mean())
        data["estoi"].append(np.array(_estoi).mean())
        data["si_sdr"].append(np.array(_si_sdr).mean())    

    df = pd.DataFrame(data)
    df.to_csv(f"results_snr_{snr:.2f}.csv", index=False)