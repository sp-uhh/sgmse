import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd

from sgmse.data_module import SpecsDataModule
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel

from pesq import pesq
from pystoi import stoi

from utils import energy_ratios, ensure_dir

def print_mean_std(data, decimal=2):
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    if decimal == 2:
        string = f'{mean:.2f} ± {std:.2f}'
    elif decimal == 1:
        string = f'{mean:.1f} ± {std:.1f}'
    return string

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test", type=str, help="Specify test set.")
    parser.add_argument("--train", type=str, help="Specify train set.")
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--sampler_type", type=str, choices=("pc", "ode"), default="pc", 
        help="Specify the sampler type")
    parser.add_argument("--corrector", type=str, choices=("ald", "none"), default="ald", 
        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for annealed Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for the ODE sampler")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for the ODE sampler")
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = "/export/home/jrichter/repos/sgmse/enhanced/test_{}/train_{}/".format(
        args.test, args.train) 

    ensure_dir(target_dir+"files/")

    # Settings
    sr = 16000
    sampler_type = args.sampler_type
    N = args.N
    corrector = args.corrector
    corrector_steps = args.corrector_steps
    snr = args.snr
    atol = args.atol
    rtol = args.rtol

    # Load score model 
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": [], "ns": []}
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file) 

        # Enhance wav
        x_hat, ns = model.enhance(y, sampler_type=sampler_type, corrector=corrector, 
            corrector_steps=corrector_steps, N=N, snr=snr, atol=atol, rtol=rtol)

        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x 

        # Write enhanced wav file
        write(target_dir+"files/"+filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_hat, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
        data["ns"].append(ns)

    # Save results as DataFrame    
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("sampler_type: {}\n".format(sampler_type))
        file.write("corrector: {}\n".format(corrector))
        file.write("corrector_steps: {}\n".format(corrector_steps))
        file.write("N: {}\n".format(N))
        file.write("snr: {}\n".format(snr))
        if sampler_type == "ode":
            file.write("atol: {}\n".format(atol))
            file.write("rtol: {}\n".format(rtol))
