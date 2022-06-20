from os.path import join 
import numpy as np
from glob import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
import pandas as pd
from uhh_sp.evaluation import polqa
from pystoi import stoi
from pysepm import composite, SNRseg
from argparse import ArgumentParser
from __paths__ import wsj0_dir, vb_dir

from utils import energy_ratios, mean_conf_int, Method, si_sdr, mean_std


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test", nargs="+", default=[], help="Specify test set.")
    parser.add_argument("--train", nargs="+", default=[], help="Specify train set.")
    parser.add_argument("--base", nargs="+", default=[], help="Specify the base dir.")
    parser.add_argument("--basic", action="store_true", default=False, help="Evaluate only the basic metrics")
    args = parser.parse_args()

    base_dirs = args.base
    basic = args.basic

    # options: "wsj0" or "vb"
    test_set = args.test
    train_set = args.train

    for i, base_dir in enumerate(base_dirs):
        if test_set[i][:4] == "wsj0":
            clean_dir = join(wsj0_dir, "test/clean")
            noisy_dir = join(wsj0_dir, "test/noisy")
        elif test_set[i][:2] == "vb":
            clean_dir = join(vb_dir, "valid/clean")
            noisy_dir = join(vb_dir, "valid/noisy")
            
        enhanced_dir = join(base_dir, f"test_{test_set[i]}", f"train_{train_set[i]}")
        #enhanced_dir = "/export/home/jrichter/data/wsj0_chime3/test/noisy" # for the mixture

        if basic:
            data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": []}
        else:
            data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], 
                "si_sar": [], "csig": [], "cbak": [], "covl": [], "ssnr": [], }
        sr = 16000

        # Evaluate standard metrics
        noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
        for noisy_file in tqdm(noisy_files):
            filename = noisy_file.split('/')[-1]
            x, _ = read(join(clean_dir, filename))
            y, _ = read(noisy_file)
            n = y - x 
            x_method, _ = read(join(enhanced_dir, filename))
            
            data["filename"].append(filename)
            data["pesq"].append(pesq(sr, x, x_method, 'wb'))
            data["estoi"].append(stoi(x, x_method, sr, extended=True))
            data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
            if not basic:
                data["si_sir"].append(energy_ratios(x_method, x, n)[1])
                data["si_sar"].append(energy_ratios(x_method, x, n)[2])
                data["csig"].append(composite(x, x_method, sr)[0])
                data["cbak"].append(composite(x, x_method, sr)[1])
                data["covl"].append(composite(x, x_method, sr)[2])
                data["ssnr"].append(SNRseg(x, x_method, sr))

        # Save results as DataFrame    
        df = pd.DataFrame(data)

        # POLQA evaluation
        if not basic:
            clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
            enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))

            clean_audios = [read(clean_file)[0] for clean_file in clean_files]
            enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]

            polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
            polqa_vals = [val[1] for val in polqa_vals]

            # Add POLQA column to DataFrame
            df['polqa'] = polqa_vals

        # Print results
        if basic:
            print(enhanced_dir)
            print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
            print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
            print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))

        # Save DataFrame as csv file
        if basic:
            df.to_csv(join(enhanced_dir, "_results_basic.csv"), index=False)
        else:
            df.to_csv(join(enhanced_dir, "_results.csv"), index=False)

        