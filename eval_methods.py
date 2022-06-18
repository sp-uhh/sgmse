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

from utils import energy_ratios, mean_conf_int, Method, si_sdr


base_dirs = ["/export/home/jrichter/repos/DVAE_SE/enhanced/", \
             "/export/home/jrichter/repos/CDiffuSE/output/Enhanced/", \
             "/export/home/jrichter/repos/score-speech/enhanced/", \
             "/export/home/jrichter/repos/sgmse/enhanced/"]

# options: "wsj0" or "vb"
test_set = "wsj0"
train_set = "vb"

for base_dir in base_dirs:
    if test_set == "wsj0":
        clean_dir = "/export/home/jrichter/data/wsj0_chime3/test/clean"
        noisy_dir = "/export/home/jrichter/data/wsj0_chime3/test/noisy"
    elif test_set == "vb":
        clean_dir = '/export/home/jrichter/data/VoiceBank/valid/clean'
        noisy_dir = '/export/home/jrichter/data/VoiceBank/valid/noisy'
        
    enhanced_dir = join(base_dir, f"test_{test_set}", f"train_{train_set}")
    #enhanced_dir = "/export/home/jrichter/data/wsj0_chime3/test/noisy" # for the mixture

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], 
            "si_sar": [], "csig": [], "cbak": [], "covl": [], "ssnr": [], }
    sr = 16000

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
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])
        data["csig"].append(composite(x, x_method, sr)[0])
        data["cbak"].append(composite(x, x_method, sr)[1])
        data["covl"].append(composite(x, x_method, sr)[2])
        data["ssnr"].append(SNRseg(x, x_method, sr))

    clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
    enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))

    clean_audios = [read(clean_file)[0] for clean_file in clean_files]
    enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]

    polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
    polqa_vals = [val[1] for val in polqa_vals]

    df = pd.DataFrame(data)
    df['polqa'] = polqa_vals
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)