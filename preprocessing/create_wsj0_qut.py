import os
from glob import glob
from librosa import load
from librosa.core import resample
import argparse
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from soundfile import write
from tqdm import tqdm


# Python script for generating noisy mixtures for training
#
# Mix WSJ0 with QUT noise with SNR sampled uniformly in [min_snr, max_snr]


min_snr = 0
max_snr = 15
sr = 16000


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("wsj0", type=str, help='path to WSJ0 directory')
    parser.add_argument("qut", type=str,  help='path to QUT directory')
    parser.add_argument("target", type=str, help='target path for training files')
    args = parser.parse_args()

    # Clean speech for training
    train_speech_files = sorted(glob(args.wsj0 + '**/si_tr_s/**/*.wav', recursive=True))
    valid_speech_files = sorted(glob(args.wsj0 + '**/si_dt_05/**/*.wav', recursive=True))
    test_speech_files = sorted(glob(args.wsj0 + '**/si_et_05/**/*.wav', recursive=True))

    # Load QUT noise files
    print('Loading QUT noise files')
    cafe, sr_QUT = load(glob(args.qut + '**/CAFE-CAFE-1.wav', recursive=True)[0], sr=None)
    car, sr_QUT = load(glob(args.qut + '**/CAR-WINDOWNB-1.wav', recursive=True)[0], sr=None)
    home, sr_QUT = load(glob(args.qut + '**/HOME-KITCHEN-1.wav', recursive=True)[0], sr=None)
    street, sr_QUT = load(glob(args.qut + '**/STREET-CITY-1.wav', recursive=True)[0], sr=None)

    print('Resampling QUT noise files to 16kHz')
    cafe = resample(cafe, sr_QUT, sr)
    car = resample(car, sr_QUT, sr)
    home = resample(home, sr_QUT, sr)
    street = resample(street, sr_QUT, sr)

    # ToDo: resampling with ffmpeg bacause librosa is soooo slow
    # cafe, fs_QUT = load(os.path.join(args.qut, 'CAFE-CAFE-1_16k.wav'), sr=None)
    # car, fs_QUT = load(os.path.join(args.qut, 'CAR-WINDOWNB-1_16k.wav'), sr=None)
    # home, fs_QUT = load(os.path.join(args.qut, 'HOME-KITCHEN-1_16k.wav'), sr=None)
    # street, fs_QUT = load(os.path.join(args.qut, 'STREET-CITY-1_16k.wav'), sr=None)

    # Remove sweeps in the first and last 2 min in car noise file
    car = car[120*sr:-120*sr]

    # Create target dir
    train_clean_path = Path(os.path.join(args.target, 'train/clean'))
    train_noisy_path = Path(os.path.join(args.target, 'train/noisy'))
    valid_clean_path = Path(os.path.join(args.target, 'valid/clean'))
    valid_noisy_path = Path(os.path.join(args.target, 'valid/noisy'))
    test_clean_path = Path(os.path.join(args.target, 'test/clean'))
    test_noisy_path = Path(os.path.join(args.target, 'test/noisy'))

    train_clean_path.mkdir(parents=True, exist_ok=True)
    train_noisy_path.mkdir(parents=True, exist_ok=True)
    valid_clean_path.mkdir(parents=True, exist_ok=True)
    valid_noisy_path.mkdir(parents=True, exist_ok=True)
    test_clean_path.mkdir(parents=True, exist_ok=True)
    test_noisy_path.mkdir(parents=True, exist_ok=True)

    # Initialize seed for reproducability
    np.random.seed(0)

    # Create files for training
    print('Create training files')
    for i, speech_file in enumerate(tqdm(train_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = np.random.uniform(min_snr, max_snr)
        noise_type = np.random.randint(4)
        speech_power = 1/len(s)*np.sum(s**2)

        if noise_type == 0:
            start = np.random.randint(len(cafe)-len(s))
            n = cafe[start:start+len(s)]
        elif noise_type == 1:
            start = np.random.randint(len(home)-len(s))
            n = home[start:start+len(s)]
        elif noise_type == 2:
            start = np.random.randint(len(street)-len(s))
            n = street[start:start+len(s)]
        elif noise_type == 3:
            start = np.random.randint(len(car)-len(s))
            n = car[start:start+len(s)]
        else:
            raise ValueError('Unexpected noise type index')
        noise_power = 1/len(n)*np.sum(n**2)
        noise_power_target = speech_power*np.power(10,-snr_dB/10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        file_name = speech_file.split('/')[-1]
        write(os.path.join(train_clean_path, file_name), s, sr)
        write(os.path.join(train_noisy_path, file_name), x, sr)

    # Create files for validation
    print('Create validation files')
    for i, speech_file in enumerate(tqdm(valid_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = np.random.uniform(min_snr, max_snr)
        noise_type = np.random.randint(4)
        speech_power = 1/len(s)*np.sum(s**2)

        if noise_type == 0:
            start = np.random.randint(len(cafe)-len(s))
            n = cafe[start:start+len(s)]
        elif noise_type == 1:
            start = np.random.randint(len(home)-len(s))
            n = home[start:start+len(s)]
        elif noise_type == 2:
            start = np.random.randint(len(street)-len(s))
            n = street[start:start+len(s)]
        elif noise_type == 3:
            start = np.random.randint(len(car)-len(s))
            n = car[start:start+len(s)]
        else:
            raise ValueError('Unexpected noise type index')
        noise_power = 1/len(n)*np.sum(n**2)
        noise_power_target = speech_power*np.power(10,-snr_dB/10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        file_name = speech_file.split('/')[-1]
        write(os.path.join(valid_clean_path, file_name), s, sr)
        write(os.path.join(valid_noisy_path, file_name), x, sr)

    # Create files for test
    print('Create test files')
    for i, speech_file in enumerate(tqdm(test_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = np.random.uniform(min_snr, max_snr)
        noise_type = np.random.randint(4)
        speech_power = 1/len(s)*np.sum(s**2)

        if noise_type == 0:
            start = np.random.randint(len(cafe)-len(s))
            n = cafe[start:start+len(s)]
        elif noise_type == 1:
            start = np.random.randint(len(home)-len(s))
            n = home[start:start+len(s)]
        elif noise_type == 2:
            start = np.random.randint(len(street)-len(s))
            n = street[start:start+len(s)]
        elif noise_type == 3:
            start = np.random.randint(len(car)-len(s))
            n = car[start:start+len(s)]
        else:
            raise ValueError('Unexpected noise type index')
        noise_power = 1/len(n)*np.sum(n**2)
        noise_power_target = speech_power*np.power(10,-snr_dB/10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        file_name = speech_file.split('/')[-1]
        write(os.path.join(test_clean_path, file_name), s, sr)
        write(os.path.join(test_noisy_path, file_name), x, sr)