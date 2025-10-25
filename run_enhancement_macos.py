#!/usr/bin/env python3
"""
COMPLETE FIX for CUDA issue on MacBook M4
This script bypasses the hardcoded .cuda() calls in sgmse/model.py
"""

import os

os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'

import torch
import torchaudio
from pathlib import Path
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec
import argparse
import time

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass


def get_device(device_arg=None):
    """Auto-detect device"""
    if device_arg and device_arg != 'auto':
        return device_arg

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enhance_audio_fixed(model, noisy_path, output_dir, device='cpu', sample_rate=16000):
    """Enhancement bypassing model.enhance()"""
    print(f"🎵 Processing: {noisy_path}")

    # Load audio
    noisy, sr = torchaudio.load(noisy_path)
    if sr != sample_rate:
        noisy = torchaudio.functional.resample(noisy, sr, sample_rate)
    if noisy.shape[0] > 1:
        noisy = torch.mean(noisy, dim=0, keepdim=True)

    # Normalize
    T_orig = noisy.size(1)
    norm_factor = noisy.abs().max().item()
    noisy = noisy / norm_factor
    noisy = noisy.to(device)

    # STFT
    Y = model._stft(noisy)
    Y = model._forward_transform(Y)
    Y = torch.unsqueeze(Y, 0)
    Y = pad_spec(Y).to(device)

    # Enhancement
    print("⚙️  Enhancement...")
    with torch.no_grad():
        if model.sde.__class__.__name__ == 'OUVESDE':
            sampler = model.get_pc_sampler('reverse_diffusion', 'ald', Y, N=30, corrector_steps=1, snr=0.5)
        else:
            sampler = model.get_sb_sampler(sde=model.sde, y=Y, sampler_type='ode')
        sample, nfe = sampler()

    # Convert to audio
    x_hat = model.to_audio(sample.squeeze(), T_orig)
    x_hat = (x_hat * norm_factor).squeeze().cpu()

    # Save
    output_path = Path(output_dir) / f"enhanced_{Path(noisy_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), x_hat.unsqueeze(0), sample_rate)
    print(f"✅ Saved: {output_path}\n")
    return x_hat


def main():
    parser = argparse.ArgumentParser(description='Fixed Enhancement for MacBook M4')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--noisy_dir', required=True, help='Noisy audio directory')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--device', default='auto', help='Device: auto/cpu/mps/cuda')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"🚀 Using device: {device}")

    # Load model
    model = ScoreModel.load_from_checkpoint(args.checkpoint, base_dir='', batch_size=1, num_workers=0,
                                            kwargs=dict(gpu=False))
    model = model.to(device)
    model.eval()
    print("✅ Model loaded!\n")

    # Process files
    audio_files = list(Path(args.noisy_dir).glob('*.wav'))
    print(f"🎯 Found {len(audio_files)} files\n")

    for audio_file in audio_files:
        try:
            enhance_audio_fixed(model, str(audio_file), args.output_dir, device)
        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("✅ Done!")


if __name__ == '__main__':
    main()