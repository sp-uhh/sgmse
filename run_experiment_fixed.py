#!/usr/bin/env python3
"""
Fixed version - Bypass model.enhance() method
"""

import os

os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'

import torch
import torchaudio
from pathlib import Path
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec
import argparse

# Set backend
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass


def enhance_audio_fixed(model, noisy_path, output_dir, device='cpu', sample_rate=16000):
    """Enhancement mà không dùng .cuda()"""
    print(f"🎵 Xử lý: {noisy_path}")

    # Load audio
    noisy, sr = torchaudio.load(noisy_path)

    # Resample
    if sr != sample_rate:
        noisy = torchaudio.functional.resample(noisy, sr, sample_rate)

    # Mono
    if noisy.shape[0] > 1:
        noisy = torch.mean(noisy, dim=0, keepdim=True)

    # Normalize
    norm_factor = noisy.abs().max().item()
    noisy = noisy / norm_factor

    # Move to device
    noisy = noisy.to(device)

    # STFT
    Y = model._forward_transform(model._stft(noisy))
    Y = torch.unsqueeze(Y, 0)
    Y = pad_spec(Y)
    Y = Y.to(device)

    # Sampling (without calling enhance())
    print("⚙️  Enhancement...")
    with torch.no_grad():
        # Get sampler
        if model.sde.__class__.__name__ == 'OUVESDE':
            sampler = model.get_pc_sampler(
                'reverse_diffusion',
                'ald',
                Y,
                N=30,
                corrector_steps=1,
                snr=0.5
            )
        else:
            sampler = model.get_sb_sampler(
                sde=model.sde,
                y=Y,
                sampler_type='ode'
            )

        sample, nfe = sampler()

    # Convert back to audio
    T_orig = noisy.size(1)
    x_hat = model.to_audio(sample.squeeze(), T_orig)
    x_hat = x_hat * norm_factor
    x_hat = x_hat.squeeze().cpu()

    # Save
    output_path = Path(output_dir) / f"enhanced_{Path(noisy_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), x_hat.unsqueeze(0), sample_rate)

    print(f"✅ Saved: {output_path}")
    return x_hat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noisy_dir', required=True)
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default='cpu', help='cpu, mps, or cuda')
    args = parser.parse_args()

    # Load model
    print(f"📥 Loading model on {args.device}...")
    model = ScoreModel.load_from_checkpoint(
        args.checkpoint,
        base_dir='',
        batch_size=1,
        num_workers=0,
        kwargs=dict(gpu=False)
    )
    model = model.to(args.device)
    model.eval()
    print("✅ Model loaded!")

    # Process files
    audio_files = list(Path(args.noisy_dir).glob('*.wav'))
    print(f"\n🎯 Found {len(audio_files)} files")

    for audio_file in audio_files:
        try:
            enhance_audio_fixed(model, str(audio_file), args.output_dir, args.device)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ Done!")


if __name__ == '__main__':
    main()