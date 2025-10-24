#!/usr/bin/env python3
"""
Script chạy thực nghiệm Speech Enhancement với SGMSE
Tái tạo kết quả như trong paper
"""

import os
import argparse
import torch
import torchaudio
from pathlib import Path
from sgmse.model import ScoreModel
import numpy as np

def load_model(checkpoint_path):
    """Load pretrained SGMSE model"""
    print(f"📥 Đang load model từ {checkpoint_path}...")
    
    model = ScoreModel.load_from_checkpoint(
        checkpoint_path, 
        base_dir='',
        batch_size=1,
        num_workers=0,
        kwargs=dict(gpu=False)
    )
    model.eval()
    
    print("✅ Model loaded thành công!")
    return model

def enhance_audio(model, noisy_audio_path, output_dir, sample_rate=16000):
    """
    Thực hiện speech enhancement trên file audio
    """
    print(f"🎵 Xử lý file: {noisy_audio_path}")
    
    # Load audio
    noisy, sr = torchaudio.load(noisy_audio_path)
    
    # Resample nếu cần
    if sr != sample_rate:
        noisy = torchaudio.functional.resample(noisy, sr, sample_rate)
    
    # Đảm bảo mono
    if noisy.shape[0] > 1:
        noisy = torch.mean(noisy, dim=0, keepdim=True)
    
    # Normalize
    noisy = noisy / (torch.abs(noisy).max() + 1e-8)
    
    # Enhancement
    print("⚙️  Đang thực hiện enhancement...")
    with torch.no_grad():
        enhanced = model.enhance(noisy.unsqueeze(0), sample_rate)
    
    # Lưu kết quả
    output_path = Path(output_dir) / f"enhanced_{Path(noisy_audio_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(str(output_path), enhanced.squeeze(0).cpu(), sample_rate)
    print(f"💾 Đã lưu kết quả tại: {output_path}")
    
    return enhanced, noisy

def main():
    parser = argparse.ArgumentParser(description='Chạy thực nghiệm Speech Enhancement')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Đường dẫn đến pretrained checkpoint')
    parser.add_argument('--noisy_dir', type=str, required=True,
                       help='Thư mục chứa audio nhiễu')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Thư mục lưu kết quả')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Sampling rate')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Tìm tất cả file audio trong thư mục
    audio_files = list(Path(args.noisy_dir).glob('*.wav'))
    audio_files.extend(list(Path(args.noisy_dir).glob('*.flac')))
    
    print(f"\n🎯 Tìm thấy {len(audio_files)} file audio")
    
    # Xử lý từng file
    for audio_file in audio_files:
        try:
            enhance_audio(model, str(audio_file), args.output_dir, args.sample_rate)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {audio_file}: {e}")
    
    print("\n✅ Hoàn thành tất cả!")

if __name__ == '__main__':
    main()