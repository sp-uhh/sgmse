#!/usr/bin/env python3
"""
Script chạy thực nghiệm Speech Enhancement với SGMSE
Tái tạo kết quả như trong paper
Fixed for Apple Silicon M4 and CUDA compatibility
"""

import os
import argparse
import torch
import torchaudio
from pathlib import Path
from sgmse.model import ScoreModel
import numpy as np

# Fix torchaudio backend for macOS
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

def get_device():
    """Tự động detect device phù hợp"""
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Sử dụng CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Sử dụng Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("💻 Sử dụng CPU")
    return device

def load_model(checkpoint_path, device):
    """Load pretrained SGMSE model"""
    print(f"📥 Đang load model từ {checkpoint_path}...")
    
    # Load model without GPU first
    model = ScoreModel.load_from_checkpoint(
        checkpoint_path, 
        base_dir='',
        batch_size=1,
        num_workers=0,
        kwargs=dict(gpu=False)
    )
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded thành công trên {device}!")
    return model

def enhance_audio(model, noisy_audio_path, output_dir, device, sample_rate=16000):
    """
    Thực hiện speech enhancement trên file audio
    """
    print(f"🎵 Xử lý file: {noisy_audio_path}")
    
    try:
        # Load audio
        noisy, sr = torchaudio.load(noisy_audio_path)
    except Exception as e:
        print(f"❌ Lỗi load audio: {e}")
        print("💡 Thử cài: pip install soundfile")
        return None, None
    
    # Resample nếu cần
    if sr != sample_rate:
        noisy = torchaudio.functional.resample(noisy, sr, sample_rate)
    
    # Đảm bảo mono
    if noisy.shape[0] > 1:
        noisy = torch.mean(noisy, dim=0, keepdim=True)
    
    # Normalize
    noisy = noisy / (torch.abs(noisy).max() + 1e-8)
    
    # Move to device
    noisy = noisy.to(device)
    
    # Enhancement
    print("⚙️  Đang thực hiện enhancement...")
    with torch.no_grad():
        try:
            enhanced = model.enhance(noisy.unsqueeze(0), sample_rate)
        except Exception as e:
            print(f"❌ Lỗi enhancement: {e}")
            # Fallback to CPU if device fails
            if device != "cpu":
                print("🔄 Thử lại với CPU...")
                noisy = noisy.cpu()
                model_cpu = model.cpu()
                enhanced = model_cpu.enhance(noisy.unsqueeze(0), sample_rate)
                model.to(device)  # Move back
            else:
                raise e
    
    # Lưu kết quả
    output_path = Path(output_dir) / f"enhanced_{{Path(noisy_audio_path).name}}"
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
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, cuda, mps')
    
    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
        print(f"🎯 Sử dụng device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Tìm tất cả file audio trong thư mục
    audio_files = list(Path(args.noisy_dir).glob('*.wav'))
    audio_files.extend(list(Path(args.noisy_dir).glob('*.flac')))
    
    print(f"\n🎯 Tìm thấy {len(audio_files)} file audio")
    
    # Xử lý từng file
    success_count = 0
    for audio_file in audio_files:
        try:
            result = enhance_audio(model, str(audio_file), args.output_dir, device, args.sample_rate)
            if result[0] is not None:
                success_count += 1
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {audio_file}: {e}")
    
    print(f"\n✅ Hoàn thành {success_count}/{len(audio_files)} files!")

if __name__ == '__main__':
    main()