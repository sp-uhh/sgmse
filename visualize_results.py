#!/usr/bin/env python3
"""
Script tạo visualization spectrograms
Tái tạo hình ảnh: Noisy mixture | Estimate | Clean speech
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def create_spectrogram_visualization(noisy_path, enhanced_path, clean_path=None, output_path='spectrogram_comparison.png'):
    """
    Tạo visualization 3 spectrograms: Noisy | Enhanced | Clean
    """
    print("📊 Đang tạo spectrograms...")
    
    # Load audio files
    noisy, sr = librosa.load(noisy_path, sr=None)
    enhanced, _ = librosa.load(enhanced_path, sr=sr)
    
    # Compute spectrograms (STFT)
    n_fft = 2048
    hop_length = 512
    
    D_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    D_enhanced = librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to dB scale
    S_noisy_db = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
    S_enhanced_db = librosa.amplitude_to_db(np.abs(D_enhanced), ref=np.max)
    
    # Setup figure
    if clean_path:
        clean, _ = librosa.load(clean_path, sr=sr)
        D_clean = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length)
        S_clean_db = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['Noisy mixture y', 'Estimate x̂', 'Clean speech x₀']
        spectrograms = [S_noisy_db, S_enhanced_db, S_clean_db]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        titles = ['Noisy mixture y', 'Estimate x̂']
        spectrograms = [S_noisy_db, S_enhanced_db]
    
    # Plot spectrograms
    for ax, spec, title in zip(axes, spectrograms, titles):
        img = librosa.display.specshow(spec, 
                                       sr=sr, 
                                       hop_length=hop_length,
                                       x_axis='time', 
                                       y_axis='hz',
                                       ax=ax,
                                       cmap='magma',
                                       vmin=-40,
                                       vmax=40)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency (kHz)', fontsize=12)
        ax.set_ylim(0, 20000)
        
        # Format y-axis to show kHz
        yticks = ax.get_yticks()
        ax.set_yticklabels([f'{int(y/1000)}' for y in yticks])
        
        # Add colorbar
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu visualization tại: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Tạo spectrogram visualization')
    parser.add_argument('--noisy', type=str, required=True,
                       help='File audio nhiễu')
    parser.add_argument('--enhanced', type=str, required=True,
                       help='File audio đã enhance')
    parser.add_argument('--clean', type=str, default=None,
                       help='File audio sạch (optional)')
    parser.add_argument('--output', type=str, default='spectrogram_comparison.png',
                       help='Đường dẫn lưu hình')
    
    args = parser.parse_args()
    
    create_spectrogram_visualization(
        args.noisy,
        args.enhanced,
        args.clean,
        args.output
    )

if __name__ == '__main__':
    main()