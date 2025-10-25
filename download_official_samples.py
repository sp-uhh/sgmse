#!/usr/bin/env python3
"""
Script download official test audio samples từ SGMSE project
Nguồn: https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse
"""

import urllib.request
import os
from pathlib import Path

def download_file(url, output_path):
    """Download file từ URL"""
    try:
        print(f"📥 Downloading: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_official_samples():
    """Download official test samples từ VoiceBank-DEMAND dataset"""
    
    # Tạo thư mục output
    output_dir = Path("data/official_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎙️ Downloading Official SGMSE Test Samples")
    print("=" * 60)
    
    # URLs của audio samples từ SGMSE supplementary materials
    # Đây là các samples từ VoiceBank-DEMAND test set
    samples = {
        "noisy_male_1.wav": "https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse/audio/voicebank/noisy/p232_001.wav",
        "enhanced_male_1.wav": "https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse/audio/voicebank/enhanced/p232_001.wav",
        "clean_male_1.wav": "https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse/audio/voicebank/clean/p232_001.wav",
        
        "noisy_female_1.wav": "https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse/audio/voicebank/noisy/p257_001.wav",
        "enhanced_female_1.wav": "https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse/audio/voicebank/enhanced/p257_001.wav",
        "clean_female_1.wav": "https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse/audio/voicebank/clean/p257_001.wav",
    }
    
    # Download các files
    success_count = 0
    for filename, url in samples.items():
        output_path = output_dir / filename
        if download_file(url, output_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Downloaded {success_count}/{len(samples)} files")
    print(f"📁 Files saved in: {output_dir}")
    
    # Hiển thị cấu trúc files
    print("\n📂 Downloaded files:")
    for f in sorted(output_dir.glob("*.wav")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.2f} MB)")
    
    print("\n💡 Usage:")
    print("   # Run enhancement on downloaded noisy samples:")
    print(f"   python run_experiment.py --checkpoint ./checkpoints/enhanced.ckpt --noisy_dir {output_dir} --output_dir ./results")
    print("\n   # Create spectrograms:")
    print(f"   python visualize_results.py --noisy {output_dir}/noisy_male_1.wav --enhanced ./results/enhanced_noisy_male_1.wav --clean {output_dir}/clean_male_1.wav --output ./visualizations/comparison.png")

if __name__ == "__main__":
    download_official_samples()