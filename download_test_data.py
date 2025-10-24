#!/usr/bin/env python3
"""
Script download test data mẫu để thử nghiệm
"""

import os
import urllib.request
from pathlib import Path
import argparse

def download_sample_audio(output_dir='./data/test'):
    """Download sample noisy audio files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading sample test data...")
    print("💡 Tip: Bạn có thể copy file audio của riêng bạn vào thư mục data/test/")
    print(f"✅ Đã tạo thư mục {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download test data')
    parser.add_argument('--output_dir', type=str, default='./data/test',
                       help='Thư mục lưu test data')
    
    args = parser.parse_args()
    download_sample_audio(args.output_dir)

if __name__ == '__main__':
    main()