# 🍎 Hướng Dẫn Chạy SGMSE trên MacBook Air M4

## ✅ Yêu cầu
- MacBook Air M4 (hoặc bất kỳ Mac với Apple Silicon)
- macOS 12.0+ (Monterey trở lên)
- RAM: 8GB+ (khuyến nghị 16GB)
- Storage: 10GB trống

## 📦 Bước 1: Cài đặt Prerequisites

### Install Homebrew (nếu chưa có)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Python 3.10+
```bash
brew install python@3.10
```

### Install FFmpeg (cho audio processing)
```bash
brew install ffmpeg
```

## 🔧 Bước 2: Setup Virtual Environment

```bash
# Clone repo của bạn
git clone https://github.com/nghiata-uit/sgmse.git
cd sgmse

# Checkout branch experiment-reproduction
git checkout experiment-reproduction

# Tạo virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

## 🎯 Bước 3: Install PyTorch với MPS Support

**Quan trọng:** Cài PyTorch version hỗ trợ Apple Silicon GPU (MPS)

```bash
# Install PyTorch với MPS support
pip3 install torch torchvision torchaudio
```

Kiểm tra MPS có hoạt động không:
```python
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## 📚 Bước 4: Install Dependencies

```bash
# Install dependencies chính
pip install -r requirements.txt

# Install packages cho experiment
pip install librosa soundfile matplotlib seaborn pesq pystoi scipy

# Install package hiện tại
pip install -e .
```

## 📥 Bước 5: Download Pretrained Model

### Cách 1: Từ Hugging Face (Khuyến nghị)

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download model
mkdir -p checkpoints
python3 << EOF
from huggingface_hub import hf_hub_download
import os

# Download main checkpoint
model_path = hf_hub_download(
    repo_id="sp-uhh/speech-enhancement-sgmse",
    filename="enhanced.ckpt",
    local_dir="./checkpoints"
)
print(f"Model downloaded to: {model_path}")
EOF
```

### Cách 2: Manual Download

```bash
mkdir -p checkpoints
cd checkpoints
# Truy cập: https://huggingface.co/sp-uhh/speech-enhancement-sgmse/tree/main
# Download file .ckpt và lưu vào đây
cd ..
```

## 🎵 Bước 6: Chuẩn bị Test Data

```bash
# Tạo thư mục test data
python3 download_test_data.py --output_dir ./data/test

# Hoặc copy audio của bạn
mkdir -p data/test
cp /path/to/your/noisy_audio.wav data/test/
```

## ⚡ Bước 7: Chạy Experiment với GPU Acceleration

### Chỉnh sửa code để dùng MPS

Tạo file `run_experiment.py`:

```python
#!/usr/bin/env python3
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import argparse
from pathlib import Path
from sgmse.model import ScoreModel
import torchaudio

# Kiểm tra MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Sử dụng device: {device}")

def enhance_audio(model, noisy_path, output_dir):
    print(f"🎵 Xử lý: {noisy_path}")
    
    noisy, sr = torchaudio.load(noisy_path)
    
    if sr != 16000:
        noisy = torchaudio.functional.resample(noisy, sr, 16000)
    
    if noisy.shape[0] > 1:
        noisy = torch.mean(noisy, dim=0, keepdim=True)
    
    noisy = noisy.to(device)
    
    with torch.no_grad():
        enhanced = model.enhance(noisy.unsqueeze(0), 16000)
    
    output_path = Path(output_dir) / f"enhanced_{Path(noisy_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(str(output_path), enhanced.squeeze(0).cpu(), 16000)
    print(f"✅ Saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noisy_dir', required=True)
    parser.add_argument('--output_dir', default='./results')
    args = parser.parse_args()
    
    model = ScoreModel.load_from_checkpoint(args.checkpoint).to(device)
    model.eval()
    
    for audio_file in Path(args.noisy_dir).glob('*.wav'):
        enhance_audio(model, str(audio_file), args.output_dir)
```

Chạy:
```bash
chmod +x run_experiment.py
python3 run_experiment.py \
    --checkpoint ./checkpoints/enhanced.ckpt \
    --noisy_dir ./data/official_samples \
    --output_dir ./results
```

## 📊 Bước 8: Tạo Spectrograms

```bash
python3 visualize_results.py \
    --noisy ./data/test/noisy_sample.wav \
    --enhanced ./results/enhanced_noisy_sample.wav \
    --output ./spectrograms/comparison.png
```

## 🔥 Performance Tips cho M4

### 1. Tối ưu Memory
```bash
# Giảm batch size nếu bị out of memory
# Trong code, set batch_size=1
```

### 2. Monitoring GPU
```bash
# Xem GPU usage
sudo powermetrics --samplers gpu_power -i 1000
```

### 3. Tăng tốc độ
```python
# Bật autocast cho MPS
with torch.autocast(device_type="mps", dtype=torch.float16):
    enhanced = model.enhance(noisy)
```

## 📈 Kết quả mong đợi trên M4

- **Tốc độ**: ~2-5 giây/audio file (3-4 seconds)
- **Memory**: ~2-4GB RAM
- **GPU Usage**: 60-80%
- **Nhiệt độ**: 40-60°C (bình thường)

## 🐛 Troubleshooting

### Lỗi: "MPS backend out of memory"
```bash
# Giảm độ dài audio hoặc process theo chunks
# Hoặc dùng CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Lỗi: "No module named 'sgmse'"
```bash
pip install -e .
```

### Lỗi: "torchaudio backend not available"
```bash
brew install ffmpeg
pip install --upgrade torchaudio
```

### Model chạy chậm
```bash
# Đảm bảo dùng MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Nếu False, reinstall PyTorch
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio
```

## 💡 Tips Bonus

### 1. Batch Processing nhiều files
```bash
for f in data/test/*.wav; do
    python3 run_experiment_mps.py \
        --checkpoint ./checkpoints/enhanced.ckpt \
        --noisy_dir $(dirname "$f") \
        --output_dir ./results
done
```

### 2. Tạo comparison video
```bash
# Install additional tools
brew install sox

# Create side-by-side comparison
sox -M data/test/noisy.wav results/enhanced_noisy.wav comparison.wav
```

## 🎉 Kết luận

MacBook Air M4 hoàn toàn đủ mạnh để chạy SGMSE! Với Apple Silicon, bạn sẽ có:
- ✅ Tốc độ xử lý tốt
- ✅ Tiết kiệm pin
- ✅ Không ồn, không nóng
- ✅ Chất lượng audio enhancement cao

Chúc bạn thành công! 🚀

---

## 📚 References
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [SGMSE Paper](https://arxiv.org/abs/2208.05830)
- [Hugging Face Model](https://huggingface.co/sp-uhh/speech-enhancement-sgmse)