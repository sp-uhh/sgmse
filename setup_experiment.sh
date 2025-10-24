#!/bin/bash
# Script để setup môi trường thực nghiệm Speech Enhancement với SGMSE

echo "🚀 Bắt đầu setup môi trường thực nghiệm SGMSE..."

# Tạo virtual environment
echo "📦 Tạo virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies từ requirements.txt
echo "📚 Cài đặt dependencies..."
pip install -r requirements.txt

# Install thêm các packages cần thiết cho visualization và metrics
echo "📊 Cài đặt packages bổ sung..."
pip install librosa soundfile matplotlib seaborn pesq pystoi scipy

# Install package hiện tại
pip install -e .

echo "✅ Setup hoàn tất!"
echo "💡 Để activate environment: source venv/bin/activate"