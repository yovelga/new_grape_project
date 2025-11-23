import torch
import sys

print("Using Python from:", sys.executable)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("✅ GPU detected:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not detected by PyTorch")
