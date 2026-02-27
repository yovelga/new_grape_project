import torch
import os

def is_running_via_ssh():
    return "SSH_CONNECTION" in os.environ or "SSH_TTY" in os.environ or "SSH_CLIENT" in os.environ

if is_running_via_ssh():
    print("The code is running via SSH")
else:
    print("The code is running locally")




def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"GPU in use: {torch.cuda.current_device()}")
    else:
        print("❌ No GPU found. Ensure your server connection is correct.")

check_gpu()
print("check changes on SSH server again")
