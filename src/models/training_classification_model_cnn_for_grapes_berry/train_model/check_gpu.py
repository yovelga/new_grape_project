"""Quick script to check GPU availability and CUDA setup"""
import torch

print("\n" + "="*60)
print("GPU AVAILABILITY CHECK")
print("="*60)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # GPU Details
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Memory info
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")

    # Test GPU operation
    print("\nTesting GPU operation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU computation test successful!")

    # Memory usage
    print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("\n⚠ WARNING: CUDA is not available!")
    print("Possible reasons:")
    print("1. No NVIDIA GPU installed")
    print("2. CUDA toolkit not installed")
    print("3. PyTorch installed without CUDA support")
    print("\nTo install PyTorch with CUDA, visit: https://pytorch.org/get-started/locally/")

print("="*60 + "\n")

