"""
Quick device utilities for Colab and Local
Use these for easy device checking and configuration
"""

import torch


def check_device():
    """Print device status"""
    print("=" * 60)
    print("DEVICE STATUS")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"Current GPU Device: {torch.cuda.current_device()}")
    else:
        print("GPU: Not available (will use CPU)")

    print("=" * 60)


def is_gpu_available():
    """Check if GPU is available"""
    return torch.cuda.is_available()


def get_device():
    """Get recommended device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_memory_gb():
    """Get GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0


def get_gpu_name():
    """Get GPU name"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "None"


# Quick usage in Colab
if __name__ == "__main__":
    check_device()
