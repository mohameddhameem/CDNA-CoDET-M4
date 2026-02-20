"""
Device Utility: GPU/CPU Detection and Configuration
Works seamlessly in Colab and Local environments
"""

import torch
from typing import Optional, Dict


class DeviceManager:
    """Manage device selection (GPU/CPU) for training"""

    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available"""
        return torch.cuda.is_available()

    @staticmethod
    def get_device_info() -> Dict[str, str]:
        """Get detailed device information"""
        is_colab = DeviceManager.is_colab()
        cuda_available = DeviceManager.is_cuda_available()

        info = {
            'environment': 'Colab' if is_colab else 'Local',
            'cuda_available': str(cuda_available),
            'pytorch_version': torch.__version__,
        }

        if cuda_available:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return info

    @staticmethod
    def auto_select_device() -> str:
        """
        Automatically select best device.
        Priority: GPU > CPU
        """
        if DeviceManager.is_cuda_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def validate_device(device: str) -> str:
        """
        Validate and return valid device string.
        If requested device unavailable, returns CPU.
        """
        if device == "cuda":
            if DeviceManager.is_cuda_available():
                return "cuda"
            else:
                print("⚠ CUDA requested but not available. Using CPU.")
                return "cpu"
        return "cpu"

    @staticmethod
    def print_device_info():
        """Print device information"""
        info = DeviceManager.get_device_info()
        print(f"Environment: {info['environment']}")
        print(f"CUDA Available: {info['cuda_available']}")
        print(f"PyTorch Version: {info['pytorch_version']}")
        if 'gpu_name' in info:
            print(f"GPU: {info['gpu_name']}")
            print(f"GPU Memory: {info['gpu_memory_gb']:.1f} GB")


def get_device(auto_select: bool = True, device: Optional[str] = None) -> str:
    """
    Get device for training.

    Args:
        auto_select: If True, automatically select GPU if available
        device: Force specific device ('cuda' or 'cpu'). Overrides auto_select.

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device is not None:
        return DeviceManager.validate_device(device)

    if auto_select:
        return DeviceManager.auto_select_device()

    return "cpu"
