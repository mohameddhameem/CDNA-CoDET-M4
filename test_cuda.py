#!/usr/bin/env python3
"""
CUDA Availability Test Script
Tests PyTorch installation and CUDA availability with detailed logging
"""

import sys
import os
from datetime import datetime

def print_log(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def main():
    print_log("="*70)
    print_log("CUDA Availability Test - Starting")
    print_log("="*70)
    
    # Python environment info
    print_log(f"Python Version: {sys.version}")
    print_log(f"Python Executable: {sys.executable}")
    print_log(f"Working Directory: {os.getcwd()}")
    print_log("")
    
    # Check PyTorch installation
    print_log("Attempting to import PyTorch...")
    try:
        import torch
        print_log("✓ PyTorch imported successfully")
        print_log(f"PyTorch Version: {torch.__version__}")
    except ImportError as e:
        print_log(f"✗ Failed to import PyTorch: {e}")
        sys.exit(1)
    
    print_log("")
    print_log("-"*70)
    print_log("CUDA Information:")
    print_log("-"*70)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print_log(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print_log(f"CUDA Version: {torch.version.cuda}")
        print_log(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print_log(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        print_log("")
        
        # Display each GPU
        for i in range(torch.cuda.device_count()):
            print_log(f"GPU {i}:")
            print_log(f"  Name: {torch.cuda.get_device_name(i)}")
            print_log(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print_log(f"  Total Memory: {total_memory:.2f} GB")
            
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print_log(f"  Allocated Memory: {allocated:.2f} GB")
                print_log(f"  Reserved Memory: {reserved:.2f} GB")
            print_log("")
        
        # Current device
        print_log(f"Current CUDA Device: {torch.cuda.current_device()}")
        print_log(f"Current Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Test tensor creation on GPU
        print_log("")
        print_log("-"*70)
        print_log("Testing GPU Tensor Operations:")
        print_log("-"*70)
        try:
            print_log("Creating test tensor on GPU...")
            test_tensor = torch.randn(1000, 1000).cuda()
            print_log(f"✓ Tensor created successfully on GPU")
            print_log(f"  Tensor shape: {test_tensor.shape}")
            print_log(f"  Tensor device: {test_tensor.device}")
            print_log(f"  Tensor dtype: {test_tensor.dtype}")
            
            print_log("Performing matrix multiplication on GPU...")
            result = torch.mm(test_tensor, test_tensor)
            print_log(f"✓ Matrix multiplication successful")
            print_log(f"  Result shape: {result.shape}")
            
            del test_tensor, result
            torch.cuda.empty_cache()
            print_log("✓ GPU memory cleared")
            
        except Exception as e:
            print_log(f"✗ GPU tensor operation failed: {e}")
            
    else:
        print_log("CUDA is NOT available")
        print_log("PyTorch will run on CPU only")
        
        # Check why CUDA might not be available
        print_log("")
        print_log("Possible reasons:")
        print_log("  - No GPU available on this node")
        print_log("  - CUDA drivers not installed")
        print_log("  - PyTorch CPU-only version installed")
        print_log("  - CUDA_VISIBLE_DEVICES set incorrectly")
        
        # Check environment variables
        print_log("")
        print_log("Environment Variables:")
        cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH']
        for var in cuda_vars:
            value = os.environ.get(var, 'Not set')
            print_log(f"  {var}: {value}")
    
    # Check if running on CPU
    print_log("")
    print_log("-"*70)
    print_log("CPU Information:")
    print_log("-"*70)
    print_log(f"Number of CPU cores: {os.cpu_count()}")
    print_log(f"PyTorch CPU threads: {torch.get_num_threads()}")
    
    print_log("")
    print_log("="*70)
    print_log("CUDA Availability Test - Completed Successfully")
    print_log("="*70)

if __name__ == "__main__":
    main()
