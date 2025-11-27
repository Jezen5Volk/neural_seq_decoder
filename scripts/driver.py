import torch
import subprocess

print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch's Compiled CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Get system's driver-supported CUDA version (via nvidia-smi)
try:
    nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
    # The output usually contains the CUDA version supported by the driver.
    print(f"\nSystem's NVIDIA-Driver-Supported CUDA Version (from nvidia-smi):\n{nvidia_smi_output.split('CUDA Version: ')[1].split()[0]}")
except:
    print("\nCould not run 'nvidia-smi'. Please run it in your terminal manually.")