import os
import subprocess
import torch

def run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(stderr.decode())
    return process.returncode

def setup_environment():
    print("Setting up environment...")
    
    # Install basic requirements
    print("\nInstalling basic requirements...")
    run_command("pip install torch>=2.0.0 transformers>=4.30.0 tokenizers>=0.13.3 datasets>=2.12.0 numpy>=1.24.3 tqdm>=4.65.0 wandb>=0.15.4")
    
    # Install NCCL developer package
    print("\nInstalling NCCL...")
    run_command("apt-get update && apt-get install -y libnccl2 libnccl-dev")
    
    # Print CUDA version
    cuda_version = torch.version.cuda
    print(f"\nCUDA version: {cuda_version}")
    
    # Install FastMoE
    print("\nInstalling FastMoE...")
    run_command("git clone https://github.com/laekov/fastmoe.git")
    os.chdir("fastmoe")
    
    # Set environment variables for installation
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6"  # Common GPU architectures in Colab
    
    # Install FastMoE
    run_command("pip install -e .")
    os.chdir("..")
    
    # Install the msingi1 package in development mode
    print("\nInstalling msingi1 package...")
    run_command("git clone https://github.com/your-username/msingi1.git")
    os.chdir("msingi1")
    run_command("pip install -e .")
    os.chdir("..")
    
    print("\nVerifying installations...")
    try:
        import fmoe
        print("FastMoE installed successfully!")
    except ImportError as e:
        print(f"Error importing FastMoE: {e}")
        
    try:
        from src.data_processor import SwahiliDataset
        print("Msingi1 package installed successfully!")
    except ImportError as e:
        print(f"Error importing Msingi1: {e}")
    
    print("\nSetup complete!")

if __name__ == "__main__":
    setup_environment()
