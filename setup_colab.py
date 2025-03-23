import os
import subprocess

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
    
    # Install FastMoE from source
    print("\nInstalling FastMoE from source...")
    run_command("git clone https://github.com/laekov/fastmoe.git")
    os.chdir("fastmoe")
    run_command("pip install -e .")
    os.chdir("..")
    
    print("\nSetup complete!")

if __name__ == "__main__":
    setup_environment()
