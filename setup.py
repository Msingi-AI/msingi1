from setuptools import setup, find_packages

setup(
    name="msingi1",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.3",
        "datasets>=2.12.0",
        "numpy>=1.24.3",
        "tqdm>=4.65.0",
        "wandb>=0.15.4"
    ],
    author="Msingi AI",
    description="A Swahili language model with Mixture of Experts",
    python_requires=">=3.7",
)
