from setuptools import setup, find_packages

setup(
    name="msingi1",
    version="0.1.0",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'numpy>=1.21.0',
        'tqdm>=4.62.0',
        'wandb>=0.15.0'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'msingi1-train=src.train_with_shards:main',
        ],
    },
)
