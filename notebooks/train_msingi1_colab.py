# -*- coding: utf-8 -*-
"""
# Training Msingi1: A Small Swahili Language Model
This script will guide you through training Msingi1 on Google Colab's GPU/TPU.
"""

# Clone the repository and install dependencies
!git clone https://github.com/YOUR_USERNAME/msingi1.git
%cd msingi1
!pip install -r requirements.txt

# Upload the dataset
from google.colab import files
print("Please upload your archive.zip file...")
uploaded = files.upload()

# Train the model
from src.train import main
main()
