#!/bin/bash
# Activate the conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate aichat_app

# Export environment variables from .env file
export $(grep -v '^#' ~/Develop/personal/solospeaks/.env | xargs)

# Navigate to the project directory
cd ~/Develop/personal/solospeaks

# Run the Streamlit app with a fixed URL
streamlit run home.py --server.port 8503 --server.address 127.0.0.1
