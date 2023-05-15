#!/bin/bash

conda create -n vicuna -y
conda activate vicuna

sudo apt install build-essential -y
sudo apt install python3-icu
pip install protobuf==3.20.3

pip3 install -U torch fschat bs4 markdownify polyglot pyicu pycld2 openai einops flash-attn deepspeed git+https://github.com/huggingface/peft.git


pip install torch fschat bs4 markdownify polyglot pycld2 openai einops flash-attn deepspeed git+https://github.com/huggingface/peft.git
# # to install flash attention, we need cuda 11.7 or below
# https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local
# pip install flash-attn