#!/bin/bash

# condaのインストール
# cd ~/
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

# conda create --name merge python=3.10
# conda activate merge

python -m pip  install flash-attn

# mergekitの取得
git clone https://github.com/arcee-ai/mergekit.git

# 1. pandasのインストール
conda install pandas --use-pep517

# 2. 依存関係の更新
python -m pip install --upgrade setuptools wheel

cd mergekit

python -m pip install -e .[evolve,vllm]

python -m pip3 install datasets huggingface_hub CMA-ES

python ../datasets_script.py

python -m pip install --upgrade tf-keras
python -m pip install --upgrade TensorFlow
python -m pip install --upgrade vllm
