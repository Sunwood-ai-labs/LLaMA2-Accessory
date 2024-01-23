# 基本イメージとしてPythonの公式イメージを使用
#FROM python:3.9
FROM nvcr.io/nvidia/pytorch:23.12-py3

# gitをインストール（一部のベースイメージにはgitがプリインストールされていません）
RUN apt-get update && apt-get install -y git

# 作業ディレクトリを設定
WORKDIR /app

# 現在のディレクトリのファイルをコンテナの作業ディレクトリにコピー
COPY . /app

# ここで必要なリポジトリをクローンする
# RUN git clone https://github.com/your-repository.git /app

# # requirements.txtを使用して必要なPythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flash-attn --no-build-isolation

# To enable the segmentation ability shown in our official demo, SAM is also needed:
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# install LLaMA2-Accessory 
RUN pip install -e .
RUN pip uninstall -y apex



