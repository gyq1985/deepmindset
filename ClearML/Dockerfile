# 使用 NVIDIA NGC 上的 TensorFlow 容器，已包含 CUDA 12.2 + CuDNN 9.3
FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    clearml==1.18.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    scikit-learn==1.4.2
