FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0;8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Python installation
RUN echo "\n===>\nINSTALLING PYTHON AND NECCESSARY DEPENDENCIES\n" && \
    apt-get -y --no-install-recommends update && \
	apt-get -y --no-install-recommends upgrade && \
	apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        cmake \
        git \
        libatlas-base-dev \
        libprotobuf-dev \
        libleveldb-dev \
        libsnappy-dev \
        libhdf5-serial-dev \
        protobuf-compiler \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        liblmdb-dev \
        pciutils \
        python3-setuptools \
        python3-dev \
        python3-pip \
        opencl-headers \
        ocl-icd-opencl-dev \
        libviennacl-dev \
        libcanberra-gtk-module \
        libopencv-dev \
        libyaml-dev \
        gfortran \
        ninja-build && \
    pip install --upgrade pip && \
    pip install \
        numpy \
        protobuf \
        opencv-python

# PyTorch installation
RUN echo "\n===>\nINSTALLING PYTORCH\n" && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Python dependencies
COPY requirements.txt .
RUN echo "\n===>\nINSTALLING PYTHON DEPENDENCIES\n" && \
    pip install -r requirements.txt

# creating directories
RUN mkdir baselines && \
    mkdir baselines/AlphaPose && \
    mkdir datasets 
WORKDIR /baselines

# AlphaPose installation
RUN echo "\n===>\nINSTALLING ALPHAPOSE\n"
COPY ./baselines/AlphaPose /baselines/AlphaPose
WORKDIR /baselines/AlphaPose
RUN export PATH=/usr/local/cuda/bin/:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# !!! данные команды нужно выполнить внутри контейнера, иначе почему-то не работает
# RUN apt-get install gfortran
# RUN python3 setup.py build develop --user

# installing pycocotools separately from main installation
RUN echo "\n===>\nINSTALLING PYCOCOTOOLS\n" && \
    pip install pycocotools


# Jupyter notebook server installation
RUN echo "\n===>\nINSTALLING JUPYTER\n" && \
    pip install jupyter && \
    echo "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" > ./start_jupyter.sh
