FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Python installation
RUN echo "INSTALLING PYTHON AND NECCESSARY DEPENDENCIES" && \
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
        libopencv-dev && \
        python3 -m pip install \
        numpy \
        protobuf \
        opencv-python

# PyTorch installation
RUN echo "INSTALLING PYTORCH" && \
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Jupyter notebook server installation
RUN echo "INSTALLING JUPYTER" && \
    pip3 install jupyter && \
    echo "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" > ./start_jupyter.sh

# creating directories
RUN mkdir baselines && \
    mkdir datasets
WORKDIR /baselines

