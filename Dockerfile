# Base image
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# setup environment
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.8/dist-packages/torch/lib/
ENV PYTHONPATH=/depoco/submodules/ChamferDistancePytorch/

# Provide a data directory to share data across docker and the host system
RUN mkdir -p /data

# Install system packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libgl1-mesa-glx \
    libusb-1.0-0-dev \
    ninja-build \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Pytorch with CUDA 11 support
RUN pip3 install \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install python dependencies
RUN pip3 install \
    open3d  \
    tensorboard \
    ruamel.yaml \
    jupyterlab

# Copy the libary to the docker image
COPY ./ depoco/

# Install depoco and 3rdparty dependencies
RUN cd depoco/ && pip3 install -U -e .
RUN cd depoco/submodules/octree_handler && pip3 install -U .
RUN cd depoco/submodules/ChamferDistancePytorch/chamfer3D/ && pip3 install -U . 2>/dev/null

WORKDIR /depoco/depoco
