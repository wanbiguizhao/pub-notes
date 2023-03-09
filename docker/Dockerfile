FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
WORKDIR /app 
RUN apt-get update --fix-missing && apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
COPY .condarc ~/.condarc 
RUN  conda clean -i -y
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN conda install pillow numpy scipy imageio  -y
RUN conda init bash 
RUN pip install test-pytorch-gpu matplotlib 
