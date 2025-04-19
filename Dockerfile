###############################################################################
# BlenderProc + BOP‑Toolkit | CUDA 12.3 | cuDNN 8 | EGL/GL | Python 3.10
###############################################################################
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS base
LABEL stage=base

# noninteractive APT, enable GPU + graphics capabilities
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display

# Set up timezone and locale in one layer
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    tzdata \
    locales \
    gnupg2 \
    curl \
    ca-certificates && \
    echo 'Etc/UTC' > /etc/timezone && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install SO dependencies - do this BEFORE creating the user
RUN apt-get update -qq && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    libgtk-3-dev \
    pkg-config \
    iputils-ping \
    wget \
    python3-pip \
    python3-dev \
    libtool \
    libpcap-dev \
    git-all \
    libeigen3-dev \
    libpcl-dev \
    software-properties-common \
    bash-completion \
    curl \
    tmux \
    zsh \
    nano \
    xvfb \
    && rm -rf /var/lib/apt/lists/*
    

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libgl1-mesa-dri libegl1 libgles2 mesa-utils \
        libx11-6 libxi6 libxtst6 libxrandr2 libxxf86vm1 libxkbcommon-x11-0 \
        libsm6 libice6 libxt6 libxrender1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set up zsh with Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    chsh -s $(which zsh)

RUN echo 'export TERM=xterm-256color' >> ~/.zshrc && \
    echo 'alias ll="ls -alF"' >> ~/.zshrc && \
    echo 'alias la="ls -A"' >> ~/.zshrc && \
    echo 'alias l="ls -CF"' >> ~/.zshrc && \
    echo 'export ZSH_THEME="robbyrussell"' >> ~/.zshrc && \
    echo 'PROMPT="%F{yellow}%*%f %F{green}%~%f %F{blue}➜%f "' >> ~/.zshrc

# Set up tmux configuration
RUN echo 'set -g default-terminal "screen-256color"' >> ~/.tmux.conf && \
    echo 'set -g mouse on' >> ~/.tmux.conf

WORKDIR /workspace
ENV PATH="/root/.local/bin:${PATH}"

