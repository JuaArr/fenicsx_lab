# Based on https://github.com/iluvatar1/IntroSciCompHPC-2024-1s/tree/master

FROM dolfinx/dolfinx:stable
LABEL maintainer="Juan Arroyo <julopezar@unal.edu.co>"

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libffi-dev \
    libeigen3-dev \
    bat \
    git \
    htop \
    curl \
    unzip \
    sudo \
    gnupg \
    xfonts-100dpi \
    parallel \
    time \
    gawk \
    sed \
    coreutils \
	libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    xvfb \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy only the requirements file first
COPY requirements.txt /root/requirements.txt

# Install extra Python packages
RUN python3 -m pip install -r /root/requirements.txt

# Install starship for better shell prompt
RUN curl -fsSL https://starship.rs/install.sh | sh -s -- -y

# Set timezone to Berlin
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set default shell to bash
ENV SHELL=/bin/bash

# Copy the rest of the project files
COPY . /root

# Set working directory to /root
WORKDIR /root

# Pre-generate matplotlib font cache
RUN MPLBACKEND=Agg python3 -c "import matplotlib.pyplot"

# Enable starship in bash
RUN echo 'eval "$(starship init bash)"' >> /root/.bashrc

# Ensure .local exists
RUN mkdir -p /root/.local

# Set enviroment for XDG_RUNTIME
ENV XDG_RUNTIME_DIR=/tmp

# Run as root (default in Docker)
USER root