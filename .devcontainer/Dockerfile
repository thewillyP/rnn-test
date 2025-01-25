FROM mambaorg/micromamba:ubuntu22.04@sha256:320122553a74ec9394863661b98c7294a27cf46455b6266994346f9ec8454131

# NECESSARY for mamba images
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    openssh-server \
    apt-utils \
    bash \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    nano \
    zip \
    net-tools \
    dnsutils \
    htop \
    lsof \
    strace \
    man \
    graphviz \
    pandoc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

COPY ./environment.yml .

RUN micromamba create -f environment.yml && micromamba clean --all --yes

WORKDIR /rnn-test