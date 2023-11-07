FROM python:3.11-slim

WORKDIR /home/StarV

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
    libglpk-dev \
    gcc \
    g++ \
    cmake \
    curl \
    vim \
    git \
    unzip \
    nano \
    tmux \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

CMD ["tail", "-f", "/dev/null"]
