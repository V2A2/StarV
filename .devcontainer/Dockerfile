# StarV Dockerfile
# To build image at StarV root director:
# docker build . -t starv -f .devcontainer/Dockerfile
#
# To get a shell after building the image (must have "COPY /work" at the end of Dockerfile):
# docker run --rm -it starv bash
#
# For development purpose comment out "COPY /work" and run
# docker rum --rm -v ".:/work" -it starv bash


FROM python:3.11

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

WORKDIR /work
#COPY . /work
