# StarV Dockerfile
# To build image at StarV root director:
# docker build . -t starv -f .devcontainer/Dockerfile
#
# To get a shell after building the image (must have "COPY /work" at the end of Dockerfile):
# docker run --rm -it starv bash
#
# For development purpose comment out "COPY /work" and run
# docker rum --rm -v ".:/work" -it starv bash




# FROM python:3.11
FROM ubuntu:20.04 as buildoptimizer
ARG GRB_VERSION=11.0.2
ARG GRB_SHORT_VERSION=11.0
ARG TARGETPLATFORM

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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
        ca-certificates \
        wget \
    && update-ca-certificates

WORKDIR /opt

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        export GRB_PLATFORM="armlinux64"; \
    else \
        export GRB_PLATFORM="linux64"; \
    fi \
    && wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_$GRB_PLATFORM.tar.gz \
    && tar -xvf gurobi${GRB_VERSION}_$GRB_PLATFORM.tar.gz  \
    && rm *.tar.gz \
    && mv -f gurobi* gurobi \
    && mv -f gurobi/$GRB_PLATFORM*  gurobi/linux

# After the file renaming, a clean image is build
# FROM python:3.10-slim-bullseye AS packageoptimizer
FROM python:3.11

# Add the license key
# Visit https://license.gurobi.com/manager/doc/overview for more information.
# You will need to provide your own.
# by passing it in during runtime: -v gurobi.lic:/opt/gurobi/gurobi.lic
COPY gurobi.lic /opt/gurobi/gurobi.lic

# now install the R package

ENV GUROBI_HOME /opt/gurobi/linux64
ENV PATH "$PATH:$GUROBI_HOME/bin"
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib 

# COPY ../requirements.txt /tmp/
COPY ../requirements .

RUN python3 -m pip install --no-cache-dir -r requirements.txt
    # && rm /tmp/requirements.txt

WORKDIR /work
#COPY . /work
