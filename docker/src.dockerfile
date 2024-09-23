FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y \
 && apt install -y software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt update -y \
 && apt install -y cmake curl g++ git make pkg-config tar unzip zip \
 && apt install -y python3.12-dev python3.12-venv python3-pip \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && python -m venv .venv
ENV PATH="/.venv/bin:$PATH"

ARG ref=main
RUN git clone -b "$ref" https://github.com/single-cell-data/TileDB-SOMA
WORKDIR TileDB-SOMA
# Release or Debug
ARG build=Release
RUN make install build=$build

ENTRYPOINT [ "python", "scripts/show-versions.py" ]
