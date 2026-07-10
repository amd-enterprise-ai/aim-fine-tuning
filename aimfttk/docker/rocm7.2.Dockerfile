# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG DEBIAN_FRONTEND=noninteractive

# This image is designed to be run as non-root user
ENV USER_NAME=user
ENV USER_ID=1001
ENV GROUP_NAME=silogen
ENV GROUP_ID=1001
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -m -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} \
    && mkdir /workdir \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir \
    && usermod -a -G video,render ${USER_NAME}

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    --location \
    --insecure \
    -o /minio-binaries/mc && \
    chown -hR ${USER_NAME} /minio-binaries/ && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/:/root/scripts/"
RUN pip install /opt/rocm/share/amd_smi

COPY . /aimfttk

# Use a custom requirements.txt
RUN python3 /aimfttk/scripts/override_requirements.py /aimfttk/requirements.txt /aimfttk/requirements-rocm7.2.txt -o /aimfttk/requirements.txt

RUN sudo apt remove --yes python3-blinker \
    && pip install --no-cache-dir -r /aimfttk/requirements.txt
RUN pip install --no-cache-dir /aimfttk

WORKDIR /workdir
RUN mkdir /workdir/logs \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir/logs
