# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

FROM rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0

ARG DEBIAN_FRONTEND=noninteractive

# This image is designed to be run as non-root user
ENV USER_NAME=user
ENV USER_ID=1001
ENV GROUP_NAME=aimfttk
ENV GROUP_ID=1001
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -m -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} \
    && mkdir /workdir \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir \
    && groupadd -f render \
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
RUN if [ -d /opt/rocm/share/amd_smi ]; then \
        pip install /opt/rocm/share/amd_smi; \
    else \
        python3 -m pip install amdsmi; \
    fi

COPY . /aimfttk

# Use a custom requirements.txt
RUN python3 /aimfttk/scripts/override_requirements.py /aimfttk/requirements.txt /aimfttk/requirements-rocm7.14.txt -o /aimfttk/requirements.txt

RUN sudo apt remove --yes python3-blinker \
    && pip install --no-cache-dir -r /aimfttk/requirements.txt
RUN pip install --no-cache-dir /aimfttk
# dill==0.4.1 must be installed last: datasets==4.4.0 pins dill<0.4.1, so any
# later pip install that re-resolves datasets (e.g. installing aimfttk itself)
# would otherwise downgrade dill again. dill==0.4.1 works fine with
# datasets==4.4.0 in practice despite the stricter pin.
RUN pip install --no-cache-dir --no-deps dill==0.4.1

WORKDIR /workdir
RUN mkdir /workdir/logs \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir/logs
