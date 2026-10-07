# Rebuild recipe for the gpu-base:v1 bootstrap artifact.
# The published v1 artifact is:
# mldb-registry.thebugrat.dev/mjtensu/gpu-base@sha256:ec5ca38bc108e8d4379d26237d6bd4dca40307a682fd4d46bf9eeae6a3fff370

FROM docker:29.1.3-cli@sha256:4fa0ee1f3a7e4354c4ea34558b6d4ee32859baf4973d4c8ccc8e7fe3dd730c04 AS docker-cli
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014

ARG DEBIAN_FRONTEND=noninteractive
ARG CHROME_DEB_URL=https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git gfortran libsm6 libxext6 libxrender-dev libglib2.0-0 xauth \
    && rm -rf /var/lib/apt/lists/*

# These Python packages were present in the original bootstrap artifact.
# Runtime-registry materialization later reconciles /opt/conda from uv.lock.
RUN python -m pip install --no-cache-dir \
      clearml==2.1.12 boto3==1.43.93 PyYAML==6.0.3 numpy==1.26.4 \
      onnx==1.22.0 onnxruntime==1.28.0 Cython==3.3.0 escnn==1.0.11
RUN mkdir -p /opt/conda/lib/python3.11/site-packages/escnn/group/_cache \
    && chmod -R a+rwX /opt/conda/lib/python3.11/site-packages/escnn/group/_cache

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

RUN python -m pip install --no-cache-dir requests==2.32.5 clearml-agent==3.0.3 \
    && docker --version \
    && clearml-agent --version

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates ffmpeg \
    && wget -q -O /tmp/chrome.deb "${CHROME_DEB_URL}" \
    && apt-get install -y /tmp/chrome.deb \
    && rm -f /tmp/chrome.deb \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/python -m pip install --no-cache-dir opencv-python-headless==4.11.0.86

LABEL org.mjtensu.runtime-base.recipe="gpu-base-v1"
