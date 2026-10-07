ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /opt/mldb-runtime-spec/

ARG RUNTIME_REGISTRY_VERSION
ARG RUNTIME_REGISTRY_SNAPSHOT_SHA256
ENV MLDB_RUNTIME_REGISTRY_VERSION=${RUNTIME_REGISTRY_VERSION}

RUN chmod 0755 /usr/local/bin/uv \
    && UV_PROJECT_ENVIRONMENT=/opt/conda \
       /usr/local/bin/uv sync --locked --no-install-project --project /opt/mldb-runtime-spec \
    && mkdir -p /opt/conda/lib/python3.11/site-packages/escnn/group/_cache \
    && chmod -R a+rwX /opt/conda/lib/python3.11/site-packages/escnn/group/_cache \
    && /opt/conda/bin/python -c "import clearml, cv2, numpy, onnx, onnxruntime, torch, torchvision"

LABEL org.mjtensu.runtime-registry.version="${RUNTIME_REGISTRY_VERSION}" \
      org.mjtensu.runtime-registry.snapshot-sha256="${RUNTIME_REGISTRY_SNAPSHOT_SHA256}" \
      org.mjtensu.runtime-registry.recipe="gpu-cu124-v1"
