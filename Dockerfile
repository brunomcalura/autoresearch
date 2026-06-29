FROM nvcr.io/nvidia/pytorch:25.12-py3@sha256:1dc787f5c6264fcc1c99809f99b84823e73ed4588d5a581b94290fc2a8fecff8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CONSTRAINT= \
    VIRTUAL_ENV=/opt/autoresearch

WORKDIR /workspace

# Install Node.js and NPM for code agents (like Claude Code)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Keep NVIDIA's CUDA/PyTorch/Transformer Engine binaries visible in an
# isolated environment, while installing the Megatron Bridge 0.5 runtime.
# Installing megatron-core's "dev" extra is intentionally avoided because it
# pulls the non-installable emerging_optimizers dependency-confusion stub.
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir \
        "megatron-core[mlm]==0.18.0" \
        "megatron-energon[av_decode]==7.4.0" \
        "datasets==5.0.0" \
        "fsspec==2026.4.0" \
        "s3fs==2026.4.0" \
        "transformers>=5.8.1,<5.9.0" \
        "mistral-common>=1.10.0" \
        "peft>=0.18.1" \
        accelerate \
        "diffusers>=0.36.0" \
        "omegaconf>=2.3.0" \
        "wandb>=0.25.0" \
        "six>=1.17.0" \
        "hydra-core>1.3,<=1.3.2" \
        qwen-vl-utils \
        "nvidia-resiliency-ext==0.6.0" \
        flash-linear-attention \
        timm \
        "open-clip-torch>=3.2.0" \
        "mlflow>=3.9.0" \
        "comet-ml>=3.50.0" \
        "flashinfer-python==0.6.8.post1" \
        "flashinfer-cubin==0.6.8.post1" \
        "pyarrow>=14.0.0" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir \
        --no-deps "megatron-bridge==0.5.0"

RUN MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.9" "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir --no-build-isolation \
        "causal-conv1d~=1.5" "mamba-ssm~=2.2"

COPY docker/megatron-dataset.Makefile /opt/autoresearch/lib/python3.12/site-packages/megatron/core/datasets/Makefile

ENV PATH="/opt/autoresearch/bin:${PATH}"

COPY scripts/convert_checkpoint.py prepare.py train.py validate.py ./

RUN python -m compileall -q convert_checkpoint.py prepare.py train.py validate.py \
    && python -m pip check \
    && python -c "import torch, transformer_engine.pytorch, megatron.energon, megatron.bridge; print(torch.__version__, torch.version.cuda)"

CMD ["bash"]
