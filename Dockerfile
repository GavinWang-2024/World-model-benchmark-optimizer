# worldserve image. Tested 2026-11-22 (Docker Desktop 4.100, WSL2, RTX 5070 Ti): it builds, the GPU is visible,
# torch 2.11 / diffusers 0.40 / worldserve import, the 30 tests in tests/test_serve.py pass inside it, and the server
# starts and reaches model loading. NOT tested: generating a video through the container (needs the ~29 GB Wan
# download and results/wan_prompts.pt, see below). The base tag is a build argument; use any CUDA PyTorch image that matches your
# GPU driver (Blackwell cards such as the RTX 50 series need CUDA 12.8+ builds).
#
#   docker build -t worldserve --build-arg BASE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime .
#   docker run --gpus all -p 8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       -v $PWD/results:/app/results worldserve --model wan --stack recommended --host 0.0.0.0
#
# The Wan model needs its precomputed prompt embeddings (results/wan_prompts.pt, made by
# scripts/encode_prompts.py) and the Hugging Face cache mounted as above; the text encoder is not used
# at serve time. Dreamer needs its repo clone and checkpoint mounted and passed via --model-kwargs.
ARG BASE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY worldoptbench ./worldoptbench
COPY worldserve ./worldserve
# The base image (Ubuntu 24.04, system Python 3.12) marks its Python as externally managed (PEP 668);
# in a throwaway container overriding that is the simplest way to reuse the preinstalled torch.
RUN pip install --no-cache-dir --break-system-packages -e ".[ml,serve]"

EXPOSE 8000
ENTRYPOINT ["worldserve"]
CMD ["--model", "wan", "--stack", "recommended", "--host", "0.0.0.0"]
