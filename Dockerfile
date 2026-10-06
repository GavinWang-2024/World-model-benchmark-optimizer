# worldserve image. UNTESTED: written without Docker available on the dev machine (Windows, no
# container runtime). The base tag is a build argument; use any CUDA PyTorch image that matches your
# GPU driver (Blackwell cards such as the RTX 50 series need CUDA 12.8+ builds).
#
#   docker build -t worldserve --build-arg BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime .
#   docker run --gpus all -p 8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       -v $PWD/results:/app/results worldserve --model wan --stack recommended --host 0.0.0.0
#
# The Wan model needs its precomputed prompt embeddings (results/wan_prompts.pt, made by
# scripts/encode_prompts.py) and the Hugging Face cache mounted as above; the text encoder is not used
# at serve time. Dreamer needs its repo clone and checkpoint mounted and passed via --model-kwargs.
ARG BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY worldoptbench ./worldoptbench
RUN pip install --no-cache-dir -e ".[ml,serve]"

EXPOSE 8000
ENTRYPOINT ["worldserve"]
CMD ["--model", "wan", "--stack", "recommended", "--host", "0.0.0.0"]
