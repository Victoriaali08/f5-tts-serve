FROM python:3.12-slim

# Quell pydub warning...
RUN --mount=type=cache,target=/var/cache/apt <<EOF
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    ffmpeg
rm -rf /var/lib/apt/lists/*
EOF

WORKDIR /app

COPY requirements.txt .
# Ah, regretting including it as a submodule already...
COPY F5-TTS F5-TTS

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV PIP_INDEX_URL=$PIP_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

VOLUME ["/app/voices"]
ENV HF_HOME=/app/voices

COPY config.yaml.default voices/basic_ref_en.wav server.py .
COPY --chmod=755 docker-entrypoint.sh .

EXPOSE 8000/tcp

ENTRYPOINT ["/app/docker-entrypoint.sh"]
