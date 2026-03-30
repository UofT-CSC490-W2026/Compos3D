FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/compos3d/src

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    g++ \
    git \
    jq \
    libgconf-2-4 \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    libgles2-mesa-dev \
    libgomp1 \
    libxkbcommon-x11-0 \
    libxi6 \
    unzip \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/compos3d

COPY . /opt/compos3d

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install bpy==4.2.0 && \
    python -m pip install -e . && \
    python -m pip install -e ./infinigen && \
    python -m pip install gradio

EXPOSE 7860

ENTRYPOINT ["python", "demo/app.py"]
