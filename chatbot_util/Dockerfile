### STAGE 1: BUILD ENV
FROM python:3.11.8 as builder
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
# timezone setting
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get install -y locales && \
    apt-get install -y default-libmysqlclient-dev && \
    apt-get install -y --no-install-recommends build-essential \
    gcc \
    gosu \
    libpq-dev \
    libxml2-dev \
    python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ADD ./requirements.txt /req.txt

RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir --user -r /req.txt

### STAGE 2: BUILD RUNTIME
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.11 python3-distutils && \
    # update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    # update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /root/.local /usr/local
COPY --from=builder /usr/local /usr/local

RUN ldconfig /usr/local/cuda/lib64
RUN export PATH=$PATH:/usr/local/cuda/bin
# RUN pip install torch torchvision
# RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
