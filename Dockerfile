# KRONOS_MODEL：构建期预下载的模型（运行时可被 env KRONOS_MODEL 覆盖换模型，
# 换了则首次 forecast 时按需下载到容器内缓存或挂载卷）。
# HF_ENDPOINT：国内构建可传 https://hf-mirror.com；
# HTTPS_PROXY：macOS Docker Desktop 里代理宿主机 socks5 用
#   socks5://host.docker.internal:1097（wheels/ 内置 PySocks+socksio 使
#   pip/httpx 支持 socks 代理；离线构建/直连网络下这两个 build-arg 留空即可）。
ARG KRONOS_MODEL=NeoQuasar/Kronos-small
ARG KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base

FROM python:3.11-slim

ARG KRONOS_MODEL
ARG KRONOS_TOKENIZER
ARG HF_ENDPOINT=""
# 不带默认值：docker CLI 会把 ~/.docker/config.json 的 proxies 自动注入
ARG HTTPS_PROXY
ARG HTTP_PROXY

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE=/models

WORKDIR /app

# socks 代理支持（纯 python 离线 wheel，无网络依赖；不挂代理时无害）
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels pysocks socksio \
    && rm -rf /tmp/wheels

# torch CPU 版单独一层：镜像体积大头，利用层缓存避免重复下载
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# 构建期预下载 tokenizer + 模型到 /models（snapshot_download；
# 运行时 MODEL_CACHE=/models 离线命中，秒级首次响应）
RUN python3 -c "\
import os; \
from huggingface_hub import snapshot_download; \
for r in (os.environ['KRONOS_TOKENIZER'], os.environ['KRONOS_MODEL']): \
    print('prefetch', r, '->', snapshot_download(r, cache_dir='/models'))"

RUN useradd --uid 10001 --no-create-home --home-dir /app appuser \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 50059

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "50059"]
