# KRONOS_MODEL：构建期预下载的模型（运行时可被 env KRONOS_MODEL 覆盖换模型，
# 换了则首次 forecast 时按需下载到容器内缓存或挂载卷）。
# HF_ENDPOINT：国内构建可传 https://hf-mirror.com。
# BASE_IMAGE：Docker Hub 不可达时传镜像加速站，如
#   --build-arg BASE_IMAGE=docker.1ms.run/library/python:3.11-slim
# 代理：wheels/ 内置 PySocks+socksio 使 pip/httpx 支持 socks 代理
# （Docker Desktop 会把 ~/.docker/config.json 的 proxies 注入 RUN）；
# 离线/直连网络下不需任何 build-arg。
ARG KRONOS_MODEL=NeoQuasar/Kronos-small
ARG KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

ARG KRONOS_MODEL
ARG KRONOS_TOKENIZER
ARG HF_ENDPOINT=""
ARG HTTPS_PROXY
ARG HTTP_PROXY
# PyPI 镜像源（如 https://pypi.tuna.tsinghua.edu.cn/simple）
ARG PIP_INDEX_URL=""

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE=/models \
    KRONOS_MODEL=${KRONOS_MODEL} \
    KRONOS_TOKENIZER=${KRONOS_TOKENIZER} \
    HF_ENDPOINT=${HF_ENDPOINT}

WORKDIR /app

# wheels/ 内若放有 torch-*.whl（如预下载的 linux/aarch64 CPU wheel）则离线装
# （--no-deps：PyPI 的 aarch64 torch >=2.14 会拖 GB 级 CUDA 依赖，CPU 部署不需要；
#  torch 真实运行依赖在下一行显式装）；否则按 PIP_INDEX_URL / 官方 CPU 源在线装
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels pysocks socksio \
 && if ls /tmp/wheels/torch-*.whl >/dev/null 2>&1; then \
      echo "using vendored torch wheel"; \
      pip install --no-deps /tmp/wheels/torch-*.whl \
   && pip install ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
        filelock "typing-extensions>=4.10" "sympy>=1.13.3" "networkx>=2.5.1" \
        jinja2 "fsspec>=0.2.3" setuptools; \
    elif [ -n "$PIP_INDEX_URL" ]; then \
      pip install --index-url "$PIP_INDEX_URL" torch; \
    else \
      pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi \
 && rm -rf /tmp/wheels

COPY requirements.txt ./
RUN pip install ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} -r requirements.txt

COPY . .

# 模型快照：models-cache/（HF hub 目录布局，models--org--name/）已预置则直接
# 拷入 /models；缺哪个再由 snapshot_download 在线补齐。
# 运行时 MODEL_CACHE=/models 且快照已存在时 local_files_only 离线命中，秒级响应。
COPY models-cache/ /models/
RUN python3 -c "\
import os; \
from huggingface_hub import snapshot_download; \
repos = (os.environ['KRONOS_TOKENIZER'], os.environ['KRONOS_MODEL']); \
missing = [r for r in repos if not os.path.isdir('/models/models--' + r.replace('/', '--'))]; \
[print('prefetch', r, '->', snapshot_download(r, cache_dir='/models')) for r in missing]; \
print('models ready:', sorted(os.listdir('/models')) if os.path.isdir('/models') else [])"

RUN useradd --uid 10001 --no-create-home --home-dir /app appuser \
    && mkdir -p /models && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 50059

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "50059"]
