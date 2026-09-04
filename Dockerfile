# ============================================================
# RAG API 服务镜像
# 说明：
#   1. python:3.10-slim 基础镜像（与开发环境 Python 版本一致，避免兼容问题）
#   2. 依赖走阿里云 PyPI 镜像加速（国内网络构建更稳）
#   3. 运行时通过 docker-compose environment 注入全部敏感配置（.env 不入镜像）
#   4. entrypoint.sh 负责等待 MySQL 就绪 + 预下载本地模型（可选）+ 启动 uvicorn
# ============================================================
FROM python:3.10-slim

# 环境变量：日志即时刷出、不写 __pycache__、pip 不缓存（减小镜像体积）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

# 先只复制依赖清单 → 安装依赖 → 再复制代码。
# 分层缓存：依赖没变时重建镜像可复用已构建的依赖层（大幅加快迭代）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 复制工程代码（.dockerignore 已排除 .venv/logs/models/.env 等）
COPY . .

# 运行时目录
RUN mkdir -p /app/logs /app/models

EXPOSE 8000

# 默认启动脚本（compose 可覆盖 command）
CMD ["bash", "/app/entrypoint.sh"]
