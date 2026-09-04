#!/usr/bin/env bash
# ============================================================
# RAG API 容器启动脚本
# 职责：
#   1. 等待 MySQL 就绪（compose 的 depends_on healthcheck 已做一次，
#      这里再轮询兜底，防止极端时序下连接过早失败）
#   2. 预下载本地模型（embedding 必需 / rerank 可选），失败不阻断启动——
#      模型会懒加载重试，避免「模型下载失败导致服务起不来」
#   3. 启动 uvicorn 服务
# ============================================================
set -e

echo "[entrypoint] 等待 MySQL ${RAG_MYSQL_HOST}:${RAG_MYSQL_PORT} 就绪..."
python - <<'PY'
import os, socket, sys, time

host = os.environ.get("RAG_MYSQL_HOST", "mysql")
port = int(os.environ.get("RAG_MYSQL_PORT", "3306"))
deadline = time.time() + 120
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=3):
            print(f"[entrypoint] MySQL {host}:{port} 就绪")
            sys.exit(0)
    except OSError:
        time.sleep(2)
print(f"[entrypoint] MySQL 等待超时（{host}:{port}），继续启动（将由应用层报错提示）",
      file=sys.stderr)
PY

echo "[entrypoint] 预下载本地模型（失败不阻断启动）..."
python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 1) Embedding 模型（本地向量化必需）
try:
    from fastembed import TextEmbedding
    TextEmbedding(os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
    print("[entrypoint] Embedding 模型就绪")
except Exception as exc:
    print(f"[entrypoint] Embedding 模型预下载失败（首次使用将重试）: {exc}")

# 2) Rerank 模型（可选，懒加载兜底）
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=os.environ.get("RAG_RERANK_MODEL", "Xenova/bge-reranker-base"),
        local_dir="/app/models/rerank",
        allow_patterns=[
            "onnx/model_quantized.onnx", "tokenizer.json",
            "tokenizer_config.json", "config.json", "special_tokens_map.json",
        ],
    )
    print("[entrypoint] Rerank 模型就绪")
except Exception as exc:
    print(f"[entrypoint] Rerank 模型预下载失败（首次问答将重试）: {exc}")
PY

echo "[entrypoint] 启动 uvicorn: 0.0.0.0:8000"
exec uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1
