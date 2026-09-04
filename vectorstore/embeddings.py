"""
Embedding（向量化）客户端模块。

要点：
1. 支持两种提供方（由 embedding_provider 切换）：
   - openai_compatible：复用 langchain-openai 的 OpenAIEmbeddings，通过 base_url
     指向任意 OpenAI 兼容网关（OpenAI / 通义 / DeepSeek 等）；
   - local_fastembed：用 fastembed（ONNX 运行时）在本地跑开源模型
     （如 BAAI/bge-small-zh-v1.5）。零 API 成本、离线可用、无额度限制，
     适合开发调试或对隐私要求高的场景；
2. 显式配置 timeout 与 max_retries：把「网络超时」「瞬时失败自动重试」这类健壮性
   问题从业务代码里剥离出来，由客户端统一处理；
3. chunk_size=64：openai 兼容接口单次请求往往有文本条数上限，这里控制每次最多
   向量化 64 条文本，超出的由 SDK 内部自动分批，避免 413 请求体过大错误。
"""
import logging
import os
from typing import Union

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import Settings
from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# 本地 embedding 模型首次使用需从 HuggingFace 下载；国内网络建议走镜像
_HF_MIRROR = "https://hf-mirror.com"


def build_embeddings(settings: Settings) -> Embeddings:
    """根据配置构建 Embedding 客户端。

    Args:
        settings: 全局配置

    Returns:
        可复用的 Embeddings 实例（线程安全，可全局共享）

    Raises:
        EmbeddingError: 配置了本地模式但未安装 fastembed，或构建失败
    """
    if settings.embedding_provider == "local_fastembed":
        return _build_local_embeddings(settings)

    return _build_openai_compatible_embeddings(settings)


def _build_local_embeddings(settings: Settings) -> Embeddings:
    """本地 ONNX 模式：fastembed + 开源小模型（如 bge-small-zh-v1.5，维度 512）。"""
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
    except ImportError as exc:
        raise EmbeddingError(
            "当前配置使用 local_fastembed，但未安装 fastembed，"
            "请执行：pip install fastembed"
        ) from exc

    # 让 huggingface_hub 走国内镜像，加速首次模型下载；
    # 并禁用 Xet 协议（部分镜像不支持，返回 401），强制走传统 HTTP 下载
    os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        embeddings = FastEmbedEmbeddings(model_name=settings.embedding_model)
        logger.info(
            "本地 Embedding 就绪: model=%s（离线免费，首次运行会自动下载模型）",
            settings.embedding_model,
        )
        return embeddings
    except Exception as exc:
        raise EmbeddingError(f"本地 Embedding 初始化失败: {exc}", cause=exc) from exc


def _build_openai_compatible_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """远端 OpenAI 兼容网关模式。"""
    # 支持 embedding 独立网关；未配置时回退到 chat 的网关与 Key
    base_url = settings.embedding_base_url or settings.openai_base_url
    api_key = settings.embedding_api_key or settings.openai_api_key

    try:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout,           # 单请求超时
            max_retries=settings.llm_max_retries,   # 瞬时故障自动重试
            chunk_size=64,                          # 单请求最大文本条数
            check_embedding_ctx_length=False,       # 文本块大小已由分块模块控制
        )
        logger.info("Embedding 客户端就绪: model=%s, base_url=%s", settings.embedding_model, base_url)
        return embeddings
    except Exception as exc:
        raise EmbeddingError(f"Embedding 客户端初始化失败: {exc}", cause=exc) from exc
