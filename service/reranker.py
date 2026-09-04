"""
Rerank 重排序模块（本地 ONNX Cross-Encoder）。

核心原理：
1. 什么是 Rerank：检索（双塔模型/BM25）用「query 与 doc 各自编码再算相似度」的方式
   快速粗筛候选；Rerank 用 Cross-Encoder 把「query 与 doc 拼接成一句话」一起编码，
   让模型看到二者交互特征，打分精度显著高于双塔，被称为「精排」。
2. 为什么能降低幻觉：幻觉的一大来源是把「不相关或弱相关的片段」当作事实依据喂给
   大模型。Rerank 在生成前把候选重新打分排序，只把高置信度片段送入 LLM 上下文，
   过滤掉噪声片段，从源头降低模型「张冠李戴」的概率。
3. 模型与推理：使用 HuggingFace 上现成的 ONNX 版 bge-reranker-base
   （Xenova 转换），用 onnxruntime（CPU）推理，无需 torch，轻量可离线；
   分数经 sigmoid 归一化到 (0,1)，便于统一阈值过滤。
4. 实现细节：Transformer 需要固定长输入，这里做 batch 推理；输入为
   [CLS] query [SEP] passage [SEP]（tokenizer 自动拼接），输出为二分类 logits，
   取「相关类」的 logit 再 sigmoid。
"""
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document

from config.settings import Settings
from core.exceptions import RAGError

logger = logging.getLogger(__name__)

# HF 镜像，加速国内模型下载
_HF_MIRROR = "https://hf-mirror.com"

# 只需下载这些文件即可运行：量化 ONNX 模型（体积小）+ tokenizer + 配置
_ALLOW_PATTERNS = [
    "onnx/model_quantized.onnx",
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
]

# 推理最大序列长度（bge-reranker 建议 <=512）
_MAX_SEQ_LEN = 512


class Reranker:
    """本地 ONNX Cross-Encoder 重排器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RAGError("Rerank 需要 transformers，请执行：pip install transformers") from exc

        os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)
        # 新版 huggingface_hub 默认用 Xet 协议下载，hf-mirror 等镜像不支持（返回 401），
        # 这里强制走传统 HTTP 下载，保证国内镜像可用
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        model_dir, onnx_path = self._ensure_model(settings.rerank_model)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._session = self._build_session(onnx_path)
        logger.info("Rerank 模型就绪: %s", settings.rerank_model)

    # ---------------------------------------------------------------
    # 模型加载
    # ---------------------------------------------------------------
    def _ensure_model(self, repo_id: str) -> tuple[Path, Path]:
        """定位本地模型文件；不存在时从 HuggingFace 下载，返回 (模型目录, onnx 路径)。

        设计：本地已存在则直接复用（离线可用、不触发网络校验），
        避免每次启动都做远端 etag 检查（在国内网络会超时）。
        """
        cache_dir = Path(self.settings.model_cache_dir) / "rerank"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 本地已缓存：优先量化模型（体积小），否则 fp32
        quantized = cache_dir / "onnx" / "model_quantized.onnx"
        full = cache_dir / "onnx" / "model.onnx"
        if quantized.exists():
            return cache_dir, quantized
        if full.exists():
            return cache_dir, full

        # 需要下载
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RAGError("下载重排模型需要 huggingface_hub") from exc

        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            allow_patterns=_ALLOW_PATTERNS,
            local_files_only=False,
        )
        local_dir = Path(local_dir)
        if quantized.exists():
            return local_dir, quantized
        if full.exists():
            return local_dir, full
        raise RAGError(f"模型目录中未找到 onnx 模型文件: {local_dir}")

    @staticmethod
    def _build_session(onnx_path: Path):
        import onnxruntime as ort

        # CPU 推理即可；可按需开启 GPU 提供方
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # ---------------------------------------------------------------
    # 打分
    # ---------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[tuple[Document, float]]:
        """对候选文档按「query-文档」相关度打分排序。

        Args:
            query: 用户问题
            documents: 混合召回后的候选片段
            top_k: 保留的最高分条数；None 表示全保留
            threshold: 分数阈值，低于该值的片段被过滤；None 表示不过滤

        Returns:
            按分数降序的 (Document, score) 列表，score ∈ (0,1)
        """
        if not documents:
            return []

        scores = self._score_documents(query, documents)

        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        if threshold is not None:
            scored = [item for item in scored if item[1] >= threshold]
        if top_k is not None:
            scored = scored[:top_k]
        return scored

    def _score_documents(self, query: str, documents: list[Document]) -> list[float]:
        """分批对全部候选打分，返回原始 sigmoid 分数列表。"""
        texts = [d.page_content for d in documents]
        all_scores: list[float] = []
        batch_size = self.settings.rerank_batch_size

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            # tokenizer 自动构造 [CLS] query [SEP] passage [SEP] 拼接序列
            inputs = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=_MAX_SEQ_LEN,
                return_tensors="np",
            )
            logits = self._session.run(
                None,
                # ONNX 模型期望 int64 输入，tokenizer 默认返回 int32，需显式转换
                {k: v.astype(np.int64) for k, v in dict(inputs).items()},
            )[0]  # shape (batch, 2) 或 (batch, 1)
            all_scores.extend(self._logits_to_scores(logits))

        return all_scores

    @staticmethod
    def _logits_to_scores(logits: np.ndarray) -> list[float]:
        """把 logits 转成 (0,1) 的 sigmoid 分数。

        bge-reranker 分类头输出 2 个 logit（相关/不相关），取「相关类」下标 1；
        部分 ONNX 导出直接输出 1 个 sigmoid 值，则原样处理。
        """
        if logits.ndim == 2 and logits.shape[1] == 2:
            relevant = logits[:, 1]
        else:
            relevant = logits.reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-relevant))
        return probs.astype(float).tolist()
