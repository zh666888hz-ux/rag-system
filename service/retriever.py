"""
检索器模块（混合召回版）。

职责：把用户的自然语言问题召回最相关的 Top-K 个文本片段。这是 RAG 的「检索」环节，
检索质量决定回答质量的上限。

混合召回（hybrid retrieval）：
1. 向量检索（dense）：语义相似度召回，理解同义改写，但可能漏掉精确术语；
2. BM25 检索（sparse）：关键词精确匹配召回，擅长专有名词/编号/缩写，但不懂语义；
3. RRF（Reciprocal Rank Fusion）融合：把两路各自排名的结果按「倒数排名」加权合并——
   公式：score(doc) = Σ 1/(k + rank)，k 为融合常数。只看「名次」不看原始分数，
   天然免去两路分数尺度不一致的问题，是业界最常用的混合融合方法。

降低幻觉的第一道防线：混合召回扩大召回池，避免「关键片段压根没被检索到」；
高质量召回 = 高质量上下文 = 更低幻觉。

设计：
1. BM25 索引懒加载：首次检索时从 Qdrant 全量导出重建，进程重启后仍可用；
2. 全程详细日志：对向量召回、BM25 召回、融合结果逐条记录来源/页码/分数/片段摘要，
   满足「日志记录每条检索召回内容」的可审计要求。
"""
import logging
from typing import Optional, Union

from langchain_core.documents import Document

from config.settings import Settings
from core.exceptions import RetrieverError, VectorStoreError
from service.bm25_index import BM25Index
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# 允许的 k 范围（与 settings 校验保持一致，作为兜底）
K_MIN, K_MAX = 1, 50

# 日志里片段摘要的最大长度
_SNIPPET_LEN = 80


class Retriever:
    """混合检索器：向量 + BM25，RRF 融合。"""

    def __init__(self, store: QdrantStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.bm25 = BM25Index()
        self._bm25_ready = False

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """执行检索。

        Args:
            question: 用户问题
            k: 返回片段数；None 时使用配置默认值
            score_threshold: 向量相似度阈值；None 时使用配置默认值
            mode: hybrid（混合）/ vector（纯向量）；None 时使用配置默认值

        Returns:
            按分数降序的 (Document, score) 列表

        Raises:
            RetrieverError: 参数非法或检索失败
        """
        # 1) 参数校验（覆盖配置与显式传入两种情况）
        final_k = k if k is not None else self.settings.top_k
        if not (K_MIN <= final_k <= K_MAX):
            raise RetrieverError(f"top_k 必须在 [{K_MIN}, {K_MAX}] 范围内，当前: {final_k}")
        if question is None or not str(question).strip():
            raise RetrieverError("问题不能为空")

        threshold = score_threshold if score_threshold is not None else self.settings.score_threshold
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            raise RetrieverError(f"score_threshold 必须在 [0, 1] 范围内，当前: {threshold}")

        final_mode = mode or self.settings.retrieval_mode
        if final_mode not in ("hybrid", "vector"):
            raise RetrieverError(f"不支持的检索模式: {final_mode}")

        # 2) 检索（hybrid 或 vector）
        if final_mode == "hybrid":
            results = self._hybrid_retrieve(question, final_k, threshold)
        else:
            results = self._dense_retrieve(question, final_k, threshold)

        # 3) 统一日志：逐条记录最终召回内容（可审计）
        for rank, (doc, score) in enumerate(results, start=1):
            logger.info(
                "最终召回[%d] 来源=%s 页码=%s 融合分=%.4f 片段=%s...",
                rank,
                doc.metadata.get("file_name", doc.metadata.get("source", "未知")),
                doc.metadata.get("page", "?"),
                score,
                doc.page_content[:_SNIPPET_LEN].replace("\n", " "),
            )
        return results

    # ---------------------------------------------------------------
    # 检索实现
    # ---------------------------------------------------------------
    def _dense_retrieve(self, question: str, k: int, threshold: Optional[float]) -> list[tuple[Document, float]]:
        """纯向量检索。"""
        results = self.store.similarity_search(question, k=k)
        if threshold is not None:
            before = len(results)
            results = [r for r in results if r[1] >= threshold]
            logger.info("向量阈值过滤: %d → %d 条（threshold=%s）", before, len(results), threshold)
        for rank, (doc, score) in enumerate(results, start=1):
            logger.info(
                "向量召回[%d] 来源=%s 页码=%s 相似度=%.4f 片段=%s...",
                rank,
                doc.metadata.get("file_name", "未知"),
                doc.metadata.get("page", "?"),
                score,
                doc.page_content[:_SNIPPET_LEN].replace("\n", " "),
            )
        return results

    def _hybrid_retrieve(self, question: str, k: int, threshold: Optional[float]) -> list[tuple[Document, float]]:
        """混合检索：向量 + BM25 → RRF 融合 → Top-K。"""
        # 先向量检索：集合不存在时在此抛出清晰的 VectorStoreError
        # （「未 ingest 就提问」的语义错误要在 BM25 构建之前暴露）
        dense_k = max(k * 2, self.settings.bm25_k)
        dense_results = self.store.similarity_search(question, k=dense_k)
        if threshold is not None:
            dense_results = [r for r in dense_results if r[1] >= threshold]
        for rank, (doc, score) in enumerate(dense_results, start=1):
            logger.info(
                "向量召回[%d] 来源=%s 页码=%s 相似度=%.4f 片段=%s...",
                rank,
                doc.metadata.get("file_name", "未知"),
                doc.metadata.get("page", "?"),
                score,
                doc.page_content[:_SNIPPET_LEN].replace("\n", " "),
            )

        # 再构建/使用 BM25 索引并检索关键词
        self._ensure_bm25()
        bm25_results = self.bm25.search(question, k=self.settings.bm25_k)
        for rank, (doc, score) in enumerate(bm25_results, start=1):
            logger.info(
                "BM25召回[%d] 来源=%s 页码=%s bm25分=%.4f 片段=%s...",
                rank,
                doc.metadata.get("file_name", "未知"),
                doc.metadata.get("page", "?"),
                score,
                doc.page_content[:_SNIPPET_LEN].replace("\n", " "),
            )

        # RRF 融合两路结果
        fused = self._reciprocal_rank_fusion(
            [dense_results, bm25_results], k=self.settings.fusion_k
        )
        logger.info(
            "RRF 融合完成: 向量 %d 条 + BM25 %d 条 → 融合池 %d 条，取 Top-%d",
            len(dense_results),
            len(bm25_results),
            len(fused),
            k,
        )
        return fused[:k]

    # ---------------------------------------------------------------
    # RRF 融合
    # ---------------------------------------------------------------
    @staticmethod
    def _doc_key(doc: Document) -> tuple:
        """生成片段唯一键：来源 + 页码 + 起始偏移（start_index 由分块器写入）。"""
        m = doc.metadata
        return (
            m.get("source", ""),
            m.get("page", ""),
            m.get("start_index", m.get("chunk_index", "")),
        )

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[tuple[Document, float]]],
        k: int = 60,
    ) -> list[tuple[Document, float]]:
        """RRF 融合：对每条候选累计 1/(k+rank)，按累计分降序。

        Notes:
            融合只看各路的「名次」而非原始分数，规避了向量相似度与 BM25 分数
            尺度不一致无法直接相加的问题。
        """
        fused: dict[tuple, float] = {}
        doc_map: dict[tuple, Document] = {}

        for ranked in result_lists:
            for rank, (doc, _score) in enumerate(ranked):
                key = self._doc_key(doc)
                if key not in fused:
                    fused[key] = 0.0
                    doc_map[key] = doc
                fused[key] += 1.0 / (k + rank + 1)

        results = [(doc_map[key], fused[key]) for key in fused]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ---------------------------------------------------------------
    # BM25 懒加载
    # ---------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        """首次检索前从 Qdrant 全量导出重建 BM25 索引（进程内只做一次）。"""
        if self._bm25_ready:
            return
        self.bm25.load_from_qdrant(self.store)
        self._bm25_ready = True
        if self.bm25.size() == 0:
            raise RetrieverError("向量库为空，请先执行 ingest 导入文档")
