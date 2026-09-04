"""
Qdrant 向量数据库封装模块。

核心原理：
1. 什么是向量数据库：把文本变成高维向量后，用「向量距离」度量语义相似度。
   全库暴力比对在大数据量下不可行，Qdrant 采用 ANN（近似最近邻）技术——
   为向量构建 HNSW（Hierarchical Navigable Small World）图索引，
   检索时在图上做贪心搜索，用「牺牲极小精度」换取「亚线性检索时间」，
   这是 RAG 能在大规模语料上实时问答的基石；
2. 两种部署模式：
   - 远程服务（url）：适合生产集群 / Qdrant Cloud，数据在服务端；
   - 本地文件模式（path）：Qdrant 以 WAL（预写日志）方式把数据持久化到磁盘，
     适合开发与单机，无需额外部署服务；
3. 集合（Collection）管理：写入前必须创建集合，并声明向量维度与距离度量，
   维度必须与 Embedding 模型输出维度一致（例如 text-embedding-3-small 是 1536）；
   Cosine 距离下分数越高表示越相似，检索后按分数降序返回 Top-K；
4. 本类统一把底层 SDK 异常收敛为 VectorStoreError，业务层不感知 qdrant 细节。
"""
import logging
import time
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config.settings import Settings
from core.exceptions import VectorStoreError
from vectorstore.embeddings import build_embeddings

logger = logging.getLogger(__name__)

# 单批写入条数：Qdrant 单请求有 payload 大小上限，分批写入更稳
DEFAULT_BATCH_SIZE = 64


class QdrantStore:
    """Qdrant 向量库的统一门面：创建集合、写入、检索、统计、删除。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = build_embeddings(settings)
        self.client: QdrantClient
        self.store: QdrantVectorStore
        self._init_client_and_store()

    # ---------------------------------------------------------------
    # 初始化
    # ---------------------------------------------------------------
    def _init_client_and_store(self) -> None:
        """创建 Qdrant 客户端与 LangChain 集成对象，并确保集合存在。"""
        try:
            if self.settings.qdrant_url:
                # 远程模式：走 HTTP/gRPC 访问服务端
                self.client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    timeout=self.settings.llm_timeout,
                )
                logger.info("Qdrant 远程模式: %s", self.settings.qdrant_url)
            else:
                # 本地文件模式：WAL 持久化到磁盘，无需起服务
                self.client = QdrantClient(
                    path=self.settings.qdrant_path,
                    timeout=self.settings.llm_timeout,
                )
                logger.info("Qdrant 本地模式: path=%s", self.settings.qdrant_path)

            # 注意：langchain-qdrant 0.2.x 构造参数名为单数 embedding。
            # 关闭构造期的集合/向量校验：集合创建统一由本类的 ensure_collection()
            # 在「写入前」幂等完成。这样「查询未建索引的库」时能给出清晰报错，
            # 而不是在构造阶段就因集合不存在而失败。
            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=self.settings.collection_name,
                embedding=self.embeddings,
                validate_collection_config=False,
                validate_embeddings=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"初始化 Qdrant 失败: {exc}", cause=exc) from exc

    def ensure_collection(self) -> None:
        """幂等确保集合存在（写入前调用）。"""
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """幂等创建集合：已存在则跳过。"""
        exists = self.client.collection_exists(self.settings.collection_name)
        if exists:
            logger.info("集合已存在，跳过创建: %s", self.settings.collection_name)
            return

        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.settings.vector_size,
                # 通过名称索引枚举，避免魔法字符串写错
                distance=qdrant_models.Distance[self.settings.distance.upper()],
            ),
        )
        logger.info(
            "已创建集合 %s（维度=%d, 距离=%s）",
            self.settings.collection_name,
            self.settings.vector_size,
            self.settings.distance,
        )

    # ---------------------------------------------------------------
    # 写入
    # ---------------------------------------------------------------
    def add_documents(self, documents: list[Document], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """将分块后的文档向量化并写入向量库。

        Args:
            documents: 待写入的 Document 列表
            batch_size: 单批写入条数

        Returns:
            实际写入的条数
        """
        if not documents:
            logger.warning("没有可写入的文档，跳过")
            return 0

        total = len(documents)
        start = time.perf_counter()
        try:
            # 写入前确保集合存在（幂等），让 ingest 流程自给自足
            self._ensure_collection()
            # add_documents 内部会调用 embeddings.embed_documents 批量向量化
            # QdrantVectorStore 按内容哈希生成稳定 UUID，重复导入会覆盖而非重复插入
            self.store.add_documents(documents, batch_size=batch_size)
        except Exception as exc:
            raise VectorStoreError(f"向量化并写入失败: {exc}", cause=exc) from exc

        elapsed = time.perf_counter() - start
        logger.info("写入完成: %d 条，耗时 %.2fs", total, elapsed)
        return total

    # ---------------------------------------------------------------
    # 检索
    # ---------------------------------------------------------------
    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """向量相似度检索，返回 (Document, score) 并按分数降序。

        Notes:
            Cosine 距离下 score 越大越相似；threshold 过滤在上层 Retriever 完成。

        Raises:
            VectorStoreError: 集合不存在或检索异常
        """
        if not self.client.collection_exists(self.settings.collection_name):
            raise VectorStoreError(
                f"集合 '{self.settings.collection_name}' 不存在，请先执行 ingest 导入文档"
            )

        try:
            hits = self.store.similarity_search_with_score(query, k=k)
            # 统一按相似度降序（不同距离度量下返回顺序可能不同）
            hits = sorted(hits, key=lambda x: x[1], reverse=True)
            return hits
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"向量检索失败: {exc}", cause=exc) from exc

    # ---------------------------------------------------------------
    # 统计 / 管理
    # ---------------------------------------------------------------
    def get_all_documents(self, limit: int = 100_000) -> list[Document]:
        """从 Qdrant 全量导出所有文档（用于重建 BM25 等派生索引）。

        原理：Qdrant 以 payload 存储文本内容（page_content 与 metadata），
        通过 client.scroll 分页遍历全部 point，还原为 Document 列表。
        Qdrant 是权威存储，BM25 等内存索引都可由此无损重建。
        """
        if not self.client.collection_exists(self.settings.collection_name):
            return []

        documents: list[Document] = []
        try:
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.settings.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    metadata = payload.get("metadata", {})
                    if content:
                        documents.append(Document(page_content=content, metadata=metadata))
                if offset is None or len(points) == 0:
                    break
        except Exception as exc:
            raise VectorStoreError(f"从 Qdrant 导出文档失败: {exc}", cause=exc) from exc

        logger.info("已从 Qdrant 导出 %d 篇文档", len(documents))
        return documents

    def count(self) -> int:
        """返回集合内向量总数。"""
        try:
            info = self.client.count(self.settings.collection_name, exact=True)
            return int(info.count or 0)
        except Exception as exc:
            raise VectorStoreError(f"统计向量数失败: {exc}", cause=exc) from exc

    def stats(self) -> dict[str, Any]:
        """返回向量库状态快照，供 CLI stats 命令展示。"""
        return {
            "collection": self.settings.collection_name,
            "mode": "remote" if self.settings.qdrant_url else "local",
            "vector_size": self.settings.vector_size,
            "distance": self.settings.distance,
            "count": self.count(),
        }

    def drop_collection(self) -> None:
        """删除整个集合（危险操作，调用方需二次确认）。"""
        try:
            self.client.delete_collection(self.settings.collection_name)
            logger.warning("已删除集合: %s", self.settings.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"删除集合失败: {exc}", cause=exc) from exc
