"""
索引流水线：把原始 PDF 变成向量库中的索引数据。

流程（RAG 的 Indexing 阶段）：加载(load) → 分块(split) → 向量化并入库(embed+store)。

设计要点：
1. 每一层只依赖自己前一层的输出，职责单一，便于单独替换实现
   （例如把 pypdf 换成 OCR、把递归切分换成语义切分，都不影响其他层）；
2. 返回结构化统计信息（文档数/块数/写入数/耗时），供调用方展示与监控；
3. 全程异常向上抛出为领域异常（DocumentLoadError / ChunkingError / VectorStoreError），
   由 CLI / API 层统一兜底。
"""
import logging
import time

from config.settings import Settings
from ingestion.loader import load_pdfs
from ingestion.splitter import split_documents
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """索引（写入）流水线。"""

    def __init__(self, settings: Settings, store: QdrantStore) -> None:
        self.settings = settings
        self.store = store

    def run(self, paths: list[str], batch_size: int = 64) -> dict:
        """执行一次完整索引流程。

        Args:
            paths: PDF 文件路径或目录列表
            batch_size: 向量库批量写入条数

        Returns:
            {"documents": 加载的文档页数, "chunks": 分块数, "stored": 写入数,
             "elapsed_seconds": 总耗时}
        """
        start = time.perf_counter()

        # 1) 加载 PDF → Document（按页）
        documents = load_pdfs(paths)

        # 2) 递归分块 → 更细粒度的 Document
        chunks = split_documents(
            documents,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        # 3) 向量化并写入 Qdrant
        stored = self.store.add_documents(chunks, batch_size=batch_size)

        elapsed = time.perf_counter() - start
        result = {
            "documents": len(documents),
            "chunks": len(chunks),
            "stored": stored,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info("索引流水线完成: %s", result)
        return result
