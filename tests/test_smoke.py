"""
端到端冒烟测试：用「本地 Qdrant 文件模式 + 本地假 OpenAI 兼容服务」跑通
完整 RAG 流程（PDF 加载 → 分块 → 向量化入库 → 检索 → 生成），
并验证参数校验与异常路径。

为什么用假服务：测试不应该依赖外网 API Key 与真实模型，
用本地 HTTP 服务模拟 OpenAI 兼容接口（/v1/embeddings、/v1/chat/completions），
即可验证我们工程的「接口对接方式」是否正确，同时保持测试可离线、可重复。
"""
from __future__ import annotations

import http.server
import json
import re
import sys
import threading
from pathlib import Path

import pytest

# 让项目根目录可被导入（无论从哪个目录运行 pytest）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from core.exceptions import RetrieverError, VectorStoreError  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from service.ingestion_pipeline import IngestionPipeline  # noqa: E402
from service.query_pipeline import QueryPipeline  # noqa: E402
from tests.make_pdf import build_pdf  # noqa: E402
from vectorstore.qdrant_store import QdrantStore  # noqa: E402

EMBED_DIM = 8  # 假 Embedding 向量维度，同时用于 vector_size 校验


# ---------------------------------------------------------------
# 假 OpenAI 兼容服务
# ---------------------------------------------------------------
class FakeOpenAIHandler(http.server.BaseHTTPRequestHandler):
    """模拟 OpenAI 兼容 API 的两个端点。

    原理：embedding 用「字符频率向量」近似（ord(char) % dim 累加后归一化），
    保证相似文本得到相似向量，让检索有意义；chat 端点直接回显问题与首个片段，
    便于断言「检索结果确实进入了生成上下文」。
    """

    def log_message(self, *args):  # 关闭默认访问日志，保持测试输出干净
        pass

    # ----- 假 embedding：字符频率向量 -----
    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for ch in text:
            vec[ord(ch) % EMBED_DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    # ----- 请求分发 -----
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")

        if self.path.rstrip("/").endswith("/embeddings"):
            resp = self._handle_embeddings(payload)
        elif self.path.rstrip("/").endswith("/chat/completions"):
            resp = self._handle_chat(payload)
        else:
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_embeddings(self, payload: dict) -> dict:
        inputs = payload.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            {"object": "embedding", "index": i, "embedding": self._embed(t)}
            for i, t in enumerate(inputs)
        ]
        return {
            "object": "list",
            "data": data,
            "model": payload.get("model", "fake"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _handle_chat(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        question = messages[-1]["content"] if messages else ""
        # 从 system 消息的 context 里抓取第一个片段，验证上下文确实被传入
        context = ""
        for m in messages:
            if m.get("role") == "system":
                match = re.search(r"\[片段1\]（来源：[^）]*）\n(.{0,100})", m["content"])
                if match:
                    context = match.group(1)
                break
        content = f"根据资料回答：{question}。资料片段摘要：{context}"
        return {
            "id": "fake-chat",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


@pytest.fixture(scope="module")
def fake_openai_server():
    """启动假 OpenAI 服务，返回其 base_url。"""
    server = http.server.HTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture()
def settings(tmp_path, fake_openai_server):
    """构造测试用配置：本地 Qdrant + 假 OpenAI + 小向量维度。"""
    return Settings(
        openai_base_url=fake_openai_server,
        openai_api_key="test-key",
        # 测试显式锁定远端兼容网关模式，避免被 .env 中的 local_fastembed 覆盖
        embedding_provider="openai_compatible",
        embedding_model="fake-embed",
        chat_model="fake-chat",
        # 测试关闭重排（避免加载本地 ONNX 模型拖慢/依赖真实模型下载）
        enable_rerank=False,
        qdrant_url=None,                      # 本地文件模式
        qdrant_path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=EMBED_DIM,
        distance="Cosine",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
    )


def _write_sample_pdfs(tmp_path: Path) -> list[Path]:
    """生成两份测试 PDF。"""
    p1 = tmp_path / "intro.pdf"
    p2 = tmp_path / "faq.pdf"
    # 用两页文本，验证逐页解析；内容包含 RAG 关键术语便于检索断言
    p1.write_bytes(build_pdf(["LangChain 是一个构建 LLM 应用的框架。", "RAG 是检索增强生成技术。" * 20]))
    p2.write_bytes(build_pdf(["Qdrant 是一个向量数据库，用于语义检索。", "OpenAI 兼容 API 可对接任意大模型网关。" * 20]))
    return [p1, p2]


# ---------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------
class TestIngestion:
    def test_full_ingest(self, settings, tmp_path):
        """端到端索引：加载 → 分块 → 入库，统计信息正确。"""
        pdfs = _write_sample_pdfs(tmp_path)
        store = QdrantStore(settings)
        result = IngestionPipeline(settings, store).run([str(p) for p in pdfs])

        assert result["documents"] >= 4        # 2 个文件 × 2 页
        assert result["chunks"] >= 1
        assert result["stored"] == result["chunks"]
        assert store.count() == result["stored"]
        assert result["elapsed_seconds"] >= 0

    def test_query_after_ingest(self, settings, tmp_path):
        """端到端问答：检索命中 → 生成器拿到上下文并产出答案与溯源。"""
        pdfs = _write_sample_pdfs(tmp_path)
        store = QdrantStore(settings)
        IngestionPipeline(settings, store).run([str(p) for p in pdfs])

        result = QueryPipeline(settings, store).ask("RAG 是什么")
        assert result["answer"].startswith("根据资料回答：RAG 是什么")
        assert result["num_sources"] >= 1
        assert result["sources"][0]["source"].endswith(".pdf")
        assert result["sources"][0]["page"] >= 1
        assert 0 <= result["sources"][0]["score"] <= 1

    def test_query_before_ingest_raises(self, settings, tmp_path):
        """未建索引就提问 → 抛 VectorStoreError，避免静默返回空。"""
        settings.collection_name = "not_exist_collection"
        store = QdrantStore(settings)
        with pytest.raises(VectorStoreError):
            QueryPipeline(settings, store).ask("任何问题")


class TestRRFusion:
    """RRF（倒数排名融合）单元测试：验证混合召回的融合正确性。"""

    @staticmethod
    def _doc(text: str, start: int):
        return Document(page_content=text, metadata={"source": "s.pdf", "page": 1, "start_index": start})

    def test_shared_doc_ranks_first(self):
        """同一片段在两路（向量+BM25）都命中时，RRF 融合分应最高。"""
        from service.retriever import Retriever

        doc1 = self._doc("向量与BM25都命中", 0)
        doc2 = self._doc("只在向量路命中", 10)
        doc3 = self._doc("只在BM25路命中", 20)

        dense = [(doc1, 0.8), (doc2, 0.7)]       # 向量路：doc1、doc2
        bm25 = [(doc1, 3.0), (doc3, 2.0)]        # BM25路：doc1、doc3

        retriever = Retriever.__new__(Retriever)  # 仅测融合逻辑，不初始化依赖
        fused = retriever._reciprocal_rank_fusion([dense, bm25], k=60)

        # doc1 在两路都排第 1 → 融合分 = 1/61 + 1/61 ≈ 0.0328，必然最高
        # doc2/doc3 都是单路第 2 名 → 融合分相等（1/62）
        assert fused[0][0] == doc1
        assert fused[0][1] > fused[1][1]
        assert fused[1][1] == pytest.approx(fused[2][1])
        # 去重：融合结果不应出现同一片段两次
        assert len(fused) == 3

    def test_dedup_by_doc_key(self):
        """同一片段（相同 source+page+start_index）只计一次。"""
        from service.retriever import Retriever

        doc = self._doc("重复片段", 0)
        retriever = Retriever.__new__(Retriever)
        fused = retriever._reciprocal_rank_fusion([[(doc, 0.9)], [(doc, 4.0)]], k=60)
        assert len(fused) == 1
        assert fused[0][0] == doc


class TestValidation:
    def test_chunk_overlap_must_be_smaller(self, fake_openai_server):
        """配置校验：overlap >= size 必须报错（fail-fast）。"""
        with pytest.raises(ValueError):
            Settings(
                openai_base_url=fake_openai_server,
                openai_api_key="k",
                chunk_size=100,
                chunk_overlap=100,
            )

    def test_bad_base_url(self):
        """配置校验：base_url 必须为 http(s)。"""
        with pytest.raises(ValueError):
            Settings(openai_base_url="api.openai.com/v1", openai_api_key="k")

    def test_retriever_bad_k(self, settings, tmp_path):
        """检索参数校验：k 超出范围必须报错。"""
        store = QdrantStore(settings)
        from service.retriever import Retriever

        retriever = Retriever(store, settings)
        with pytest.raises(RetrieverError):
            retriever.retrieve("问题", k=999)
        with pytest.raises(RetrieverError):
            retriever.retrieve("   ")
