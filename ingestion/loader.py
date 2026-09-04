"""
PDF 文档加载模块。

核心原理：
1. 使用 pypdf 直接抽取 PDF 的「文本层」（内容型 PDF 内嵌的文本对象）。
   注意边界：扫描版/图片型 PDF 没有文本层，需要走 OCR（如 PaddleOCR、Tesseract），
   本模块不做 OCR，属于明确的已知边界；
2. 逐页抽取文本并保留 page_number 元数据。这一步非常关键：检索命中后我们能
   告诉用户「答案来自哪个文件的第几页」，实现可溯源；
3. 每页做空白规范化（多个空白/换行压缩为单个空格），减少噪声对向量化的干扰；
4. 每个文件独立 try/except：批量导入时单个文件损坏不阻塞整体流程；
5. 入参校验：文件存在性、后缀、体积上限，把「数据问题」在入口处就拦截掉。
"""
import logging
from pathlib import Path
from typing import Iterable, Union

from langchain_core.documents import Document
from pypdf import PdfReader

from core.exceptions import DocumentLoadError

logger = logging.getLogger(__name__)

# 单文件体积上限（MB），防止超大文件拖垮内存
MAX_PDF_SIZE_MB = 200


def _validate_pdf(path: Path) -> None:
    """参数/数据校验：文件必须存在、是文件、是 .pdf、体积未超限。"""
    if not path.exists():
        raise DocumentLoadError(f"文件不存在: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"不是普通文件: {path}")
    if path.suffix.lower() != ".pdf":
        raise DocumentLoadError(f"不支持的文件类型: {path}（仅支持 .pdf）")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise DocumentLoadError(f"文件过大（{size_mb:.1f}MB > 上限 {MAX_PDF_SIZE_MB}MB）: {path}")


def load_pdf(path: Union[str, Path]) -> list[Document]:
    """解析单个 PDF，返回按页拆分、带元数据的 Document 列表。

    Args:
        path: PDF 文件路径

    Returns:
        每页一个 Document，metadata 含 source（绝对路径）、file_name、page（页码，从 1 起）

    Raises:
        DocumentLoadError: 校验失败或解析失败
    """
    p = Path(path).resolve()
    _validate_pdf(p)

    try:
        reader = PdfReader(str(p))
        documents: list[Document] = []
        for page_no, page in enumerate(reader.pages, start=1):
            # extract_text() 可能返回 None，需兜底为空串
            text = page.extract_text() or ""
            # 规范化空白：压缩换行/制表/连续空格，提升向量化质量
            text = " ".join(text.split())
            if not text.strip():
                # 空页（常见于扫描件、纯图片页、封面）跳过并告警
                logger.warning("第 %d 页无文本，跳过（可能是扫描件/图片页）: %s", page_no, p.name)
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(p), "file_name": p.name, "page": page_no},
                )
            )
        logger.info("解析完成: %s（%d 页，有效文本 %d 页）", p.name, len(reader.pages), len(documents))
        return documents
    except DocumentLoadError:
        raise
    except Exception as exc:  # pypdf 各类解析异常统一归类
        raise DocumentLoadError(f"PDF 解析失败: {p.name}: {exc}", cause=exc) from exc


def load_pdfs(paths: Iterable[Union[str, Path]]) -> list[Document]:
    """批量加载：支持多个文件与目录混传。

    - 传入目录时递归扫描该目录下所有 .pdf；
    - 单个文件失败只记录日志，不中断整体（批量容错）。

    Raises:
        DocumentLoadError: 全部文件都失败时抛出（避免上层拿到空结果还继续建索引）
    """
    all_docs: list[Document] = []
    failed: list[str] = []

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            # glob 同时匹配大小写两种后缀，再用 dict.fromkeys 去重
            files = list(dict.fromkeys([*sorted(p.glob("*.pdf")), *sorted(p.glob("*.PDF"))]))
            for f in files:
                try:
                    all_docs.extend(load_pdf(f))
                except DocumentLoadError as exc:
                    failed.append(str(exc))
        else:
            try:
                all_docs.extend(load_pdf(p))
            except DocumentLoadError as exc:
                failed.append(str(exc))

    if failed:
        logger.warning("有 %d 个文件加载失败: %s", len(failed), failed)
    if not all_docs and failed:
        # 全失败属于整体性错误，必须显式抛出，不能静默返回空
        raise DocumentLoadError(f"全部文件加载失败，前 5 条原因: {'; '.join(failed[:5])}")

    return all_docs
