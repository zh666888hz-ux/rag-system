"""
统一异常体系。

设计原则：
1. 所有业务异常统一继承自 RAGError，上层（CLI / API 层）只需捕获一个基类即可完成
   全局降级与错误提示，避免散落 try/except；
2. 按「出错阶段」细分异常子类，方便区分：
   - 外部依赖类（Embedding / VectorStore / Generation）：多为瞬时故障，可重试或熔断；
   - 数据类（DocumentLoad / Chunking）：多为数据本身问题，重试无意义，需修正输入；
3. 每个异常都携带 cause 字段保留原始异常链，便于排查根因（logging 可打印 traceback）。
"""
from typing import Optional


class RAGError(Exception):
    """RAG 系统所有业务异常的基础类。"""

    def __init__(self, message: str, *, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause  # 保留原始异常，便于根因追踪


class ConfigError(RAGError):
    """配置错误：参数缺失、格式非法、规则冲突等。"""


class DocumentLoadError(RAGError):
    """文档加载/解析失败：文件不存在、类型不支持、解析异常等。"""


class ChunkingError(RAGError):
    """文本分块失败：分块参数非法等。"""


class EmbeddingError(RAGError):
    """向量化失败：网络超时、服务不可用、模型错误等。"""


class VectorStoreError(RAGError):
    """向量库操作失败：集合不存在、写入/查询异常等。"""


class RetrieverError(RAGError):
    """检索失败：检索参数非法、向量库异常等。"""


class GenerationError(RAGError):
    """大模型生成失败：调用超时、返回异常、上下文超长等。"""


class PipelineError(RAGError):
    """流水线编排层错误：多个步骤组合时出现的整体性失败。"""


class DatabaseError(RAGError):
    """数据库操作失败：连接异常、SQL 执行错误、约束冲突等。"""


class AuthError(RAGError):
    """认证/鉴权失败：用户名或密码错误、token 非法或过期、权限不足等。"""
    # HTTP 状态码映射由 API 层处理：401 未认证 / 403 无权限


class UserAlreadyExistsError(AuthError):
    """注册时用户名已存在（可捕获后转 409 冲突）。"""


class SessionError(RAGError):
    """会话管理失败：会话不存在、无权限访问等。"""
