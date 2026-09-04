"""
配置管理模块。

核心原则：
1. 所有可调参数集中在这里管理，避免魔法数字散落在各业务文件中；
2. 使用 pydantic-settings 从「环境变量 + .env 文件」读取配置，并在启动时统一做
   类型与业务规则校验，让配置错误尽早暴露（fail-fast），而不是运行到一半才报错；
3. 通过 env_prefix="RAG_" 给配置项划分清晰命名空间，防止与其他应用的环境变量冲突；
4. get_settings() 使用 lru_cache 做进程级单例，整个进程生命周期只解析一次配置。
"""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------- 基础元信息 ----------
    # env_prefix：要求所有环境变量以 RAG_ 开头，如 RAG_OPENAI_BASE_URL
    # extra="ignore"：忽略未声明的多余环境变量，避免误配置被带进来
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用基础 ----------
    app_name: str = "rag-langchain-qdrant"
    log_level: str = Field(default="INFO", description="日志级别：DEBUG/INFO/WARNING/ERROR")
    log_dir: str = Field(default="logs", description="日志文件目录")

    # ---------- OpenAI 兼容 API ----------
    # 通过 base_url 指向任意 OpenAI 兼容网关（OpenAI/DeepSeek/通义/vLLM/OneAPI 等），
    # 这是本工程对「OpenAI 兼容 API」的实现方式：不绑定任何单一厂商。
    openai_base_url: str = Field(..., description="OpenAI 兼容接口地址，如 https://api.deepseek.com/v1")
    openai_api_key: str = Field(..., description="OpenAI 兼容接口的 API Key")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding 模型名")
    chat_model: str = Field(default="gpt-4o-mini", description="对话（生成）模型名")
    llm_timeout: float = Field(default=60.0, ge=1.0, description="LLM/Embedding 单请求超时（秒）")
    llm_max_retries: int = Field(default=3, ge=0, le=10, description="LLM/Embedding 失败重试次数")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="生成温度；检索问答建议 0 保证确定性")

    # ---------- Embedding 独立配置（可选） ----------
    # 生产环境中 Chat 与 Embedding 常来自不同厂商（如 Chat 用 DeepSeek、Embedding 用本地模型）。
    # 这两个字段用于把 Embedding 与 Chat 解耦；留空则回退到 openai_base_url / openai_api_key。
    embedding_provider: Literal["openai_compatible", "local_fastembed"] = Field(
        default="openai_compatible",
        description="Embedding 提供方：openai_compatible=远端兼容网关；local_fastembed=本地 ONNX 模型（离线免费）",
    )
    embedding_base_url: Optional[str] = Field(default=None, description="Embedding 网关地址；为空回退 openai_base_url")
    embedding_api_key: Optional[str] = Field(default=None, description="Embedding 网关 Key；为空回退 openai_api_key")

    # ---------- Qdrant ----------
    # 两种部署模式：
    #   1) 远程服务：qdrant_url 非空（生产，如 Docker / Qdrant Cloud）
    #   2) 本地文件模式：qdrant_url 为空，数据以 WAL 方式持久化到 qdrant_path（开发/单机）
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant 服务地址；为空则使用本地文件模式")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant 服务 API Key（云托管/鉴权时填写）")
    qdrant_path: str = Field(default="./qdrant_data", description="本地持久化路径（仅本地模式生效）")
    collection_name: str = Field(default="rag_documents", description="向量集合名称")
    vector_size: int = Field(default=1536, ge=1, description="向量维度，必须与 Embedding 模型输出维度一致")
    distance: Literal["Cosine", "Euclid", "Dot"] = Field(default="Cosine", description="向量距离度量")

    # ---------- 文本分块 ----------
    chunk_size: int = Field(default=512, ge=50, le=8192, description="文本块大小（字符数）")
    chunk_overlap: int = Field(default=64, ge=0, description="相邻文本块重叠字符数，保持边界上下文连续")

    # ---------- 检索 ----------
    top_k: int = Field(default=4, ge=1, le=50, description="检索返回的片段数量")
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="相似度阈值，None 表示不过滤")

    # ---------- 检索增强：混合召回 + 重排 ----------
    # 混合召回：向量检索（语义）+ BM25（关键词）双路召回，再用 RRF（倒数排名融合）合并，
    # 兼顾语义相似与关键词精确命中，显著提升召回覆盖率，是降低幻觉的第一道防线。
    retrieval_mode: Literal["hybrid", "vector"] = Field(
        default="hybrid",
        description="检索模式：hybrid=向量+BM25 混合召回；vector=纯向量检索",
    )
    bm25_k: int = Field(default=8, ge=1, le=50, description="BM25 独立召回条数（进入融合池）")
    fusion_k: int = Field(default=60, ge=1, le=1000, description="RRF 融合常数 k（越大排名权重越平缓）")

    # 重排：Cross-Encoder 对 query 与每个候选片段联合编码打分，比双塔向量相似度更精确，
    # 重排后只保留最相关的 top_k 喂给大模型，过滤低质量噪声片段，是降低幻觉的第二道防线。
    enable_rerank: bool = Field(default=True, description="是否启用 Rerank 重排序")
    rerank_provider: Literal["local_onnx", "none"] = Field(
        default="local_onnx",
        description="重排提供方：local_onnx=本地 ONNX cross-encoder（离线免费）；none=不重排",
    )
    rerank_model: str = Field(
        default="Xenova/bge-reranker-base",
        description="本地重排模型（HuggingFace 上的 ONNX 版 bge-reranker）",
    )
    model_cache_dir: str = Field(default="./models", description="本地模型缓存目录（重排模型等）")
    rerank_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="重排分数阈值，低于该分数的片段不进上下文；None 表示只排序不过滤",
    )
    rerank_batch_size: int = Field(default=16, ge=1, le=128, description="重排批处理大小")

    # ---------- MySQL 聊天历史 ----------
    # 会话管理依赖 MySQL 持久化用户/会话/聊天消息，实现多轮对话记忆：
    # 每次提问把当前会话的最近若干轮历史一并送入大模型，使回答能理解上文语境。
    mysql_host: str = Field(default="127.0.0.1", description="MySQL 主机地址")
    mysql_port: int = Field(default=3306, ge=1, le=65535, description="MySQL 端口")
    mysql_user: str = Field(default="root", description="MySQL 用户名")
    mysql_password: str = Field(default="", description="MySQL 密码")
    mysql_db: str = Field(default="rag_chat", description="MySQL 数据库名（不存在则自动创建）")
    mysql_pool_size: int = Field(default=5, ge=1, le=100, description="数据库连接池大小")
    mysql_pool_recycle: int = Field(default=3600, ge=60, description="连接回收时间（秒），防止 MySQL wait_timeout 断连")
    mysql_connect_timeout: int = Field(default=10, ge=1, description="数据库连接超时（秒）")

    # 多轮记忆：一次「一问一答」记为 1 轮，只保留最近 N 轮进上下文，
    # 防止历史无限膨胀超出模型上下文窗口，也让记忆始终聚焦最近的对话。
    memory_rounds: int = Field(default=6, ge=0, le=50, description="多轮记忆保留最近轮数；0 表示关闭多轮记忆")

    # ---------- JWT 登录鉴权 ----------
    # JWT（JSON Web Token）：登录成功后服务端签发一个自包含、带签名的 token，
    # 客户端后续请求在 Authorization 头携带，服务端验签即可确认身份，无需存储会话。
    jwt_secret: str = Field(
        default="change-me-in-production-please-set-a-long-random-secret",
        description="JWT 签名密钥；生产环境必须改为足够长的随机字符串",
    )
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = Field(default="HS256", description="JWT 签名算法")
    jwt_expire_minutes: int = Field(default=1440, ge=1, le=10080, description="JWT 有效期（分钟），默认 1 天")

    # ---------- HTTP 服务 ----------
    api_host: str = Field(default="127.0.0.1", description="API 服务监听地址；生产建议 0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API 服务监听端口")

    # ---------- 业务规则校验 ----------
    @model_validator(mode="after")
    def _check_chunk_params(self) -> "Settings":
        """分块参数合理性校验：重叠必须小于块大小，否则切分循环无法收敛。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})"
            )
        return self

    @field_validator("openai_base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        """base_url 必须是合法 http(s) 地址，并去掉末尾斜杠避免拼接出双斜杠。"""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"openai_base_url 必须以 http(s):// 开头: {v!r}")
        return v.rstrip("/")

    @field_validator("jwt_secret")
    @classmethod
    def _validate_jwt_secret(cls, v: str) -> str:
        """JWT 密钥安全校验：生产环境必须替换默认值，并满足最小长度。"""
        if v == "change-me-in-production-please-set-a-long-random-secret":
            # 允许开发默认值，但记录警告由上层打印；这里只做长度约束
            return v
        if len(v) < 16:
            raise ValueError(f"jwt_secret 长度不能小于 16，当前长度: {len(v)}")
        return v


@lru_cache
def get_settings() -> Settings:
    """进程级单例：首次调用时解析配置并缓存，后续直接复用。"""
    return Settings()
