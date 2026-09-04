"""
API 请求/响应模型（Pydantic v2）。

职责：
1. 请求体校验：对用户输入做类型与业务规则校验（如密码强度、用户名格式），
   非法输入在进入业务层之前就被 422 拒绝（fail-fast）；
2. 响应模型：约束返回结构，配合 FastAPI 自动生成 OpenAPI 文档，前端可对照契约开发。
"""
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# 用户名规则：3~32 位字母/数字/下划线，且不能纯数字
_USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,31}$")


# ---------------------------------------------------------------
# 认证
# ---------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="登录用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码（至少 6 位）")

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        if not _USERNAME_RE.match(v):
            raise ValueError("用户名需以字母开头，3~32 位字母/数字/下划线")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., description="登录用户名")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT 访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="有效期（秒）")
    user: "UserInfo"


class UserInfo(BaseModel):
    id: int
    username: str
    created_at: Optional[datetime] = None  # /me 从 token 解析时可为空


# ---------------------------------------------------------------
# 会话
# ---------------------------------------------------------------
class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255, description="会话标题；空则用默认")


class ConversationItem(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(default=0, description="会话内消息条数")


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[list[dict]] = Field(default=None, description="assistant 消息的引用溯源")
    created_at: datetime


# ---------------------------------------------------------------
# 对话
# ---------------------------------------------------------------
class ChatRequest(BaseModel):
    conversation_id: int = Field(..., description="目标会话 id")
    message: str = Field(..., min_length=1, description="用户提问内容")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="检索片段数（覆盖配置）")

    @field_validator("message")
    @classmethod
    def _check_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("提问内容不能为空")
        return v


class ChatSourceItem(BaseModel):
    ref_id: int = Field(..., description="引用编号，对应答案中的 [n]")
    file_name: str
    page: Optional[object] = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    refs: list[int] = Field(default_factory=list, description="答案中合法引用编号")
    sources: list[ChatSourceItem] = Field(default_factory=list, description="片段溯源明细")
    num_sources: int
    reranked: bool
    memory_rounds: int = Field(..., description="本次注入的多轮记忆轮数")


TokenResponse.model_rebuild()
