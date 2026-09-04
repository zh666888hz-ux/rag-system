"""
对话路由：多轮 RAG 问答接口。

- POST /api/chat          ：同步 JSON 问答（程序化调用 / 测试）
- POST /api/chat/stream   ：SSE 流式问答（前端打字机效果，推荐）

流程：鉴权 → 会话归属校验 → 多轮记忆组装（ChatService 内部完成）→
混合检索+重排+生成 → 历史持久化 → 返回带引用溯源的结果。
"""
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.deps import get_chat_service, get_current_user, get_database, get_db
from api.schemas import ChatRequest, ChatResponse, ChatSourceItem
from persistence.db import Database
from persistence.repositories import ConversationRepository
from service.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["对话"])

# SSE 响应必需的响应头：
# - Cache-Control: no-cache    禁止代理/浏览器缓存流式内容
# - Connection: keep-alive     保持长连接
# - X-Accel-Buffering: no      禁用 nginx 等反向代理的缓冲，保证实时推送
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("", response_model=ChatResponse, summary="多轮 RAG 问答（同步 JSON）")
def chat(
    body: ChatRequest,
    current: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """在指定会话内执行一次问答，并自动保存本轮对话到 MySQL。

    回答中 [编号] 为引用标记，与返回的 sources.ref_id 一一对应，实现引用溯源。
    """
    result = service.ask(
        session=db,
        user_id=current["user_id"],
        conversation_id=body.conversation_id,
        message=body.message,
        top_k=body.top_k,
    )
    return ChatResponse(
        conversation_id=result["conversation_id"],
        answer=result["answer"],
        refs=result.get("refs", []),
        sources=[ChatSourceItem(**s) for s in result.get("sources", [])],
        num_sources=result.get("num_sources", 0),
        reranked=result.get("reranked", False),
        memory_rounds=result.get("memory_rounds", 0),
    )


@router.post("/stream", summary="多轮 RAG 问答（SSE 流式）")
def chat_stream(
    body: ChatRequest,
    current: dict = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
    database: Database = Depends(get_database),
) -> StreamingResponse:
    """SSE 流式问答：实时推送答案增量，结束后附引用溯源。

    事件协议（每条 `data: {json}`，客户端按 type 区分）：
      meta → source* → delta* → refs → done
    异常时推送 error 事件。

    注意：这里不用 get_db 依赖，而是手动管理数据库会话——
    StreamingResponse 的生成器在路由返回后才被迭代，此时 FastAPI 的依赖
    teardown 已执行（get_db 会提前关闭 session），故由生成器自己负责
    commit/rollback/close，保证持久化与流生命周期一致。
    """
    # 开流前完成会话归属校验：错误可在流开始前以 404 语义返回
    session = database.new_session()
    try:
        ConversationRepository.get_for_user(session, current["user_id"], body.conversation_id)
    except Exception:
        session.close()
        raise

    def event_gen():
        try:
            for event in service.ask_stream(
                session=session,
                user_id=current["user_id"],
                conversation_id=body.conversation_id,
                message=body.message,
                top_k=body.top_k,
            ):
                # SSE 格式：以 "data: " 开头，空行结束；中文不转义便于调试
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            session.commit()
            logger.info("SSE 流结束并提交: conv=%s", body.conversation_id)
        except GeneratorExit:
            # 客户端中途断开：不提交（丢弃未完成轮次），释放会话
            session.rollback()
            session.close()
            raise
        except Exception as exc:
            logger.error("SSE 流异常: %s", exc)
            session.rollback()
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            session.close()

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)
