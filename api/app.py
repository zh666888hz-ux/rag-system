"""
FastAPI 应用装配：把配置、数据库、向量库、业务服务挂到 app.state，
注册路由与全局异常处理。

运行方式：
  uvicorn api.app:app --host 127.0.0.1 --port 8000
或（python -m api.app，内置 uvicorn 启动）
  python -m api.app

生命周期：
- 启动时：连接 MySQL（自动建库建表）→ 初始化 QdrantStore → 创建 ChatService；
- 退出时：释放数据库连接池。
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.deps import get_settings
from api.routes import auth as auth_routes
from api.routes import chat as chat_routes
from api.routes import conversations as conv_routes
from config.settings import Settings, get_settings as build_settings
from core.exceptions import (
    AuthError,
    DatabaseError,
    RAGError,
    SessionError,
    UserAlreadyExistsError,
)
from core.logging import setup_logging
from persistence.db import Database
from service.chat_service import ChatService
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _register_exception_handlers(app: FastAPI) -> None:
    """把业务异常统一映射为 HTTP 状态码与友好错误体。"""

    @app.exception_handler(AuthError)
    async def _auth_error(_req: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(UserAlreadyExistsError)
    async def _conflict_error(_req: Request, exc: UserAlreadyExistsError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SessionError)
    async def _not_found_error(_req: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(DatabaseError)
    async def _db_error(_req: Request, exc: DatabaseError) -> JSONResponse:
        logger.error("数据库错误: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "数据库服务暂不可用"})

    @app.exception_handler(RAGError)
    async def _rag_error(_req: Request, exc: RAGError) -> JSONResponse:
        logger.error("业务错误: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂：隔离装配逻辑，便于测试时注入自定义 Settings。"""
    if settings is None:
        settings = build_settings()

    setup_logging(settings.log_level, settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ---------- 启动：初始化重量级依赖 ----------
        logger.info("初始化 MySQL 连接...")
        db = Database(settings).connect()      # 自动建库建表（幂等）
        logger.info("初始化 Qdrant...")
        store = QdrantStore(settings)
        app.state.database = db
        app.state.store = store
        app.state.chat_service = ChatService(settings, store)
        app.state.settings = settings
        logger.info("RAG 服务启动完成: 监听 %s:%s", settings.api_host, settings.api_port)
        yield
        # ---------- 退出：释放资源 ----------
        try:
            db.close()
        except Exception as exc:  # pragma: no cover
            logger.warning("释放数据库连接池失败: %s", exc)

    app = FastAPI(
        title="RAG 知识库问答服务",
        description="LangChain + Qdrant + OpenAI 兼容 API 的 RAG 服务，"
                    "支持用户会话、多轮记忆、混合检索、引用溯源与 JWT 鉴权",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    _register_exception_handlers(app)
    app.include_router(auth_routes.router)
    app.include_router(conv_routes.router)
    app.include_router(chat_routes.router)

    @app.get("/api/health", tags=["系统"], summary="健康检查")
    def health() -> dict:
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
