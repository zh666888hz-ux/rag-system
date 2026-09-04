"""
RAG 系统命令行入口。

支持子命令：
  ingest  导入 PDF 文档构建向量索引
  query   向 RAG 提问并获取带溯源的答案
  stats   查看向量库统计信息
  drop    删除向量集合（危险操作，需输入 yes 二次确认）
  serve   启动 HTTP API 服务（用户会话/多轮记忆/JWT 鉴权）

整体异常处理策略：
1. 配置阶段失败（Settings 校验不通过）→ 输出错误并退出码 2；
2. 业务执行阶段失败（RAGError 体系）→ 记录 ERROR 日志、输出用户友好错误并退出码 1；
3. 用户中断（Ctrl+C）→ 优雅退出码 130。
"""
import argparse
import json
import logging
import sys

from config.settings import Settings, get_settings
from core.exceptions import RAGError
from core.logging import setup_logging
from service.ingestion_pipeline import IngestionPipeline
from service.query_pipeline import QueryPipeline
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 子命令解析器。"""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="LangChain + Qdrant + OpenAI 兼容 API 的基础 RAG 系统",
    )
    sub = parser.add_subparsers(dest="command", required=True, help="子命令")

    # ingest：导入文档
    ingest = sub.add_parser("ingest", help="导入 PDF 文档构建向量索引")
    ingest.add_argument("paths", nargs="+", help="PDF 文件路径或目录（可多个）")
    ingest.add_argument("--batch-size", type=int, default=64, help="向量库批量写入条数")

    # query：问答
    query = sub.add_parser("query", help="向 RAG 提问")
    query.add_argument("question", help="用户问题")
    query.add_argument("--top-k", type=int, default=None, help="检索片段数（覆盖配置）")
    query.add_argument("--score-threshold", type=float, default=None, help="相似度阈值（覆盖配置）")

    # stats：统计
    sub.add_parser("stats", help="查看向量库统计信息")

    # drop：删除集合
    drop = sub.add_parser("drop", help="删除向量集合（危险操作，需二次确认）")
    drop.add_argument("--yes", action="store_true", help="跳过确认（脚本场景慎用）")

    # serve：启动 HTTP API（会话管理 / 多轮记忆 / JWT 鉴权）
    serve = sub.add_parser("serve", help="启动 HTTP API 服务")
    serve.add_argument("--host", type=str, default=None, help="监听地址（覆盖配置）")
    serve.add_argument("--port", type=int, default=None, help="监听端口（覆盖配置）")
    return parser


def print_answer(result: dict) -> None:
    """格式化打印问答结果，便于人读。"""
    print("\n" + "=" * 60)
    print(f"问题：{result['question']}")
    print("=" * 60)
    print(f"答案：\n{result['answer']}")
    print("-" * 60)
    print(f"参考片段（{result['num_sources']} 条）：")
    for i, src in enumerate(result["sources"], start=1):
        print(
            f"  [{i}] {src.get('file_name', src.get('source', ''))} "
            f"第{src.get('page', '?')}页 相似度={src.get('score')}"
        )
    print("=" * 60)


def _serve(args, settings: Settings) -> int:
    """启动 HTTP API 服务（由 uvicorn 承载）。"""
    import uvicorn

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"启动 RAG API 服务: http://{host}:{port}  （接口文档 /docs）")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ---------- 阶段一：配置加载与日志初始化 ----------
    try:
        settings: Settings = get_settings()
        setup_logging(settings.log_level, settings.log_dir)
    except Exception as exc:  # 配置错误单独处理：此时日志系统可能还没就绪
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    # ---------- 阶段二：业务执行 ----------
    try:
        # serve 分支由 api.app 自己初始化 QdrantStore，
        # 这里不得预先创建（本地文件模式的 Qdrant 不允许两个客户端同时打开同一目录）
        if args.command == "serve":
            return _serve(args, settings)

        store = QdrantStore(settings)

        if args.command == "ingest":
            result = IngestionPipeline(settings, store).run(args.paths, args.batch_size)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.command == "query":
            result = QueryPipeline(settings, store).ask(
                args.question, args.top_k, args.score_threshold
            )
            print_answer(result)

        elif args.command == "stats":
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))

        elif args.command == "drop":
            if not args.yes:
                answer = input(
                    f"确认删除集合 '{settings.collection_name}' 及其全部向量？输入 yes 确认: "
                )
                if answer.strip().lower() != "yes":
                    print("已取消删除。")
                    return 0
            store.drop_collection()
            print("已删除集合。")

        return 0

    except RAGError as exc:
        # 业务异常统一兜底：记录完整错误栈，向用户输出友好信息
        logger.error("任务失败: %s", exc, exc_info=(exc.cause is not None))
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
