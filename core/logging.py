"""
日志初始化模块。

要点：
1. 双通道输出：
   - 控制台（StreamHandler）：开发、排查时即时可见；
   - 滚动文件（RotatingFileHandler）：生产环境按大小滚动，保留最近 N 份，
     避免单个日志文件无限膨胀占满磁盘；
2. 统一格式化：时间 | 级别 | logger 名 | 线程 | 消息，便于 grep 与链路排查；
3. 对第三方 HTTP/客户端库做日志降噪：它们的 DEBUG/INFO 过于嘈杂，
   生产环境统一收敛到 WARNING，减少日志量并提升信噪比；
4. 幂等：重复调用不会重复挂 handler，避免热重载/多线程场景下日志重复。
"""
import logging
import os
from logging.handlers import RotatingFileHandler

# 统一日志格式：含时间、级别、模块名、线程名，方便生产环境定位问题
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s"

# 生产环境下过于嘈杂的第三方库，统一降噪
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "openai", "qdrant_client", "pypdf")

# 文件日志参数：单文件 10MB，滚动保留 5 份
_FILE_MAX_BYTES = 10 * 1024 * 1024
_FILE_BACKUP_COUNT = 5


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """初始化全局日志。

    Args:
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_dir: 日志文件目录，不存在时自动创建

    Notes:
        本函数应只在进程启动入口（main.py）调用一次。
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # 已初始化过则直接返回，保证幂等
    if root.handlers:
        return

    fmt = logging.Formatter(_LOG_FORMAT)

    # 1) 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # 2) 滚动文件输出
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "rag.log"),
        maxBytes=_FILE_MAX_BYTES,
        backupCount=_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 3) 第三方库降噪
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
