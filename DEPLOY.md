# RAG 一键部署教程（Docker）

本教程面向**生产/演示环境**：用 Docker 一键拉起「MySQL + Qdrant + RAG API + Streamlit 前端」四个服务，实现完整的检索增强生成问答系统（含混合检索、重排、多轮会话、引用溯源、JWT 鉴权）。

---

## 一、整体架构

```
浏览器 ──> Streamlit 前端 (8501)
                │  SSE 流式 / REST
                ▼
           RAG API (8000)  FastAPI
             │ 检索          │ 生成
             ▼               ▼
       Qdrant(6333)    OpenAI 兼容网关(DeepSeek)
       向量检索+BM25      + 本地 Embedding/Rerank
             │
             ▼
       MySQL(3306)  ← 用户/会话/聊天历史
```

- **Qdrant**：向量库（存储 chunk 向量 + 元数据）
- **MySQL 8.4**：业务数据（用户、会话、消息），表由 API 启动时自动创建
- **RAG API**：FastAPI 应用，负责登录鉴权、混合检索（向量 + BM25 + RRF 融合）、本地 Rerank 重排、大模型生成（OpenAI 兼容 API）、SSE 流式输出
- **Streamlit**：聊天前端，登录/会话管理/流式打字机/引用溯源

> 说明：Embedding（`BAAI/bge-small-zh-v1.5`）与 Rerank（`Xenova/bge-reranker-base`）均为**本地模型**，首次启动自动从 HuggingFace 镜像下载并缓存在 `model_cache` 卷中；只有对话生成走 DeepSeek（OpenAI 兼容）接口。

---

## 二、前置条件

| 组件 | 要求 | 说明 |
|---|---|---|
| Docker Desktop | ≥ 4.x（含 docker compose v2） | Windows/macOS 使用；**引擎依赖 WSL2**，见下方「常见问题」 |
| 网络 | 可访问 api.deepseek.com 与 hf-mirror.com | DeepSeek 生成 + HuggingFace 模型下载 |
| .env 配置 | 提供 `RAG_OPENAI_API_KEY` | 其余有默认值 |

**本机已知情况**：Docker CLI/compose 已安装，但**未安装 WSL2**，导致 Docker 引擎无法启动。请先完成 WSL2 安装（见「常见问题」）再执行部署。

---

## 三、一键部署步骤

### 1. 准备配置

```bash
# 复制配置模板（或沿用已有 .env）
copy .env.example .env
```

编辑 `.env`，至少确认/填写：

```ini
RAG_OPENAI_API_KEY=sk-xxxxxxxx                 # DeepSeek 的 API Key（必填）
RAG_OPENAI_BASE_URL=https://api.deepseek.com/v1
RAG_CHAT_MODEL=deepseek-v4-flash

MYSQL_ROOT_PASSWORD=your_mysql_root_password                    # MySQL 密码（与 RAG_MYSQL_PASSWORD 保持一致）
RAG_JWT_SECRET=生产环境请改成足够长的随机串
```

> `.env` 已被 `.dockerignore` 排除，**不会**打进镜像；仅 compose 启动时读取注入。

### 2. 构建并启动

```bash
# 首次构建（下载依赖/模型镜像，国内网络已配置阿里云 PyPI 加速，约 5~15 分钟）
docker compose up -d --build

# 查看启动状态（等待 mysql/qdrant 通过 healthcheck、api/webapp 变为 healthy）
docker compose ps
```

### 3. 灌入知识文档（首次必做）

将 PDF 放入宿主机的 `./docs` 目录（**需要提前创建**），然后在 api 容器内执行 ingest：

```bash
# 宿主机创建文档目录并放入 PDF
mkdir docs
# 把 D:\rag_app\docs\测试知识文档.pdf 复制进 docs 目录后执行：

# 在容器内执行索引（复用 API 的 Qdrant 与模型环境）
docker compose exec api python main.py ingest /app/docs/测试知识文档.pdf
```

> 说明：`docs` 目录通过 `docker-compose.yml` 挂载进 api 容器（`./docs:/app/docs`），ingest 与在线问答共用同一个 Qdrant 集合。

### 4. 访问系统

| 入口 | 地址 | 用途 |
|---|---|---|
| 聊天前端 | http://localhost:8501 | 注册/登录 → 新建会话 → 提问（SSE 流式） |
| API 文档 | http://localhost:8000/docs | Swagger UI，可直接调试接口 |
| 健康检查 | http://localhost:8000/api/health | 探活 |

首次使用：在前端注册账号 → 登录 → 新建会话 → 提问（如「什么是 RAG？」），回答底部会列出「引用溯源」（来源文件、页码、相似度、片段编号）。

### 5. 常用运维命令

```bash
docker compose logs -f api          # 跟踪 API 日志
docker compose logs -f webapp       # 跟踪前端日志
docker compose ps                   # 查看各服务状态
docker compose down                 # 停止（保留数据卷）
docker compose down -v              # 停止并删除数据卷（清空所有数据，慎用）
docker compose pull                 # 更新基础镜像
docker compose build --no-cache api # 强制重建 API 镜像（改了依赖后）
```

---

## 四、数据持久化

compose 定义 4 个命名卷，`docker compose down` 不删除数据，重启/重建镜像后数据仍在：

| 卷 | 挂载点 | 内容 |
|---|---|---|
| `mysql_data` | MySQL 数据目录 | 用户/会话/聊天历史 |
| `qdrant_data` | Qdrant 存储目录 | 向量数据（知识库） |
| `model_cache` | api 容器 `/app/models` | 本地 Embedding/Rerank 模型缓存 |
| `rag_logs` | api 容器 `/app/logs` | 应用日志（含每条检索召回记录） |

---

## 五、本地开发模式（不使用 Docker）

若想在本机直接跑（不依赖 Docker），环境与步骤：

```bash
# 1. 前置：Python 3.10 + MySQL（root/your_mysql_root_password，建库 rag_chat）
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # 后端
pip install -r webapp\requirements.txt   # 前端

# 2. 准备 .env（含 RAG_OPENAI_API_KEY、RAG_MYSQL_PASSWORD 等）

# 3. 灌库
python main.py ingest D:\rag_app\docs\测试知识文档.pdf

# 4. 启动 API（终端 1）
$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
python main.py serve --port 8000

# 5. 启动前端（终端 2）
$env:API_BASE_URL="http://127.0.0.1:8000/api"
streamlit run webapp/app.py
```

---

## 六、测试

```bash
# 离线冒烟测试（检索/重排/代码结构）
python -m pytest tests/ -q

# API 集成测试（需真实 MySQL，设置环境变量后运行；覆盖注册/登录/会话/聊天/SSE 流式）
$env:RAG_TEST_MYSQL="1"; python -m pytest tests/test_api.py -q
```

---

## 七、常见问题（FAQ）

**Q1：Docker 引擎无法启动 / docker info 报 npipe 连接失败**
本机很可能缺少 WSL2。Docker Desktop 需要 WSL2 作为后端：

```powershell
# 以管理员身份打开 PowerShell
wsl --install            # 安装 WSL2 + 默认发行版
# 完成后【重启电脑】
# 重启后打开 Docker Desktop → Settings → Resources → WSL Integration 勾选启用
docker info              # 看到 Server Version 即就绪
```

若不想装 WSL2，可改用 **Docker Desktop 的 Hyper-V 后端**（需 Windows 专业版 + 开启 Hyper-V）。

**Q2：模型下载慢/失败**
compose 已内置 `HF_ENDPOINT=https://hf-mirror.com`（国内镜像）与 `HF_HUB_DISABLE_XET=1`（镜像站不支持 Xet 协议）。若仍失败：检查网络能否访问 hf-mirror.com，或挂代理后给 api 服务追加 `HTTP_PROXY/HTTPS_PROXY` 环境变量。

**Q3：首次问答很慢**
首次启动时 API 会预下载 Embedding/Rerank 模型（数百 MB），之后缓存于 `model_cache` 卷，不再重复下载。

**Q4：DeepSeek 报鉴权/超时**
确认 `.env` 的 `RAG_OPENAI_API_KEY` 正确；超时可适当调大 `service/generator.py` 中 LLM 请求的 timeout 配置（默认 120s）。

**Q5：重新灌库后旧文档仍在**
`docker compose exec api python main.py drop` 可清空 Qdrant 集合，再重新 ingest。

**Q6：如何修改服务端口**
编辑 `docker-compose.yml` 中 `ports` 映射（如 `"8501:8501"` → `"9001:8501"`），前端/API 端口改动需保持 `webapp` 的 `API_BASE_URL` 指向正确地址。

---

## 八、文件清单（本次部署相关）

```
├── Dockerfile               # API 镜像（python:3.10-slim + 阿里云 pip 加速）
├── Dockerfile.webapp        # 前端镜像（streamlit）
├── docker-compose.yml       # 四服务编排（mysql/qdrant/api/webapp + 健康检查）
├── entrypoint.sh            # API 容器启动脚本（等 MySQL + 预下载模型 + 起 uvicorn）
├── .dockerignore            # 构建上下文排除（.venv/logs/models/.env 等）
├── webapp/
│   ├── app.py               # Streamlit 聊天页面（登录/会话/SSE 流式/溯源）
│   └── requirements.txt     # streamlit / requests
└── .env.example             # 环境变量模板（RAG_*）
```
