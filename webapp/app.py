"""
RAG 聊天前端（Streamlit）。

功能：
1. 注册 / 登录（对接后端 JWT 鉴权，token 存于 session_state）；
2. 会话管理：侧边栏新建 / 切换 / 删除会话，加载历史消息；
3. 流式聊天：通过 SSE 接口实时渲染答案（打字机效果）；
4. 引用溯源：每条回答展示候选片段，实际引用编号高亮展示。

设计说明：
- 前端只负责展示与调用后端 REST/SSE 接口，不含任何业务逻辑；
- API 地址通过环境变量 API_BASE_URL 注入（本地/容器均可配置）；
- SSE 用 requests 流式读取，按 `data: {json}` 解析事件，把 delta 文本
  交给 st.write_stream 实现打字机渲染。

启动：streamlit run webapp/app.py --server.port 8501
"""
import json
import os
import urllib.parse

import requests
import streamlit as st

# 后端 API 基址：容器部署时由 docker-compose 注入 http://api:8000/api
API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000/api")

st.set_page_config(page_title="RAG 知识库问答", page_icon="📚", layout="wide")


# ---------------------------------------------------------------
# 后端调用封装
# ---------------------------------------------------------------
def api_json(method: str, path: str, body=None, token: str | None = None, timeout: int = 120):
    """同步 JSON 调用；非 2xx 抛出带后端 detail 的异常。"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.request(method, API_BASE + path, json=body, headers=headers, timeout=timeout)
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"[{resp.status_code}] {detail}")
    return resp.json() if resp.content else None


def sse_stream(path: str, body: dict, token: str):
    """SSE 流式调用：返回 (事件生成器)。生成器逐条产出解析后的事件 dict。"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.post(API_BASE + path, json=body, headers=headers, stream=True, timeout=300)
    resp.raise_for_status()

    def gen():
        # SSE 以空行分帧，每帧 data: {json}
        for raw in resp.iter_lines(decode_unicode=True):
            line = raw.strip() if raw else ""
            if line.startswith("data: "):
                try:
                    yield json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
        resp.close()

    return gen()


# ---------------------------------------------------------------
# 状态初始化
# ---------------------------------------------------------------
def init_state():
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("conversations", [])
    st.session_state.setdefault("conv_id", None)
    st.session_state.setdefault("messages", [])  # 当前会话消息（本地缓存，用于渲染）
    # 延迟创建标记：True 表示「新会话已就绪但尚未落库」，输入第一条消息时才真正创建，
    # 从源头避免「建了会话却一条消息没发」的空会话被保存
    st.session_state.setdefault("new_pending", False)


# ---------------------------------------------------------------
# 认证页面
# ---------------------------------------------------------------
def auth_page():
    st.title("📚 RAG 知识库问答")
    st.caption("多轮对话 · 混合检索 · 引用溯源 · JWT 鉴权")
    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        with st.form("login"):
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pwd")
            if st.form_submit_button("登录", type="primary"):
                try:
                    data = api_json("POST", "/auth/login", {"username": username, "password": password})
                    st.session_state.token = data["access_token"]
                    st.session_state.username = data["user"]["username"]
                    load_conversations()  # 登录后立即加载会话列表，避免需手动刷新才显示
                    st.rerun()
                except Exception as e:
                    st.error(f"登录失败：{e}")

    with tab_register:
        with st.form("register"):
            username = st.text_input("用户名", key="reg_user")
            password = st.text_input("密码", type="password", key="reg_pwd")
            if st.form_submit_button("注册"):
                # 注意：st.text_input 不支持 min_value 参数，需手动校验密码长度
                if not username or len(password) < 6:
                    st.error("用户名不能为空，且密码至少 6 位")
                else:
                    try:
                        api_json("POST", "/auth/register", {"username": username, "password": password})
                        st.success("注册成功，请登录")
                    except Exception as e:
                        st.error(f"注册失败：{e}")


# ---------------------------------------------------------------
# 会话管理（侧边栏）
# ---------------------------------------------------------------
def load_conversations():
    try:
        st.session_state.conversations = api_json(
            "GET", "/conversations", token=st.session_state.token
        )
    except Exception as e:
        st.error(f"加载会话失败：{e}")
        st.session_state.conversations = []


def load_messages(conv_id: int):
    """拉取指定会话的全部历史并缓存到本地渲染。"""
    msgs = api_json(
        "GET", f"/conversations/{conv_id}/messages", token=st.session_state.token
    )
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "sources": m.get("sources")}
        for m in msgs
    ]


def _conv_label(c: dict, used: set[str]) -> str:
    """会话显示名：只显示标题（自动取第一个问题），不暴露内部 id。

    标题可能重复（如历史遗留的多个「新会话」），此时追加序号保证 radio
    选项唯一可区分。
    """
    title = (c.get("title") or "").strip() or f"会话 {c['id']}"
    if title in used:
        n = 2
        base = title
        while f"{base} ({n})" in used:
            n += 1
        title = f"{base} ({n})"
    used.add(title)
    return title


def _on_radio_change():
    """radio 切换回调：仅在用户主动点击 radio 时由 Streamlit 触发。

    设计要点：
    - 回调只响应「值变化」的用户点击；程序 rerun 的默认选中不触发，避免误切换；
    - 待新建状态下列表首项是「✨ 新会话」占位（映射 None）：点它保持待新建，
      点任意旧会话则正常切换——从而修复「新建后因 radio 粘着旧选中而回不去」。
    """
    chosen = st.session_state.get("conv_radio")
    entries = st.session_state.get("conv_entries", [])
    cid = None
    for lab, c in entries:
        if lab == chosen:
            cid = c
            break
    if cid is None:
        # 用户点了「新会话」占位项 → 保持待新建状态（不落库）
        st.session_state.new_pending = True
        st.session_state.conv_id = None
        st.session_state.messages = []
        return
    st.session_state.conv_id = cid
    st.session_state.new_pending = False  # 手动切换到具体会话，取消待新建
    try:
        load_messages(cid)
    except Exception as e:
        st.error(f"加载历史失败：{e}")
        st.session_state.messages = []


def sidebar():
    with st.sidebar:
        st.header("会话")
        if st.button("➕ 新建会话", use_container_width=True):
            try:
                # 延迟创建：这里只进入「待新建」状态，不调后端落库；
                # 等用户输入第一条消息时才真正创建会话（见 send_message），
                # 从而保证「没发过话的新会话」不会被保存、不会堆积在列表。
                # 顺手清理历史遗留的空会话（旧版本可能已落库的 0 消息会话）。
                for c in st.session_state.conversations:
                    if c.get("message_count", 0) == 0 and c["id"] != st.session_state.conv_id:
                        api_json("DELETE", f"/conversations/{c['id']}", token=st.session_state.token)
                st.session_state.new_pending = True
                st.session_state.conv_id = None
                st.session_state.messages = []
                load_conversations()
                st.rerun()
            except Exception as e:
                st.error(f"新建会话失败：{e}")

        # 会话列表（radio 切换）
        if st.session_state.conversations:
            # 关键设计：
            # 1. label 只用稳定内容（会话标题，自动取第一个问题），不暴露 id，
            #    且不包含会变的 message_count —— 提问后条数变化不会导致 radio 重置跳转；
            # 2. 待新建状态插入「✨ 新会话」占位项（映射 None）：radio 选中它时
            #    表示「未落库的新会话」，用户点任意旧会话都能触发切换（修复回不去）。
            used: set[str] = set()
            entries: list[tuple[str, int | None]] = []
            if st.session_state.new_pending:
                entries.append(("✨ 新会话（输入第一条消息开始）", None))
            for c in st.session_state.conversations:
                entries.append((_conv_label(c, used), c["id"]))
            st.session_state.conv_entries = entries  # 供 on_change 回调读取 label→id 映射
            label_list = [e[0] for e in entries]
            current_idx = 0
            for i, (lab, cid) in enumerate(entries):
                if cid == st.session_state.conv_id:
                    current_idx = i
                    break
            # 用 on_change 而非「rerun 后 if 判断」：
            # on_change 只在用户主动点击 radio 时触发，程序 rerun 的默认选中不触发。
            st.radio("我的会话", label_list, index=current_idx,
                     key="conv_radio", on_change=_on_radio_change,
                     label_visibility="collapsed")

            col1, col2 = st.columns(2)
            if col1.button("🔄 刷新", use_container_width=True):
                load_conversations()
                if st.session_state.conv_id:
                    try:
                        load_messages(st.session_state.conv_id)
                    except Exception as e:
                        st.error(f"加载历史失败：{e}")
                        st.session_state.messages = []
                st.rerun()
            if col2.button("🗑️ 删除", use_container_width=True,
                           disabled=(st.session_state.conv_id is None)):
                try:
                    api_json("DELETE", f"/conversations/{st.session_state.conv_id}",
                             token=st.session_state.token)
                    st.session_state.conv_id = None
                    st.session_state.messages = []
                    load_conversations()
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败：{e}")

        st.divider()
        st.caption(f"👤 {st.session_state.username}")
        if st.button("退出登录", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.conv_id = None
            st.session_state.messages = []
            st.session_state.new_pending = False
            st.rerun()


# ---------------------------------------------------------------
# 消息渲染 + 引用溯源
# ---------------------------------------------------------------
def render_messages():
    """渲染当前会话消息；assistant 消息附带引用溯源折叠区。"""
    for m in st.session_state.messages:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])
                sources = m.get("sources") or []
                if sources:
                    _render_sources(sources, refs=None)


def _render_sources(sources: list[dict], refs: list[int] | None):
    """引用溯源：每条片段一个折叠区，refs 高亮「答案实际引用」的编号。

    两种来源：
    - 实时问答：refs 非空，按 refs 判断哪些片段被引用（✅ 徽标）；
    - 历史回看：refs 为 None，改用每条落库时的 cited 标记（后端已持久化），
      保证切会话/刷新后溯源区内容与引用标记仍完整展示。
    """
    with st.expander(f"📎 引用溯源（{len(sources)} 条）"):
        for s in sources:
            ref = s.get("ref_id")
            if refs is not None:
                cited = ref in refs
            else:
                cited = bool(s.get("cited", False))
            badge = "✅ 引用" if cited else ""
            st.markdown(
                f"**[{ref}] {s.get('file_name','')} 第{s.get('page','?')}页** "
                f"· 相似度 {s.get('score','-')} {badge}"
            )
            snippet = s.get("snippet") or ""
            if snippet:
                st.caption(snippet[:300])
            st.divider()


# ---------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------
def send_message(text: str):
    """发送提问并流式渲染回答；结束后缓存消息并更新会话标题。"""
    # 延迟创建：待新建状态下输入第一条消息时，才真正创建会话并落库。
    # 这样「没发过话的新会话」永远不会保存；创建失败时不会残留空会话。
    if st.session_state.new_pending and not st.session_state.conv_id:
        try:
            conv = api_json(
                "POST", "/conversations", {"title": "新会话"}, token=st.session_state.token
            )
            st.session_state.conv_id = conv["id"]
            st.session_state.new_pending = False
        except Exception as e:
            st.error(f"创建会话失败：{e}")
            return

    if not st.session_state.conv_id:
        st.error("请先新建或选择一个会话")
        return

    # 立即回显用户消息
    st.session_state.messages.append({"role": "user", "content": text, "sources": None})
    with st.chat_message("user"):
        st.markdown(text)

    payload = {"conversation_id": st.session_state.conv_id, "message": text}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_parts: list[str] = []
        sources: list[dict] = []
        refs: list[int] = []

        def render():
            """把已收集的 delta 实时渲染到占位区（打字机效果）。"""
            placeholder.markdown("".join(answer_parts) + "▌")

        try:
            for event in sse_stream("/chat/stream", payload, st.session_state.token):
                etype = event.get("type")
                if etype == "delta":
                    answer_parts.append(event.get("text", ""))
                    render()
                elif etype == "source":
                    sources.append(event)
                elif etype == "refs":
                    refs = event.get("refs", [])
                elif etype == "done":
                    answer_parts = [event.get("answer", "")]
                    refs = event.get("refs", [])
                    sources = event.get("sources", sources)
                elif etype == "error":
                    placeholder.markdown(f"⚠️ {event.get('message','生成失败')}")
        except Exception as e:
            placeholder.markdown(f"⚠️ 请求失败：{e}")

        # 渲染完成态（去掉光标）
        final_text = "".join(answer_parts)
        placeholder.markdown(final_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": final_text, "sources": sources}
        )
        if sources:
            _render_sources(sources, refs=refs or None)

    # 会话有新消息 → 刷新列表（标题/条数变化）
    load_conversations()


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main():
    init_state()
    if not st.session_state.token:
        auth_page()
        return

    sidebar()
    st.title("📚 RAG 知识库问答")

    if st.session_state.new_pending:
        # 新会话就绪但尚未落库：提示用户直接输入第一条消息即可自动创建
        st.info("✨ 新会话已就绪 —— 在下方输入第一条消息即自动创建并开始对话（不会保存空会话）。")
    elif not st.session_state.conv_id:
        st.info("请先在左侧新建或选择一个会话开始提问。")
        return

    render_messages()

    prompt = st.chat_input("输入你的问题…")
    if prompt and prompt.strip():
        send_message(prompt.strip())
        st.rerun()


if __name__ == "__main__":
    main()
