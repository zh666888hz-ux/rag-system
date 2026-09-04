"""
生成器模块：调用 OpenAI 兼容大模型生成最终答案。

核心原理：
1. ChatOpenAI 通过 base_url 对接任意 OpenAI 兼容服务（OpenAI / DeepSeek / 通义 /
   vLLM 网关等），因此本模块对「模型厂商」完全无感；
2. 消息序列结构（支持多轮记忆）：
   - system：RAG 约束规则 + 参考资料（带编号的片段）。参考资料编号 [1][2]... 与
     回答中的引用编号一一对应，是实现「引用溯源」的关键——模型只需在句末写 [n]
     即可声明「这句话来自片段 n」，同时被强制禁止编造编号；
   - history（可选）：当前会话的最近若干轮「用户提问/助手回答」，让模型理解上文
     语境，这是多轮对话记忆的载体（纯上下文窗口式记忆，无需训练）；
   - human：本轮用户问题；
3. temperature=0：检索问答场景优先事实正确性，关掉随机性，避免同一问题多次回答漂移。
4. 引用校验：生成后校验答案里出现的编号是否都在片段编号范围内，越界编号记录
   警告日志（提示模型可能产生了幻觉引用），并返回合法引用编号供溯源展示。
"""
import json
import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import Settings
from core.exceptions import GenerationError

logger = logging.getLogger(__name__)

# RAG 系统提示词：约束模型行为 + 规定编号引用规则（引用溯源的核心约束）
_SYSTEM_PROMPT = """你是一个严谨的文档问答助手。请严格遵循以下规则：
1. 仅依据下面提供的「参考资料」回答用户问题，不要使用参考资料之外的知识；
2. 如果参考资料中没有足够信息，请直接回答“根据现有资料无法回答”，不要编造；
3. 回答要准确、简洁、条理清晰；
4. 引用溯源：当你引用某个片段的观点或事实时，请在句末用 [编号] 标注，例如“……bge-small-zh 是常用的中文向量模型[2]。”
   编号必须是「参考资料」中存在的片段编号（1 到 {max_ref}），禁止编造不存在的编号；
5. 结合多轮对话历史理解上下文，但回答依据仍以本轮参考资料为准。

参考资料：
{context}"""

# 匹配答案中的 [编号] 引用标记，如 [1]、[12]
_REF_PATTERN = re.compile(r"\[(\d{1,3})\]")


class Generator:
    """基于 OpenAI 兼容 API 的答案生成器（支持多轮记忆与引用编号）。"""

    def __init__(self, settings: Settings) -> None:
        # 对话模型客户端：base_url 指向兼容网关
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,  # 检索问答用 0 保证确定性
            timeout=settings.llm_timeout,          # 单请求超时
            max_retries=settings.llm_max_retries,  # 瞬时故障自动重试
            max_tokens=1024,                       # 控制单次输出上限
        )

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """把检索到的片段拼成带编号与来源标注的文本，喂给模型。

        编号从 1 开始，与回答中的 [编号] 引用一一对应。
        """
        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "?")
            parts.append(f"[片段{i}]（来源：{source}，第{page}页）\n{doc.page_content}")
        return "\n\n".join(parts)

    def generate(
        self,
        question: str,
        contexts: list[Document],
        history: Optional[list[dict]] = None,
    ) -> str:
        """基于检索上下文与多轮历史生成答案。

        Args:
            question: 用户问题
            contexts: 检索到的相关片段（Document 列表），顺序即编号顺序（1..N）
            history: 多轮对话历史 [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            生成的答案文本

        Raises:
            GenerationError: 模型调用失败
        """
        if not contexts:
            # 没有检索到任何片段时不调用模型，直接给出兜底，节省一次调用
            logger.warning("无检索上下文，跳过模型调用")
            return "根据现有资料无法回答（未检索到相关片段）。"

        try:
            system_text = _SYSTEM_PROMPT.format(
                context=self._format_context(contexts), max_ref=len(contexts)
            )
            # 组装消息序列：system（约束+资料）→ 历史 → 本轮问题
            messages = [SystemMessage(content=system_text)]
            for item in history or []:
                role = item.get("role")
                content = str(item.get("content", ""))
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                # 其它角色（如 system 提示）忽略，避免污染对话

            messages.append(HumanMessage(content=question))
            response = self.llm.invoke(messages)
            answer = response.content if isinstance(response.content, str) else str(response.content)
            logger.info(
                "生成完成: 上下文片段=%d, 历史轮次=%d, 答案长度=%d 字",
                len(contexts),
                len(history or []),
                len(answer),
            )
            return answer
        except Exception as exc:
            raise GenerationError(f"大模型调用失败: {exc}", cause=exc) from exc

    def stream_generate(
        self,
        question: str,
        contexts: list[Document],
        history: Optional[list[dict]] = None,
    ):
        """流式生成答案，逐段 yield 文本增量（用于 SSE 打字机效果）。

        Args:
            同 generate()。

        Yields:
            字符串增量；调用方负责拼接与异常处理。

        Raises:
            GenerationError: 模型调用中途失败（由调用方捕获）。
        """
        system_text = _SYSTEM_PROMPT.format(
            context=self._format_context(contexts), max_ref=len(contexts)
        )
        messages = [SystemMessage(content=system_text)]
        for item in history or []:
            role = item.get("role")
            content = str(item.get("content", ""))
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))

        # ChatOpenAI.stream() 返回增量块（chunk.content 为文本增量），
        # 底层仍走 OpenAI 兼容的 SSE 协议，由 langchain 屏蔽细节
        stream = self.llm.stream(messages)
        for chunk in stream:
            delta = getattr(chunk, "content", "")
            if delta:
                yield delta if isinstance(delta, str) else str(delta)

    @staticmethod
    def extract_references(answer: str, max_ref: int) -> list[int]:
        """从答案中抽取引用编号，并校验是否越界。

        返回去重后的合法编号列表；越界编号（模型幻觉引用）记录警告并剔除，
        供上层做引用溯源展示时对齐真实的片段列表。
        """
        found = [int(m) for m in _REF_PATTERN.findall(answer)]
        if not found:
            return []
        valid = sorted({n for n in found if 1 <= n <= max_ref})
        invalid = [n for n in found if not (1 <= n <= max_ref)]
        if invalid:
            logger.warning("答案中出现越界引用编号 %s（片段总数=%d），已剔除（可能为幻觉引用）",
                           invalid, max_ref)
        return valid

    @staticmethod
    def format_refs_for_log(refs: list[int]) -> str:
        """引用编号转日志字符串。"""
        return json.dumps(refs, ensure_ascii=False)
