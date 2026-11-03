"""RAG Prompt 模板：带知识片段引用的问答生成。"""

SYSTEM_PROMPT = (
    "你是企业知识库智能问答助手，面向金融垂直领域。请严格依据下面提供的知识片段回答用户问题，"
    "并在使用某片段信息时标注引用编号，如 [1][2]。\n"
    "要求：\n"
    "1. 优先使用知识片段中的事实与数字，不要编造；\n"
    "2. 若片段不足以回答，请明确说明知识库中暂无相关信息；\n"
    "3. 回答简洁、结构化，适合企业内部使用。"
)


def build_rag_prompt(question: str, hits: list[dict], memory_context: str = "") -> str:
    """组装最终 Prompt：知识片段 + 分级记忆 + 当前问题。"""
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(f"【知识片段{i}】来源《{h['doc_name']}》第{h['seq'] + 1}段：\n{h['text']}\n")
    knowledge = "\n".join(parts)

    memory_block = f"\n【对话记忆】\n{memory_context}\n" if memory_context else ""
    return (
        f"{knowledge}\n{memory_block}"
        f"【当前问题】\n{question}\n\n"
        "请基于以上知识片段回答，并标注引用编号。"
    )
