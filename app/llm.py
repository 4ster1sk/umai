import os
from collections.abc import Generator

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}

SYSTEM_PROMPT = """あなたは食の好み分析の専門家です。
ユーザーが記録した「食べておいしかったもの」のデータをもとに、
好みの傾向を分析し、次に食べるとよさそうなものを提案してください。
日本語で回答してください。"""


def _build_user_prompt(data: dict[str, list[dict[str, str]]]) -> str:
    lines = ["以下は記録されたおいしかったものの一覧です。\n"]
    for category, items in data.items():
        lines.append(f"【{category}】")
        for entry in items:
            lines.append(f"  {entry['date']}: {entry['item']}")
        lines.append("")
    lines.append(
        "上記のデータをもとに、\n"
        "1. 好みの傾向・共通点の分析\n"
        "2. カテゴリごとに次に食べるとよさそうなもの3件の提案（理由付き）\n"
        "をお願いします。"
    )
    return "\n".join(lines)


def _get_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))


def stream_analyze(data: dict[str, list[dict[str, str]]]) -> Generator[str, None, None]:
    """LLMに分析を依頼し、テキストをストリーミングで yield する。"""
    user_prompt = _build_user_prompt(data)
    model = _get_model()

    if PROVIDER == "anthropic":
        yield from _stream_anthropic(user_prompt, model)
    elif PROVIDER == "openai":
        yield from _stream_openai(user_prompt, model)
    elif PROVIDER == "gemini":
        yield from _stream_gemini(user_prompt, model)
    else:
        raise ValueError(f"未対応のプロバイダー: {PROVIDER!r}  (anthropic/openai/gemini を指定してください)")


def _stream_anthropic(user_prompt: str, model: str) -> Generator[str, None, None]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    with client.messages.stream(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def _stream_openai(user_prompt: str, model: str) -> Generator[str, None, None]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _stream_gemini(user_prompt: str, model: str) -> Generator[str, None, None]:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = gemini_model.generate_content(user_prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text
