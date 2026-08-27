from openai import OpenAI
from app.config import settings


def get_client():
    kwargs = {"api_key": settings.openai_api_key}

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return OpenAI(**kwargs)


def chat(messages, temperature=0.2):
    response = get_client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


def chat_stream(messages, temperature=0.2):
    stream = get_client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta
