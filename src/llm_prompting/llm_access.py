import logging

from openai import OpenAI

from src.utils.config import DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT


# ---------------------- ChatGPT Stuf------------
def classify_sentiment_chatgpt(
        client: OpenAI,
        model_type,
        prompt: str,
        user_prompt: str,
        message: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
        top_p: float = 0.95,
):
    text = user_prompt.format(text=message)
    msgs = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': text},
    ]

    # Disable HTTP info
    response = client.chat.completions.create(
        model=model_type,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    result = response.choices[0].message.content
    return result


def load_credentials_open_ai(credentials_path: str = DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT):
    with open(credentials_path) as f:
        key = f.readline().strip()

    return key


def init_open_ai(api_key: str) -> OpenAI:
    client = OpenAI(api_key=api_key)
    logging.info("Api key set")
    return client
