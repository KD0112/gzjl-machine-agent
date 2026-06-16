from openai import OpenAI

from settings import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, require_deepseek_key


def main() -> None:
    client = OpenAI(
        api_key=require_deepseek_key(),
        base_url=DEEPSEEK_BASE_URL,
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise Chinese technical assistant."},
            {"role": "user", "content": "用一句话解释什么是RAG。"},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
