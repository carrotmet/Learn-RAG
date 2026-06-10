from openai import OpenAI
import os


class OpenRouterLLM:
    """通过 OpenRouter 统一路由所有 LLM 调用，屏蔽模型差异"""

    def __init__(self, model_id: str = None):
        self.model_id = model_id or os.getenv("DEFAULT_MODEL", "openai/gpt-3.5-turbo")
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
