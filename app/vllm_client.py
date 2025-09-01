from typing import Optional
from openai import AsyncOpenAI
from .config import settings

class OpenAIClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0):
        self.base_url = base_url or getattr(settings, "vllm_base_url", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or getattr(settings, "vllm_api_key", "not-used")
        self.model = model or settings.model_id
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)

    async def aclose(self):
        # AsyncOpenAI doesn't require an explicit close, but keep for symmetry
        return

    async def chat(self, system_prompt: str, user_text: str, max_tokens: int = 512) -> str | None:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return resp.choices[0].message.content
