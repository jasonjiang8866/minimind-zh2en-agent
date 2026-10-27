import asyncio
import random
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import AsyncOpenAI, APIConnectionError, APIStatusError
from .config import settings

logger = logging.getLogger(__name__)


class InvalidCompletion(ValueError):
    """An empty, truncated or otherwise incomplete model response."""


class OpenAIClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=None, *, config=settings, http_client=None):
        self.config = config
        self.model = model or config.model_id
        # One retry policy, not multiplicative SDK + application retries.
        self.client = AsyncOpenAI(
            base_url=base_url or config.vllm_base_url,
            api_key=api_key or config.vllm_api_key,
            timeout=timeout if timeout is not None else config.request_timeout,
            max_retries=0, http_client=http_client,
        )

    async def aclose(self):
        await self.client.close()

    def retry_delay(self, attempt, error):
        delay = random.uniform(0, min(self.config.retry_max, self.config.retry_base * 2 ** min(attempt, 30)))
        if isinstance(error, APIStatusError):
            value = error.response.headers.get("retry-after")
            if value:
                try:
                    retry_after = float(value)
                except ValueError:
                    try:
                        retry_after = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        retry_after = 0
                delay = max(delay, min(self.config.retry_max, max(0, retry_after)))
        return delay

    async def chat(self, system_prompt: str, user_text: str, max_tokens: int = 2048) -> str:
        for attempt in range(self.config.request_attempts):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, temperature=0.2, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_text}],
                )
                if not resp.choices:
                    raise InvalidCompletion("No completion choices returned")
                choice = resp.choices[0]
                if choice.finish_reason != "stop":
                    raise InvalidCompletion(f"Incomplete completion: finish_reason={choice.finish_reason!r}")
                content = choice.message.content
                if not isinstance(content, str) or not content.strip():
                    raise InvalidCompletion("Empty completion returned")
                return content.strip()
            except (APIConnectionError, APIStatusError, InvalidCompletion) as error:
                if isinstance(error, APIStatusError) and error.status_code not in (408, 409, 429) and error.status_code < 500:
                    raise
                if attempt + 1 == self.config.request_attempts:
                    raise
                delay = self.retry_delay(attempt, error)
                logger.warning("vLLM request failed (%s); retry %s/%s in %.2fs", type(error).__name__,
                               attempt + 2, self.config.request_attempts, delay)
                await asyncio.sleep(delay)
        raise AssertionError("Unreachable")
