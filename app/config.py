import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    vllm_base_url: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    vllm_api_key: str = os.getenv("OPENAI_API_KEY", "not-used")
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507-FP8")
    concurrency: int = Field(default=int(os.getenv("CONCURRENCY", "10")), ge=1)
    output_path: str = os.getenv("OUTPUT_PATH", "./out/pretrain_hq.en.jsonl")
    input_file: str = os.getenv("INPUT_FILE", "pretrain_hq.jsonl")
    checkpoint: int = int(os.getenv("CHECK_POINT", "0"))
    request_attempts: int = Field(default=int(os.getenv("REQUEST_ATTEMPTS", "6")), ge=1)
    request_timeout: float = Field(default=float(os.getenv("REQUEST_TIMEOUT", "120")), gt=0)
    retry_base: float = Field(default=float(os.getenv("RETRY_BASE", "1")), gt=0)
    retry_max: float = Field(default=float(os.getenv("RETRY_MAX", "60")), gt=0)
    max_tokens: int = Field(default=int(os.getenv("MAX_TOKENS", "2048")), ge=1)
    max_fixes: int = Field(default=int(os.getenv("MAX_FIXES", "2")), ge=0)


settings = Settings()
