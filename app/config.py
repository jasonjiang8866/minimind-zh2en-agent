import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    vllm_base_url: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    vllm_api_key: str = os.getenv("OPENAI_API_KEY", "not-used")
    model_id: str     = os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507-FP8")
    concurrency: int  = int(os.getenv("CONCURRENCY", "10"))
    output_path: str  = os.getenv("OUTPUT_PATH", "./out/pretrain_hq.en.jsonl")
    dataset_name: str = os.getenv("DATASET_NAME", "jingyaogong/minimind_dataset")
    dataset_split: str = os.getenv("DATASET_SPLIT", "train")
    input_file: str   = os.getenv("INPUT_FILE", "pretrain_hq.jsonl")

settings = Settings()
