from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()  # take environment variables from .env.

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))

stream = client.chat.completions.create(
    model=os.getenv("MODEL_ID"),
    messages=[{"role":"user","content":"give me example why rust is faster than go"}],
    max_tokens=5120,
    temperature=0.2,
    stream=True,
)

full = []
for chunk in stream:  # chunk is ChatCompletionChunk
    for choice in chunk.choices:
        delta = choice.delta
        if delta.content:               # may be None for non-text deltas (e.g., role)
            print(delta.content, end="", flush=True)
            full.append(delta.content)
print()  # newline
text = "".join(full)
