from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()  # take environment variables from .env.

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
r = client.chat.completions.create(
    model=os.getenv("MODEL_ID"),
    messages=[{"role":"user","content":"give me example why rust is faster than go"}],
    max_tokens=5120,
    temperature=0.2
)
print(r.choices[0].message.content)