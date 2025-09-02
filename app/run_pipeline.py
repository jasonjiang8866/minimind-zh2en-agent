import os, asyncio, json, sys
from datasets import load_dataset, IterableDataset
from rich.progress import Progress, BarColumn, TimeElapsedColumn
from .config import settings
from .vllm_client import OpenAIClient
from .graph import build_graph, Nodes
from .detector import has_chinese

ROOT = os.path.dirname(os.path.dirname(__file__))

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            yield json.loads(line)

async def main():
    # Load prompts
    with open(os.path.join(ROOT, "prompts", "translate_system.txt"), "r", encoding="utf-8") as f:
        translate_sys = f.read().strip()
    with open(os.path.join(ROOT, "prompts", "fix_system.txt"), "r", encoding="utf-8") as f:
        fix_sys = f.read().strip()

    client = OpenAIClient()
    write_lock = asyncio.Lock()
    nodes = Nodes(translate_sys, fix_sys, settings.output_path, write_lock, client)
    graph = build_graph(nodes, max_fixes=1)

    # Input stream: either HF dataset or local jsonl
    src_iter = None
    if os.path.exists(settings.input_file):
        src_iter = read_text(settings.input_file)
    else:
        sys.exit(f"Input file {settings.input_file} not found")
        # # Expect lines like {"text": "..."} in pretrain_hq.jsonl
        # # This dataset’s card shows that format (key "text"). :contentReference[oaicite:0]{index=0}
        # hf = load_dataset(settings.dataset_name, split=settings.dataset_split, streaming=True)  # IterableDataset
        # def gen():
        #     for row in hf:  # each row should have "text"
        #         yield row
        # src_iter = gen()

    sem = asyncio.Semaphore(settings.concurrency)

    async def worker(rec):
        async with sem:
            state = {"record": rec, "text": rec.get("text","")}
            # cheap skip: if not Chinese, don't hit the LLM at all
            if not has_chinese(state["text"]):
                state["translated"] = state["text"]
                state["needs_fix"] = False
                await nodes.write_jsonl(state)
                return
            await graph.ainvoke(state)

    # progress
    tasks = []
    total = 0
    with Progress("[progress.description]{task.description}", BarColumn(), "{task.completed}", TimeElapsedColumn()) as prog:
        t = prog.add_task("Translating…", total=None)
        for rec in src_iter:
            recs = rec["text"].split("<|im_end|> <|im_start|>")
            for rec in recs:
                tasks.append(asyncio.create_task(worker({"text": "translate CN/ZH to EN: " + rec.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()})))
            total += 1
            if total % 2000 == 0:
                prog.update(t, completed=total)
                await asyncio.sleep(0)  # let loop breathe
        await asyncio.gather(*tasks)
        prog.update(t, completed=total)
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
