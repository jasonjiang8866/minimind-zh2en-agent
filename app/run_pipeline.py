import asyncio
import json
import logging
from pathlib import Path

from filelock import FileLock
from .config import settings
from .graph import build_graph, Nodes
from .ledger import Ledger
from .vllm_client import OpenAIClient

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


class IncompleteRun(RuntimeError):
    pass


async def run_pipeline(config=settings, client=None):
    if config.checkpoint != 0:
        raise ValueError("CHECK_POINT skipping is unsafe. Unset CHECK_POINT; verified ledger resume is automatic.")
    output = Path(config.output_path).resolve()
    input_path = Path(config.input_file).resolve()
    artifacts = [output, *(Path(str(output) + suffix) for suffix in
                           (".sqlite", ".sqlite-wal", ".sqlite-shm", ".lock", ".pairs.jsonl", ".summary.json"))]
    if any(input_path == path or (path.exists() and input_path.exists() and path.samefile(input_path)) for path in artifacts):
        raise ValueError("Input path must not overlap output or recovery files")
    output.parent.mkdir(parents=True, exist_ok=True)
    # OS lock is released on process exit, including a crash. No stale PID cleanup.
    with FileLock(str(output) + ".lock", timeout=0):
        ledger = Ledger(output)
        owned_client = client is None
        try:
            translator = (ROOT / "prompts/translate_system.txt").read_text(encoding="utf-8").strip()
            fixer = (ROOT / "prompts/fix_system.txt").read_text(encoding="utf-8").strip()
            ledger.prepare(input_path, {"schema": 1, "model": config.model_id,
                                       "translator": translator, "fixer": fixer,
                                       "max_tokens": config.max_tokens, "max_fixes": config.max_fixes})
            ledger.report("running")
            if client is None:
                client = OpenAIClient(config=config)
            graph = build_graph(Nodes(translator, fixer, client, config.max_tokens), config.max_fixes)
            queue = asyncio.Queue(maxsize=config.concurrency * 2)

            async def producer():
                last_id = -1
                while rows := ledger.pending_after(last_id, config.concurrency * 2):
                    for record_id, raw in rows:
                        await queue.put((record_id, raw))
                        last_id = record_id
                for _ in range(config.concurrency):
                    await queue.put(None)

            async def worker():
                while (item := await queue.get()) is not None:
                    record_id, raw = item
                    try:
                        state = await graph.ainvoke({"text": json.loads(raw)["text"]},
                                                   {"recursion_limit": 2 * config.max_fixes + 10})
                    except Exception as error:
                        # A failed row stays visible and retryable; cancellation
                        # propagates so interrupted rows remain pending.
                        ledger.record_result(record_id, error=f"{type(error).__name__}: {error}")
                        logger.warning("Source record %s failed (%s); retained for resume", record_id, type(error).__name__)
                    else:
                        ledger.record_result(record_id, translated=state["translated"])

            finished = False
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(producer())
                    for _ in range(config.concurrency):
                        group.create_task(worker())
                finished = True
            finally:
                summary = ledger.export(finished)
            if summary["status"] != "complete":
                raise IncompleteRun(f"Incomplete translation: {summary}. Rerun with the same input/output to retry unfinished rows.")
            return summary
        finally:
            try:
                if owned_client and client is not None:
                    await client.aclose()
            finally:
                ledger.close()


async def main():
    summary = await run_pipeline()
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
