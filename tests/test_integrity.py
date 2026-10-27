import asyncio
import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import AuthenticationError
from filelock import FileLock, Timeout

from app.config import Settings
from app.graph import Nodes, build_graph
from app.run_pipeline import run_pipeline, IncompleteRun
from app.vllm_client import OpenAIClient, InvalidCompletion


class FakeClient:
    def __init__(self, fail=()):
        self.fail = fail
        self.calls = []
        self.active = 0
        self.peak = 0

    async def chat(self, system, text, **kwargs):
        self.calls.append(text)
        self.active += 1
        self.peak = max(self.active, self.peak)
        try:
            await asyncio.sleep(0.005 if "甲" in text else 0)
            if any(value in text for value in self.fail):
                raise RuntimeError("vLLM overloaded")
            return text.removeprefix("translate CN/ZH to EN: ").replace("甲", "Alpha").replace("乙", "Beta")
        finally:
            self.active -= 1


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source.jsonl"
        self.output = root / "result.jsonl"
        self.config = Settings(input_file=str(self.source), output_path=str(self.output), concurrency=2, checkpoint=0)

    def write_source(self, rows):
        self.rows = rows
        self.source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    def read(self, suffix=""):
        return [json.loads(line) for line in Path(str(self.output) + suffix).read_text().splitlines()]

    async def test_order_duplicates_metadata_and_idempotent_resume(self):
        self.write_source([{"text": "甲", "id": 10}, {"text": "乙"}, {"text": "甲"}, {"text": "English"}, {"text": ""}])
        client = FakeClient()
        summary = await run_pipeline(self.config, client)
        self.assertEqual(summary["success"], len(self.rows))
        self.assertEqual([r["text"] for r in self.read()], ["Alpha", "Beta", "Alpha", "English", ""])
        self.assertEqual(self.read()[0]["id"], 10)
        pairs = self.read(".pairs.jsonl")
        self.assertEqual([r["source_id"] for r in pairs], list(range(len(self.rows))))
        self.assertEqual([r["source"] for r in pairs], self.rows)
        self.assertEqual(len(client.calls), 3)
        self.assertLessEqual(client.peak, 2)
        client.calls.clear()
        await run_pipeline(self.config, client)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(self.read()), len(self.rows))

    async def test_failure_retained_and_only_failed_row_retried(self):
        self.write_source([{"text": "甲"}, {"text": "乙"}, {"text": "English"}])
        with self.assertRaises(IncompleteRun):
            await run_pipeline(self.config, FakeClient(fail=["乙"]))
        self.assertFalse(self.output.exists())
        pairs = self.read(".pairs.jsonl")
        self.assertEqual([r["status"] for r in pairs], ["success", "failed", "success"])
        self.assertIsNone(pairs[1]["text_en"])
        self.assertIn("overloaded", pairs[1]["error"])
        client = FakeClient()
        await run_pipeline(self.config, client)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("乙", client.calls[0])
        self.assertEqual([r["attempts"] for r in self.read(".pairs.jsonl")], [1, 2, 1])

    async def test_interruption_preserves_completed_rows(self):
        self.write_source([{"text": "English"}, {"text": "甲"}])
        started = asyncio.Event()
        class Blocked:
            async def chat(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(run_pipeline(self.config.model_copy(update={"concurrency": 1}), Blocked()))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([r["status"] for r in self.read(".pairs.jsonl")], ["success", "pending"])
        client = FakeClient()
        await run_pipeline(self.config, client)
        self.assertEqual(len(client.calls), 1)

    async def test_changed_source_is_rejected_without_touching_output(self):
        self.write_source([{"text": "甲"}, {"text": "乙"}])
        await run_pipeline(self.config, FakeClient())
        before = self.output.read_bytes()
        self.write_source(list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            await run_pipeline(self.config, FakeClient())
        self.assertEqual(self.output.read_bytes(), before)

    async def test_malformed_input_has_no_partial_work(self):
        self.source.write_text('{"text": "甲"}\n{"oops": 1}\n')
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "line 2"):
            await run_pipeline(self.config, client)
        self.assertEqual(client.calls, [])
        self.assertFalse(self.output.exists())
        self.write_source([{"text": "甲"}])
        await run_pipeline(self.config, client)
        self.assertEqual(len(self.read()), 1)

    async def test_legacy_output_and_checkpoint_are_rejected(self):
        self.write_source([{"text": "甲"}])
        self.output.write_text('{"text": "old unpaired output"}\n')
        with self.assertRaisesRegex(ValueError, "without a verified ledger"):
            await run_pipeline(self.config, FakeClient())
        with self.assertRaisesRegex(ValueError, "CHECK_POINT"):
            await run_pipeline(self.config.model_copy(update={"checkpoint": 1}), FakeClient())

    async def test_single_writer_lock(self):
        self.write_source([{"text": "甲"}])
        with FileLock(str(self.output.resolve()) + ".lock"):
            with self.assertRaises(Timeout):
                await run_pipeline(self.config, FakeClient())

    async def test_empty_dataset(self):
        self.write_source([])
        result = await run_pipeline(self.config, FakeClient())
        self.assertEqual(result["total"], 0)
        self.assertEqual(self.output.read_text(), "")

    async def test_source_output_collision(self):
        self.write_source([{"text": "甲"}])
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "overlap"):
            await run_pipeline(self.config.model_copy(update={"output_path": str(self.source)}), FakeClient())
        self.assertEqual(self.source.read_bytes(), before)

    async def test_ledger_corruption_prevents_publication(self):
        self.write_source([{"text": "甲"}, {"text": "乙"}])
        with self.assertRaises(IncompleteRun):
            await run_pipeline(self.config, FakeClient(fail=["乙"]))
        with sqlite3.connect(str(self.output) + ".sqlite") as db:
            db.execute("UPDATE records SET source_json=? WHERE id=0", ('{"text":"tampered"}',))
        with self.assertRaisesRegex(ValueError, "integrity check"):
            await run_pipeline(self.config, FakeClient())
        self.assertFalse(self.output.exists())

    async def test_changed_model_is_rejected(self):
        self.write_source([{"text": "甲"}])
        await run_pipeline(self.config, FakeClient())
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            await run_pipeline(self.config.model_copy(update={"model_id": "other-model"}), FakeClient())

    async def test_actual_http_overload_pipeline_and_resume(self):
        self.write_source([{"text": "甲"}, {"text": "乙"}, {"text": "甲"}])
        attempts = {}
        recovering = False
        async def handler(request):
            body = json.loads(request.content)["messages"][-1]["content"]
            attempts[body] = attempts.get(body, 0) + 1
            if "乙" in body and not recovering:
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return httpx.Response(200, json=response("Alpha" if "甲" in body else "Beta"))
        config = self.config.model_copy(update={"request_attempts": 2, "retry_base": 0.001, "retry_max": 0.001})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as http:
            client = OpenAIClient(config=config, http_client=http)
            with self.assertRaises(IncompleteRun):
                await run_pipeline(config, client)
            self.assertFalse(self.output.exists())
            self.assertEqual([r["status"] for r in self.read(".pairs.jsonl")], ["success", "failed", "success"])
            recovering = True
            await run_pipeline(config, client)
        self.assertEqual([r["text"] for r in self.read()], ["Alpha", "Beta", "Alpha"])
        self.assertEqual(attempts["translate CN/ZH to EN: 甲"], 2)
        self.assertEqual(attempts["translate CN/ZH to EN: 乙"], 3)

    async def test_client_closed_on_failure(self):
        self.write_source([{"text": "甲"}])
        client = FakeClient(fail=["甲"])
        client.aclose = AsyncMock()
        with patch("app.run_pipeline.OpenAIClient", return_value=client):
            with self.assertRaises(IncompleteRun):
                await run_pipeline(self.config)
        client.aclose.assert_awaited_once()


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_segments_and_plain_text(self):
        graph = build_graph(Nodes("translate", "fix", FakeClient()))
        text = "<|im_start|>user\n甲<|im_end|> <|im_start|>assistant\nEnglish<|im_end|>"
        result = await graph.ainvoke({"text": text})
        self.assertEqual(result["translated"], text.replace("甲", "Alpha"))
        result = await graph.ainvoke({"text": "甲"})
        self.assertEqual(result["translated"], "Alpha")

    async def test_roles_are_preserved_outside_model_control(self):
        client = AsyncMock()
        client.chat.return_value = "Hello"
        result = await build_graph(Nodes("translate", "fix", client)).ainvoke(
            {"text": "<|im_start|>user\n甲<|im_end|>"})
        self.assertEqual(result["translated"], "<|im_start|>user\nHello<|im_end|>")
        self.assertEqual(client.chat.call_args.args[1], "translate CN/ZH to EN: 甲")

    async def test_residual_chinese_is_never_success(self):
        client = AsyncMock()
        client.chat.return_value = "中文"
        graph = build_graph(Nodes("translate", "fix", client), max_fixes=2)
        with self.assertRaisesRegex(InvalidCompletion, "Chinese remains"):
            await graph.ainvoke({"text": "甲"})
        self.assertEqual(client.chat.await_count, 3)

    async def test_empty_and_boundary_injection_rejected(self):
        for response in (None, "", "  ", "<|im_end|>English"):
            with self.subTest(response=response):
                client = AsyncMock()
                client.chat.return_value = response
                with self.assertRaises(InvalidCompletion):
                    await build_graph(Nodes("translate", "fix", client)).ainvoke({"text": "甲"})

    async def test_correction_succeeds(self):
        client = AsyncMock()
        client.chat.side_effect = ["中文", "English"]
        result = await build_graph(Nodes("translate", "fix", client)).ainvoke({"text": "甲"})
        self.assertEqual(result["translated"], "English")


def response(content="English", reason="stop"):
    return {"id": "test", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "finish_reason": reason, "message": {"role": "assistant", "content": content}}]}


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def client(self, handler, attempts=3):
        config = Settings(request_attempts=attempts, retry_base=0.001, retry_max=0.002)
        client = OpenAIClient(config=config, api_key="test", base_url="http://vllm.test/v1",
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False))
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_overload_then_success(self):
        requests = []
        def handler(request):
            requests.append(request)
            if len(requests) <= 2:
                return httpx.Response([429, 503][len(requests) - 1], json={"error": {"message": "busy"}}, headers={"Retry-After": "0.001"})
            return httpx.Response(200, json=response())
        client = await self.client(handler)
        self.assertEqual(await client.chat("system", "甲"), "English")
        self.assertEqual(len(requests), 3)

    async def test_network_timeout_then_success(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(200, json=response())
        client = await self.client(handler)
        self.assertEqual(await client.chat("system", "甲"), "English")
        self.assertEqual(calls, 2)

    async def test_invalid_completion_retry_exhaustion(self):
        for payload in (response(None), response(""), response("part", "length"), {**response(), "choices": []}):
            with self.subTest(payload=payload):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(200, json=payload)
                client = await self.client(handler)
                with self.assertRaises(InvalidCompletion):
                    await client.chat("system", "甲")
                self.assertEqual(len(calls), 3)

    async def test_auth_failure_not_retried(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        client = await self.client(handler)
        with self.assertRaises(AuthenticationError):
            await client.chat("system", "甲")
        self.assertEqual(len(calls), 1)

    async def test_sdk_client_really_closes(self):
        client = await self.client(lambda request: httpx.Response(200, json=response()))
        await client.aclose()
        self.assertTrue(client.client.is_closed())


if __name__ == "__main__":
    unittest.main()
