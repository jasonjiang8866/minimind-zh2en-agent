"""Synthetic accounting regressions, not a vLLM throughput benchmark.

The default suite uses real LangGraph, SQLite durability and atomic exports.
Only model responses are simulated. Larger HTTP runs are explicitly opt-in.
"""
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.config import Settings
from app.run_pipeline import IncompleteRun, run_pipeline
from app.vllm_client import OpenAIClient


PREFIX = "translate CN/ZH to EN: "


def source_rows(count):
    return [{"text": f"甲 {i}", "metadata": {"index": i, "tags": ["合成", i % 7]}}
            for i in range(count)]


def completion(text):
    return {"id": "synthetic", "object": "chat.completion", "created": 0,
            "model": "synthetic", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": text}}]}


class ScaleIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Debug tracing is expensive for tens of thousands of graph tasks. This
        # does not bypass graph execution, SQLite FULL sync, retries or exports.
        loop = asyncio.get_running_loop()
        previous_debug = loop.get_debug()
        loop.set_debug(False)
        self.addCleanup(loop.set_debug, previous_debug)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.source = Path(temp.name) / "source.jsonl"
        self.output = Path(temp.name) / "result.jsonl"
        self.config = Settings(
            input_file=str(self.source), output_path=str(self.output), checkpoint=0,
            concurrency=16, request_attempts=3, retry_base=0.001, retry_max=0.001,
            request_timeout=1, max_fixes=0, max_tokens=64,
            vllm_base_url="http://synthetic.invalid/v1", vllm_api_key="synthetic",
            model_id="synthetic",
        )

    def write_source(self, rows):
        self.rows = rows
        self.source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                               encoding="utf-8")

    def assert_accounting(self, failed, attempts):
        """Independently recompute hashes and compare every positional pair."""
        count = len(self.rows)
        pairs_path = Path(str(self.output) + ".pairs.jsonl")
        lines = pairs_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), count)
        pairs = [json.loads(line) for line in lines]
        self.assertEqual([p["source_id"] for p in pairs], list(range(count)))
        for i, (row, pair) in enumerate(zip(self.rows, pairs, strict=True)):
            self.assertEqual(pair["source"], row)
            canonical = json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
            self.assertEqual(pair["source_sha256"], hashlib.sha256(canonical).hexdigest())
            self.assertEqual(pair["attempts"], attempts[i])
            if i in failed:
                self.assertEqual(pair["status"], "failed")
                self.assertIsNone(pair["text_en"])
                self.assertTrue(pair["error"])
            else:
                self.assertEqual(pair["status"], "success")
                self.assertEqual(pair["text_en"], row["text"].replace("甲", "Alpha"))
                self.assertIsNone(pair["error"])
        summary = json.loads(Path(str(self.output) + ".summary.json").read_text())
        self.assertEqual(summary, {
            "status": "incomplete" if failed else "complete",
            "input_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "total": count, "success": count - len(failed), "failed": len(failed), "pending": 0,
        })
        # Check the durable state, not only its JSONL projection.
        with sqlite3.connect(str(self.output) + ".sqlite") as db:
            records = db.execute("SELECT id, source_json, source_sha256, status, translated, attempts, error "
                                 "FROM records ORDER BY id").fetchall()
        self.assertEqual(len(records), count)
        for record, pair in zip(records, pairs, strict=True):
            rid, raw, sha, status, translated, tries, error = record
            self.assertEqual((rid, json.loads(raw), sha, status, translated, tries, error),
                             (pair["source_id"], pair["source"], pair["source_sha256"],
                              pair["status"], pair["text_en"], pair["attempts"], pair["error"]))
        if not failed:
            output_lines = self.output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(output_lines), count)
            self.assertEqual([json.loads(line) for line in output_lines],
                             [{**row, "text": row["text"].replace("甲", "Alpha")} for row in self.rows])
        return pairs

    async def http_overload_and_resume(self, count):
        self.write_source(source_rows(count))
        calls = Counter()
        faults = Counter()
        failed = {i for i in range(count) if i % 20 in (4, 5, 6, 7)}
        recovering = False
        active = peak = 0
        completed = []
        later_row_finished = asyncio.Event()

        async def handler(request):
            nonlocal active, peak
            self.assertEqual(str(request.url), "http://synthetic.invalid/v1/chat/completions")
            text = json.loads(request.content)["messages"][-1]["content"].removeprefix(PREFIX)
            rid = int(text.split()[-1])
            calls[rid] += 1
            active += 1
            peak = max(peak, active)
            try:
                # Force overlap and a known out-of-order completion without
                # depending on wall-clock timing or random fault selection.
                if rid == 0 and not recovering:
                    await asyncio.wait_for(later_row_finished.wait(), timeout=10)
                await asyncio.sleep(0)
                bucket = rid % 20
                if not recovering and (rid in failed or (bucket < 4 and calls[rid] == 1)):
                    fault = bucket % 4
                    faults[fault] += 1
                    if fault in (0, 1):
                        return httpx.Response(429 if fault == 0 else 503,
                                              json={"error": {"message": "synthetic overload"}},
                                              headers={"Retry-After": "0"})
                    if fault == 2:
                        raise httpx.ReadTimeout("synthetic read timeout", request=request)
                    return httpx.Response(200, json=completion(""))
                completed.append(rid)
                if rid == 8:
                    later_row_finished.set()
                return httpx.Response(200, json=completion(text.replace("甲", "Alpha")))
            finally:
                active -= 1

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as http:
            client = OpenAIClient(config=self.config, http_client=http)
            self.addAsyncCleanup(client.aclose)
            # No stochastic waiting or hundreds of expected warning lines in CI.
            # The real retry loop and exact request budget remain under test.
            with patch.object(client, "retry_delay", return_value=0), \
                    patch("app.vllm_client.logger.warning"), patch("app.run_pipeline.logger.warning"):
                with self.assertRaises(IncompleteRun):
                    await run_pipeline(self.config, client)
                self.assertFalse(self.output.exists())
                self.assert_accounting(failed, [1] * count)
                expected = Counter({i: 3 if i in failed else (2 if i % 20 < 4 else 1)
                                    for i in range(count)})
                self.assertEqual(calls, expected)
                self.assertEqual(faults, Counter({i: 4 * (count // 20) for i in range(4)}))
                self.assertLess(completed.index(8), completed.index(0))
                self.assertGreater(peak, 1)
                self.assertLessEqual(peak, self.config.concurrency)
                self.assertEqual(active, 0)

                # A stale prior training artifact must also survive an incomplete
                # retry byte-for-byte. It is a fixture, never reported complete.
                stale = b'{"text":"previous complete dataset"}\n'
                self.output.write_bytes(stale)
                with self.assertRaises(IncompleteRun):
                    await run_pipeline(self.config, client)
                self.assertEqual(self.output.read_bytes(), stale)
                self.assert_accounting(failed, [2 if i in failed else 1 for i in range(count)])
                expected.update({i: 3 for i in failed})
                self.assertEqual(calls, expected)

                recovering = True
                await run_pipeline(self.config, client)
                expected.update({i: 1 for i in failed})
                self.assertEqual(calls, expected)  # No successful row retranslates.
                self.assert_accounting(set(), [3 if i in failed else 1 for i in range(count)])
                before = self.output.read_bytes()
                await run_pipeline(self.config, client)
                self.assertEqual(calls, expected)
                self.assertEqual(self.output.read_bytes(), before)
                self.assert_accounting(set(), [3 if i in failed else 1 for i in range(count)])

    async def test_1000_http_overload_and_resume(self):
        await asyncio.wait_for(self.http_overload_and_resume(1000), timeout=120)

    async def test_above_10000_checkpoint_boundary(self):
        async def scenario():
            count = 10_017
            rows = source_rows(count)
            # Exact duplicates straddle the historical skip boundary, including
            # nested metadata. Positional identity must retain all four copies.
            duplicates = (42, 9_999, 10_000, 10_001)
            for i in duplicates[1:]:
                rows[i] = json.loads(json.dumps(rows[duplicates[0]]))
            self.write_source(rows)
            failed = {0, 9_998, 10_002, count - 1}
            blocked_texts = {rows[i]["text"] for i in failed}
            calls = Counter()

            class Client:
                async def chat(inner, system, text, **kwargs):
                    source = text.removeprefix(PREFIX)
                    calls[source] += 1
                    await asyncio.sleep(0)
                    if source in blocked_texts:
                        raise RuntimeError("synthetic exhausted request")
                    return source.replace("甲", "Alpha")

            client = Client()
            with patch("app.run_pipeline.logger.warning"):
                with self.assertRaises(IncompleteRun):
                    await run_pipeline(self.config, client)
            self.assertFalse(self.output.exists())
            self.assert_accounting(failed, [1] * count)
            expected = Counter(row["text"] for row in rows)
            self.assertEqual(calls, expected)
            self.assertEqual(calls[rows[42]["text"]], len(duplicates))
            blocked_texts.clear()
            await run_pipeline(self.config, client)
            expected.update(rows[i]["text"] for i in failed)
            self.assertEqual(calls, expected)
            pairs = self.assert_accounting(set(), [2 if i in failed else 1 for i in range(count)])
            self.assertEqual(len({pairs[i]["source_sha256"] for i in duplicates}), 1)
            self.assertEqual([pairs[i]["source_id"] for i in duplicates], list(duplicates))
            before = self.output.read_bytes()
            await run_pipeline(self.config, client)
            self.assertEqual(calls, expected)
            self.assertEqual(self.output.read_bytes(), before)
            self.assert_accounting(set(), [2 if i in failed else 1 for i in range(count)])

        await asyncio.wait_for(scenario(), timeout=120)

    @unittest.skipUnless(os.getenv("RUN_INTEGRITY_STRESS") == "1", "optional synthetic HTTP stress")
    async def test_optional_large_http_stress(self):
        count = int(os.getenv("INTEGRITY_STRESS_RECORDS", "20000"))
        if not 1000 <= count <= 100000 or count % 20:
            self.fail("INTEGRITY_STRESS_RECORDS must be a multiple of 20 between 1000 and 100000")
        await asyncio.wait_for(self.http_overload_and_resume(count), timeout=900)


if __name__ == "__main__":
    unittest.main()
