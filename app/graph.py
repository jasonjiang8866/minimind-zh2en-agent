import os
import aiofiles
import asyncio
from typing import TypedDict, Optional, Dict
from langgraph.graph import StateGraph, START, END
from .detector import has_chinese
from .vllm_client import OpenAIClient

# ---- State ----
class RecordState(TypedDict, total=False):
    record: Dict
    text: str
    translated: str
    needs_fix: bool
    attempts: int
    error: Optional[str]

# ---- Nodes ----
class Nodes:
    def __init__(self, translator_sys: str, fixer_sys: str, out_path: str, write_lock: asyncio.Lock, client: OpenAIClient):
        self.translator_sys = translator_sys
        self.fixer_sys = fixer_sys
        self.out_path = out_path
        self.write_lock = write_lock
        self.client = client

    async def detect_and_translate(self, state: RecordState) -> RecordState:
        state.setdefault("attempts", 0)
        text = state["text"]
        if not has_chinese(text):
            state["translated"] = text
            state["needs_fix"] = False
            return state
        # translate only if needed
        texts = text.split("<|im_end|> <|im_start|>")
        out_array = []
        for single_text in texts:
            single_out = await self.client.chat(self.translator_sys, "translate CN/ZH to EN: " + single_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip(), max_tokens=5120) or ""
            out_array.append(single_out.strip())
        out = "<|im_start|>" + "<|im_end|> <|im_start|>".join(out_array)  + "<|im_end|>"
        state["translated"] = out.strip()
        state["needs_fix"] = has_chinese(state["translated"])
        return state

    async def validate(self, state: RecordState) -> RecordState:
        state["needs_fix"] = has_chinese(state["translated"])
        return state

    async def fix_pass(self, state: RecordState) -> RecordState:
        state["attempts"] = int(state.get("attempts", 0)) + 1
        texts = state["translated"].split("<|im_end|> <|im_start|>")
        fixed_array = []
        for single_text in texts:
            single_fixed = await self.client.chat(self.fixer_sys, "translate CN/ZH to EN: " + single_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip(), max_tokens=5120) or ""
            fixed_array.append(single_fixed.strip())
        state["translated"] = "<|im_start|>" + "<|im_end|> <|im_start|>".join(fixed_array)  + "<|im_end|>"
        state["needs_fix"] = has_chinese(state["translated"])
        return state

    async def write_jsonl(self, state: RecordState) -> RecordState:
        # Output schema mirrors the input but replaces "text" with English text under "text_en"
        out_obj = {"text":  state["translated"]}
        # out_obj = {**state["record"], "text_en": state["translated"]}
        line = (await asyncio.to_thread(lambda: __import__("json").dumps(out_obj, ensure_ascii=False))) + "\n"
        async with self.write_lock:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            async with aiofiles.open(self.out_path, "a", encoding="utf-8") as f:
                await f.write(line)
        return state

# ---- Graph factory ----
def build_graph(nodes: Nodes, max_fixes: int = 1):
    g = StateGraph(RecordState)

    g.add_node("detect_and_translate", nodes.detect_and_translate)
    g.add_node("validate", nodes.validate)
    g.add_node("fix_pass", nodes.fix_pass)
    g.add_node("write_jsonl", nodes.write_jsonl)

    g.add_edge(START, "detect_and_translate")
    g.add_edge("detect_and_translate", "validate")

    # conditional routing
    def needs_fix_router(state: RecordState):
        if state.get("needs_fix"):
            # retry fix if attempts < max_fixes
            if int(state.get("attempts", 0)) < max_fixes:
                return "fix_pass"
        return "write_jsonl"

    g.add_conditional_edges("validate", needs_fix_router, {"fix_pass": "fix_pass", "write_jsonl": "write_jsonl"})
    g.add_edge("fix_pass", "validate")
    g.add_edge("write_jsonl", END)

    return g.compile()
