import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .detector import has_chinese
from .vllm_client import InvalidCompletion, OpenAIClient

# Keep every conversation boundary byte-for-byte; plain text stays plain text.
SPECIAL_TOKEN = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")
ROLE_PREFIX = re.compile(r"^(?:system|user|assistant|tool)\r?\n")


class RecordState(TypedDict, total=False):
    text: str
    translated: str
    needs_fix: bool
    attempts: int


class Nodes:
    def __init__(self, translator_sys: str, fixer_sys: str, client: OpenAIClient, max_tokens=2048):
        self.translator_sys = translator_sys
        self.fixer_sys = fixer_sys
        self.client = client
        self.max_tokens = max_tokens

    async def translate_parts(self, text, prompt):
        parts = SPECIAL_TOKEN.split(text)
        for index in range(0, len(parts), 2):
            source = parts[index]
            if not has_chinese(source):
                continue
            role = ROLE_PREFIX.match(source) if index and parts[index - 1] == "<|im_start|>" else None
            prefix = role.group() if role else ""
            body = source[len(prefix):]
            result = await self.client.chat(prompt, "translate CN/ZH to EN: " + body, max_tokens=self.max_tokens)
            if not isinstance(result, str) or not result.strip() or SPECIAL_TOKEN.search(result):
                raise InvalidCompletion("Empty segment or unexpected conversation token")
            parts[index] = prefix + result.strip()
        return "".join(parts)

    async def detect_and_translate(self, state):
        return {"translated": await self.translate_parts(state["text"], self.translator_sys), "attempts": 0}

    async def validate(self, state):
        translated = state["translated"]
        if state["text"].strip() and not translated.strip():
            raise InvalidCompletion("Nonempty source has empty translation")
        if SPECIAL_TOKEN.findall(state["text"]) != SPECIAL_TOKEN.findall(translated):
            raise InvalidCompletion("Conversation boundaries changed")
        return {"needs_fix": has_chinese(translated)}

    async def fix_pass(self, state):
        return {"translated": await self.translate_parts(state["translated"], self.fixer_sys),
                "attempts": state["attempts"] + 1}


def build_graph(nodes: Nodes, max_fixes: int = 2):
    g = StateGraph(RecordState)
    g.add_node("detect_and_translate", nodes.detect_and_translate)
    g.add_node("validate", nodes.validate)
    g.add_node("fix_pass", nodes.fix_pass)
    g.add_edge(START, "detect_and_translate")
    g.add_edge("detect_and_translate", "validate")

    def route(state):
        if state["needs_fix"]:
            if state["attempts"] < max_fixes:
                return "fix_pass"
            raise InvalidCompletion("Chinese remains after all correction passes")
        return END

    g.add_conditional_edges("validate", route, {"fix_pass": "fix_pass", END: END})
    g.add_edge("fix_pass", "validate")
    return g.compile()
