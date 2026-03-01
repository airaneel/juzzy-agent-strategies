import json
import re
from collections.abc import Generator
from enum import Enum, auto
from typing import Union

from dify_plugin.entities.model.llm import LLMResultChunk, LLMUsage
from dify_plugin.interfaces.agent import AgentScratchpadUnit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"(?:^|(?<= ))(Thought:|FinalAnswer:|Action:)",
    re.IGNORECASE | re.MULTILINE,
)
_HOLD_BACK = len("FinalAnswer:")  # keep tail to avoid splitting a partial prefix
_JSON_DECODER = json.JSONDecoder(strict=False)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ReactState(Enum):
    THINKING = auto()
    ANSWER = auto()


_STATE_MAP = {
    "thought": ReactState.THINKING,
    "finalanswer": ReactState.ANSWER,
}


class ReactChunk:
    __slots__ = ("state", "content")

    def __init__(self, state: ReactState, content: str):
        self.state = state
        self.content = content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content(chunk: LLMResultChunk) -> str:
    """Extract text from an LLM result chunk."""
    raw = chunk.delta.message.content
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = getattr(item, "data", None) or getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _strip_think_tags(text: str, in_think: bool) -> tuple[str, bool]:
    """Strip <think>…</think> blocks, handling chunks that span tags.

    Returns (cleaned_text, still_inside_think).
    """
    if not in_think and "<" not in text:
        return text, False

    result: list[str] = []
    pos = 0
    while pos < len(text):
        if in_think:
            end = text.find("</think>", pos)
            if end == -1:
                break  # rest consumed
            in_think = False
            pos = end + len("</think>")
        else:
            start = text.find("<think>", pos)
            if start == -1:
                result.append(text[pos:])
                break
            result.append(text[pos:start])
            in_think = True
            pos = start + len("<think>")

    return "".join(result), in_think


def _parse_action(
    json_str: str,
) -> Union[AgentScratchpadUnit.Action, ReactChunk]:
    """Parse JSON into an Action, or fall back to a ReactChunk."""
    try:
        obj = json.loads(json_str, strict=False)
        if isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]
        if isinstance(obj, dict):
            name = input_val = None
            for key, value in obj.items():
                if "input" in key.lower():
                    input_val = value
                else:
                    name = value
            if name is not None and input_val is not None:
                return AgentScratchpadUnit.Action(
                    action_name=name, action_input=input_val
                )
    except (json.JSONDecodeError, AttributeError):
        pass
    return ReactChunk(ReactState.THINKING, json_str or "")


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------


class ReactStreamParser:
    """Streaming parser for ReAct LLM output.

    Buffers text, finds ``Thought:``, ``FinalAnswer:``, ``Action:`` via
    regex, and extracts action JSON with ``json.JSONDecoder.raw_decode``.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._state = ReactState.THINKING
        self._in_think = False

    def parse(
        self,
        llm_response: Generator[LLMResultChunk, None, None],
        usage_dict: dict[str, LLMUsage | None],
    ) -> Generator[Union[ReactChunk, AgentScratchpadUnit.Action], None, None]:
        for chunk in llm_response:
            if chunk.delta.usage:
                usage_dict["usage"] = chunk.delta.usage
            text = _extract_content(chunk)
            if text:
                text, self._in_think = _strip_think_tags(text, self._in_think)
            if text:
                self._buf += text
                yield from self._drain()
        yield from self._drain(final=True)

    def _drain(
        self, final: bool = False
    ) -> Generator[Union[ReactChunk, AgentScratchpadUnit.Action], None, None]:
        while True:
            m = _PREFIX_RE.search(self._buf)
            if not m:
                break

            keyword = m.group(1).lower().rstrip(":")
            before = self._buf[: m.start()]
            after = self._buf[m.end() :]

            if keyword == "action":
                action, remainder = self._try_parse_action(after, final)
                if remainder is None:
                    # JSON incomplete — emit text before prefix, hold the rest
                    yield from self._emit(before)
                    self._buf = self._buf[len(before) :]
                    return
                yield from self._emit(before)
                self._buf = remainder
                if action is not None:
                    yield action
            else:
                yield from self._emit(before)
                self._buf = after
                if keyword in _STATE_MAP:
                    self._state = _STATE_MAP[keyword]

        if final:
            yield from self._emit(self._buf)
            self._buf = ""
        else:
            safe = max(0, len(self._buf) - _HOLD_BACK)
            if safe > 0:
                yield from self._emit(self._buf[:safe])
                self._buf = self._buf[safe:]

    def _emit(self, text: str) -> Generator[ReactChunk, None, None]:
        if text:
            yield ReactChunk(self._state, text)

    @staticmethod
    def _try_parse_action(
        text: str, final: bool
    ) -> tuple[Union[AgentScratchpadUnit.Action, ReactChunk, None], str | None]:
        """Extract action JSON from text after ``Action:``.

        Returns ``(result, remaining_text)`` on success, or
        ``(None, None)`` when more data is needed.
        """
        stripped = text.lstrip()
        if not stripped:
            return (None, text) if final else (None, None)
        if stripped[0] not in ("{", "["):
            return None, text  # not JSON — skip prefix
        try:
            _, end = _JSON_DECODER.raw_decode(stripped)
            ws = len(text) - len(stripped)
            return _parse_action(stripped[:end]), text[ws + end :]
        except json.JSONDecodeError:
            if final:
                return _parse_action(stripped), ""
            return None, None
