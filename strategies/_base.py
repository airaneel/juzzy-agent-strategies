"""Shared utilities for agent strategy implementations."""

import json
import os
import time
from collections.abc import Generator
from typing import Any, TypeVar, cast

from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.interfaces.agent import AgentModelConfig, AgentStrategy, ToolEntity
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class LogMetadata:
    """Metadata keys for logging."""

    STARTED_AT = "started_at"
    PROVIDER = "provider"
    FINISHED_AT = "finished_at"
    ELAPSED_TIME = "elapsed_time"
    TOTAL_PRICE = "total_price"
    CURRENCY = "currency"
    TOTAL_TOKENS = "total_tokens"


# ---------------------------------------------------------------------------
# Shared Pydantic models
# ---------------------------------------------------------------------------


class ContextItem(BaseModel):
    content: str
    title: str
    metadata: dict[str, Any]


class AgentParams(BaseModel):
    """Common parameters for all agent strategy invocations."""

    query: str
    instruction: str | None = None
    model: AgentModelConfig
    tools: list[ToolEntity] | None = None
    maximum_iterations: int = 3
    context: list[ContextItem] | None = None


class ExecutionMetadata(BaseModel):
    """Execution metadata with default values.

    Pydantic v2 coerces Decimal→float automatically based on field types.
    extra="ignore" filters out fields from LLMUsage that don't exist here.
    """

    model_config = ConfigDict(extra="ignore")

    total_price: float = 0.0
    currency: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_unit_price: float = 0.0
    prompt_price_unit: float = 0.0
    prompt_price: float = 0.0
    completion_tokens: int = 0
    completion_unit_price: float = 0.0
    completion_price_unit: float = 0.0
    completion_price: float = 0.0
    latency: float = 0.0

    @classmethod
    def from_llm_usage(cls, usage: LLMUsage | None) -> "ExecutionMetadata":
        if usage is None:
            return cls()
        return cls.model_validate(usage.model_dump())


# ---------------------------------------------------------------------------
# Usage accumulator (required by AgentStrategy.increase_usage() interface)
# ---------------------------------------------------------------------------

UsageAccumulator = dict[str, LLMUsage | None]


def create_usage_accumulator() -> UsageAccumulator:
    """Create a fresh usage accumulator for AgentStrategy.increase_usage()."""
    return {"usage": None}


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------

_Y = TypeVar("_Y")
_R = TypeVar("_R")


def consume_generator(
    gen: Generator[_Y, None, _R],
) -> tuple[list[_Y], _R]:
    """Consume a generator, collecting yielded items and capturing return value."""
    items: list[_Y] = []
    try:
        while True:
            items.append(next(gen))
    except StopIteration as e:
        return items, e.value


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def build_usage_metadata(usage: LLMUsage | None) -> dict[str, Any]:
    """Build the 3-field usage metadata dict, handling None."""
    return {
        LogMetadata.TOTAL_PRICE: usage.total_price if usage else 0,
        LogMetadata.CURRENCY: usage.currency if usage else "",
        LogMetadata.TOTAL_TOKENS: usage.total_tokens if usage else 0,
    }


def build_timing_metadata(
    started_at: float,
    *,
    provider: str = "",
    usage: LLMUsage | None = None,
) -> dict[str, Any]:
    """Build a full timing + usage metadata dict for finish_log_message calls."""
    now = time.perf_counter()
    meta: dict[str, Any] = {
        LogMetadata.STARTED_AT: started_at,
        LogMetadata.FINISHED_AT: now,
        LogMetadata.ELAPSED_TIME: now - started_at,
    }
    if provider:
        meta[LogMetadata.PROVIDER] = provider
    meta.update(build_usage_metadata(usage))
    return meta


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


def extract_tool_calls_from_message(
    tool_calls_list: list,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Extract (tool_call_id, name, args) from AssistantPromptMessage.ToolCall objects.

    Works for both streaming (.delta.message.tool_calls) and
    blocking (.message.tool_calls) responses.
    """
    result: list[tuple[str, str, dict[str, Any]]] = []
    for tc in tool_calls_list:
        args: dict[str, Any] = {}
        if tc.function.arguments != "":
            args = json.loads(tc.function.arguments)
        result.append((tc.id, tc.function.name, args))
    return result


# ---------------------------------------------------------------------------
# Tool response processing
# ---------------------------------------------------------------------------


def process_tool_invoke_responses(
    responses: Any,
    strategy: AgentStrategy,
) -> tuple[str, list[ToolInvokeMessage]]:
    """Process tool invoke responses into a text result and additional messages.

    Handles TEXT, LINK, IMAGE_LINK, IMAGE, JSON, BLOB message types.
    Returns (result_text, additional_messages_to_yield).
    """
    result = ""
    additional_messages: list[ToolInvokeMessage] = []

    for response in responses:
        msg_type = response.type

        if msg_type == ToolInvokeMessage.MessageType.TEXT:
            result += cast(ToolInvokeMessage.TextMessage, response.message).text

        elif msg_type == ToolInvokeMessage.MessageType.LINK:
            link_text = cast(ToolInvokeMessage.TextMessage, response.message).text
            result += f"result link: {link_text}. please tell user to check it."

        elif msg_type in {
            ToolInvokeMessage.MessageType.IMAGE_LINK,
            ToolInvokeMessage.MessageType.IMAGE,
        }:
            if hasattr(response.message, "text"):
                file_info = cast(ToolInvokeMessage.TextMessage, response.message).text
                # Try to create a blob from local file (plugin daemon environment)
                if file_info.startswith("/files/") and os.path.exists(file_info):
                    try:
                        with open(file_info, "rb") as f:
                            file_content = f.read()
                        additional_messages.append(
                            strategy.create_blob_message(
                                blob=file_content,
                                meta={
                                    "mime_type": "image/png",
                                    "filename": os.path.basename(file_info),
                                },
                            )
                        )
                    except OSError:
                        pass
            additional_messages.append(response)
            # LLM-facing instruction (not user-facing)
            result += (
                "image has been created and sent to user already, "
                "you do not need to create it, just tell the user to check it now."
            )

        elif msg_type == ToolInvokeMessage.MessageType.JSON:
            text = json.dumps(
                cast(ToolInvokeMessage.JsonMessage, response.message).json_object,
                ensure_ascii=False,
            )
            result += f"tool response: {text}."

        elif msg_type == ToolInvokeMessage.MessageType.BLOB:
            result += "Generated file ... "
            additional_messages.append(response)

        else:
            result += f"tool response: {response.message!r}."

    return result, additional_messages


# ---------------------------------------------------------------------------
# Retriever resources
# ---------------------------------------------------------------------------


def build_retriever_resources(
    context: list[ContextItem],
) -> list[ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource]:
    """Build RetrieverResource list from ContextItems."""
    Resource = ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource
    return [
        Resource(
            content=ctx.content,
            position=ctx.metadata.get("position"),
            dataset_id=ctx.metadata.get("dataset_id"),
            dataset_name=ctx.metadata.get("dataset_name"),
            document_id=ctx.metadata.get("document_id"),
            document_name=ctx.metadata.get("document_name"),
            data_source_type=ctx.metadata.get("document_data_source_type"),
            segment_id=ctx.metadata.get("segment_id"),
            retriever_from=ctx.metadata.get("retriever_from"),
            score=ctx.metadata.get("score"),
            hit_count=ctx.metadata.get("segment_hit_count"),
            word_count=ctx.metadata.get("segment_word_count"),
            segment_position=ctx.metadata.get("segment_position"),
            index_node_hash=ctx.metadata.get("segment_index_node_hash"),
            page=ctx.metadata.get("page"),
            doc_metadata=ctx.metadata.get("doc_metadata"),
        )
        for ctx in context
    ]
