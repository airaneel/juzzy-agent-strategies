"""Shared utilities for agent strategy implementations."""

import json
import os
import time
from collections.abc import Generator, Iterable
from decimal import Decimal
from typing import Any, NamedTuple, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.model.message import AssistantPromptMessage
from dify_plugin.entities.provider_config import LogMetadata
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.interfaces.agent import AgentStrategy
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Log metadata
# ---------------------------------------------------------------------------


_WALL_OFFSET = time.time() - time.perf_counter()


def perf_to_wall(perf: float) -> float:
    return perf + _WALL_OFFSET


def finish_log_metadata(
    started_at: float,
    *,
    provider: str = "",
    usage: LLMUsage | None = None,
) -> dict[LogMetadata, str | int | float | Decimal]:
    """Build timing + usage metadata dict for finish_log_message calls."""
    now = time.perf_counter()
    total_price = usage.total_price if usage else 0
    meta: dict[LogMetadata, str | int | float | Decimal] = {
        LogMetadata.STARTED_AT: perf_to_wall(started_at),
        LogMetadata.FINISHED_AT: perf_to_wall(now),
        LogMetadata.ELAPSED_TIME: now - started_at,
        LogMetadata.TOTAL_PRICE: total_price,
        LogMetadata.CURRENCY: usage.currency if usage else "",
        LogMetadata.TOTAL_TOKENS: usage.total_tokens if usage else 0,
    }
    if provider:
        meta[LogMetadata.PROVIDER] = provider
    return meta


# ---------------------------------------------------------------------------
# Shared Pydantic models
# ---------------------------------------------------------------------------


class ContextItem(BaseModel):
    content: str
    title: str
    metadata: dict[str, Any]


def build_execution_metadata(
    usage: LLMUsage | None, *, usd_to_rub: float = 0
) -> dict[str, object]:
    """Dump LLMUsage to dict and add rub price fields."""
    if usage is None:
        return {}
    data: dict[str, object] = usage.model_dump()
    if usd_to_rub > 0:
        data["total_price_rub"] = round(float(usage.total_price) * usd_to_rub, 6)
        data["prompt_price_rub"] = round(float(usage.prompt_price) * usd_to_rub, 6)
        data["completion_price_rub"] = round(
            float(usage.completion_price) * usd_to_rub, 6
        )
    return data


# ---------------------------------------------------------------------------
# Final metadata (shared by all strategies)
# ---------------------------------------------------------------------------


def emit_final_metadata(
    strategy: AgentStrategy,
    context: list["ContextItem"] | None,
    llm_usage: dict[str, LLMUsage | None],
    usd_to_rub: float,
) -> Generator[AgentInvokeMessage, None, None]:
    """Yield retriever resources and execution metadata at end of invocation."""
    if isinstance(context, list):
        yield strategy.create_retriever_resource_message(
            retriever_resources=build_retriever_resources(context),
            context="",
        )
    yield strategy.create_json_message(
        {
            "execution_metadata": build_execution_metadata(
                llm_usage["usage"], usd_to_rub=usd_to_rub
            )
        }
    )


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


class ToolCall(NamedTuple):
    id: str
    name: str
    args: dict[str, object]


def extract_tool_calls_from_message(
    tool_calls_list: list[AssistantPromptMessage.ToolCall],
) -> list[ToolCall]:
    """Extract parsed ToolCall objects from AssistantPromptMessage.ToolCall list."""
    result: list[ToolCall] = []
    for tc in tool_calls_list:
        args: dict[str, object] = {}
        if tc.function.arguments != "":
            args = json.loads(tc.function.arguments)
        result.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
    return result


# ---------------------------------------------------------------------------
# Tool response processing
# ---------------------------------------------------------------------------


def process_tool_invoke_responses(
    responses: Iterable[ToolInvokeMessage],
    strategy: AgentStrategy,
) -> tuple[str, list[AgentInvokeMessage]]:
    """Process tool invoke responses into a text result and additional messages.

    Handles TEXT, LINK, IMAGE_LINK, IMAGE, JSON, BLOB message types.
    Returns (result_text, additional_messages_to_yield).
    """
    result = ""
    additional_messages: list[AgentInvokeMessage] = []

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
            additional_messages.append(cast(AgentInvokeMessage, response))
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
            additional_messages.append(cast(AgentInvokeMessage, response))

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
