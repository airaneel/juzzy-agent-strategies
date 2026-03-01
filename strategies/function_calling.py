import json
import re
import time
from collections.abc import Generator, Iterator
from copy import deepcopy
from typing import Any, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.provider_config import LogMetadata
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentStrategy,
    ToolEntity,
    ToolInvokeMeta,
)

from strategies._base import (
    ContextItem,
    ToolCall,
    emit_final_metadata,
    extract_tool_calls_from_message,
    finish_log_metadata,
    perf_to_wall,
    process_tool_invoke_responses,
)


class FunctionCallingAgentStrategy(AgentStrategy):
    _LLMCallResult = tuple[
        list[AgentInvokeMessage],
        str,
        list[ToolCall],
        LLMUsage | None,
    ]

    def _invoke(
        self, parameters: dict[str, object]
    ) -> Generator[AgentInvokeMessage, None, None]:
        self._model = AgentModelConfig.model_validate(parameters["model"])
        self._llm_config = LLMModelConfig.model_validate(parameters["model"])
        raw_tools = cast(list[dict[str, Any]] | None, parameters.get("tools"))
        tools = [ToolEntity.model_validate(t) for t in raw_tools] if raw_tools else None

        self._llm_usage: dict[str, LLMUsage | None] = {"usage": None}
        self._history_messages = self._model.history_prompt_messages
        self._history_messages.insert(
            0, SystemPromptMessage(content=cast(str, parameters.get("instruction", "")))
        )
        self._history_messages.append(
            UserPromptMessage(content=cast(str, parameters["query"]))
        )
        self._thoughts: list[PromptMessage] = []
        self._tool_instances = {t.identity.name: t for t in tools} if tools else {}
        self._prompt_tools = self._init_prompt_tools(tools)

        yield from self._run(
            context=cast(list[ContextItem] | None, parameters.get("context")),
            max_iterations=cast(int, parameters.get("maximum_iterations", 3)),
            usd_to_rub=cast(float, parameters.get("usd_to_rub", 0)),
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(
        self,
        *,
        context: list[ContextItem] | None,
        max_iterations: int,
        usd_to_rub: float,
    ) -> Generator[AgentInvokeMessage, None, None]:
        for step in range(1, max_iterations + 1):
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {step}",
                data={},
                metadata={LogMetadata.STARTED_AT: perf_to_wall(round_started_at)},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            text_messages, response, tool_calls, usage = yield from self._invoke_model(
                round_log
            )

            if not tool_calls:
                yield from text_messages
                yield self.finish_log_message(
                    log=round_log,
                    data={"output": response},
                    metadata=finish_log_metadata(round_started_at, usage=usage),
                )
                break

            yield from self._handle_tool_round(tool_calls, response, round_log)

            yield self.finish_log_message(
                log=round_log,
                data={},
                metadata=finish_log_metadata(round_started_at, usage=usage),
            )

            if step == max_iterations:
                yield from text_messages
                self._thoughts.append(
                    UserPromptMessage(
                        content=(
                            "You have reached the maximum number of tool calls. "
                            "You cannot call any more tools. Based on all the "
                            "information you have gathered above, provide your "
                            "best final answer now."
                        )
                    )
                )
                yield from self._run_summary_call()
                break

            for prompt_tool in self._prompt_tools:
                self.update_prompt_message_tool(
                    self._tool_instances[prompt_tool.name], prompt_tool
                )

        yield from emit_final_metadata(self, context, self._llm_usage, usd_to_rub)

    # ------------------------------------------------------------------
    # Model invocation
    # ------------------------------------------------------------------

    def _invoke_model(
        self,
        round_log: AgentInvokeMessage,
    ) -> Generator[AgentInvokeMessage, None, _LLMCallResult]:
        prompt_messages = self._organize_prompt_messages()
        if self._model.entity and self._model.completion_params:
            self.recalc_llm_max_tokens(
                self._model.entity,
                prompt_messages,
                self._model.completion_params,
            )
        model_started_at = time.perf_counter()

        model_log = self.create_log_message(
            label=f"{self._model.model} Thought",
            data={},
            metadata={
                LogMetadata.STARTED_AT: perf_to_wall(model_started_at),
                LogMetadata.PROVIDER: self._model.provider,
            },
            parent=round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield model_log

        text_messages, response, tool_calls, usage = self._call_llm(
            prompt_messages, tools=self._prompt_tools
        )

        # Only include full response in model log when there are tool calls.
        # For the final answer (no tool calls), the round log carries the output
        # to avoid the SDK duplicating it via parent aggregation.
        if tool_calls:
            model_data: dict[str, Any] = {
                "response": response,
                "tool_name": ";".join(tc.name for tc in tool_calls),
                "tool_input": [{"name": tc.name, "args": tc.args} for tc in tool_calls],
            }
        else:
            model_data = {}

        yield self.finish_log_message(
            log=model_log,
            data=model_data,
            metadata=finish_log_metadata(
                model_started_at,
                provider=self._model.provider,
                usage=usage,
            ),
        )

        return text_messages, response, tool_calls, usage

    def _call_llm(
        self,
        prompt_messages: list[PromptMessage],
        *,
        tools: list[PromptMessageTool] | None = None,
    ) -> _LLMCallResult:
        can_stream = (
            ModelFeature.STREAM_TOOL_CALL in self._model.entity.features
            if self._model.entity and self._model.entity.features
            else False
        )
        stop = (
            self._model.completion_params.get("stop", [])
            if self._model.completion_params
            else []
        )
        result = self.session.model.llm.invoke(
            model_config=self._llm_config,
            prompt_messages=prompt_messages,
            stop=stop,
            stream=can_stream,
            tools=tools if tools is not None else [],
        )

        chunks: Iterator[LLMResultChunk]
        if isinstance(result, LLMResult):
            chunks = iter([result.to_llm_result_chunk()])
        else:
            chunks = result

        response = ""
        text_messages: list[AgentInvokeMessage] = []
        tool_calls: list[ToolCall] = []
        last_usage: LLMUsage | None = None

        for chunk in chunks:
            if chunk.delta.message.tool_calls:
                tool_calls.extend(
                    extract_tool_calls_from_message(chunk.delta.message.tool_calls)
                )
            if chunk.delta.message and chunk.delta.message.content:
                if text := self._extract_content(chunk.delta.message.content):
                    response += text
                    text_messages.append(self.create_text_message(text))
            if chunk.delta.usage:
                self.increase_usage(self._llm_usage, chunk.delta.usage)
                last_usage = chunk.delta.usage

        return text_messages, response, tool_calls, last_usage

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _handle_tool_round(
        self,
        tool_calls: list[ToolCall],
        response: str,
        round_log: AgentInvokeMessage,
    ) -> Generator[AgentInvokeMessage, None, None]:
        if msg := self._build_assistant_message(response, tool_calls):
            self._thoughts.append(msg)

        tool_responses = yield from self._invoke_tools(tool_calls, round_log)
        self._thoughts.extend(self._build_tool_messages(tool_calls, tool_responses))

    def _invoke_tools(
        self,
        tool_calls: list[ToolCall],
        round_log: AgentInvokeMessage,
    ) -> Generator[AgentInvokeMessage, None, list[dict[str, Any]]]:
        tool_responses: list[dict[str, Any]] = []

        for tool_call_id, tool_call_name, tool_call_args in tool_calls:
            tool_instance = self._tool_instances.get(tool_call_name)
            tool_call_started_at = time.perf_counter()
            tool_provider = tool_instance.identity.provider if tool_instance else ""
            tool_call_log = self.create_log_message(
                label=f"CALL {tool_call_name}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: perf_to_wall(tool_call_started_at),
                    LogMetadata.PROVIDER: tool_provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield tool_call_log

            if not tool_instance:
                tool_response = {
                    "tool_call_id": tool_call_id,
                    "tool_call_name": tool_call_name,
                    "tool_response": f"there is not a tool named {tool_call_name}",
                    "meta": ToolInvokeMeta.error_instance(
                        f"there is not a tool named {tool_call_name}"
                    ).to_dict(),
                }
            else:
                try:
                    tool_result, additional_messages = process_tool_invoke_responses(
                        self.session.tool.invoke(
                            provider_type=ToolProviderType(tool_instance.provider_type),
                            provider=tool_instance.identity.provider,
                            tool_name=tool_instance.identity.name,
                            parameters={
                                **tool_instance.runtime_parameters,
                                **tool_call_args,
                            },
                        ),
                        self,
                    )
                    yield from additional_messages
                except Exception as e:
                    tool_result = f"tool invoke error: {e!s}"
                tool_response = {
                    "tool_call_id": tool_call_id,
                    "tool_call_name": tool_call_name,
                    "tool_call_input": {
                        **tool_instance.runtime_parameters,
                        **tool_call_args,
                    },
                    "tool_response": tool_result,
                }

            yield self.finish_log_message(
                log=tool_call_log,
                data=tool_response,
                metadata=finish_log_metadata(
                    tool_call_started_at, provider=tool_provider
                ),
            )
            tool_responses.append(tool_response)

        return tool_responses

    # ------------------------------------------------------------------
    # Summary call
    # ------------------------------------------------------------------

    def _run_summary_call(self) -> Generator[AgentInvokeMessage, None, None]:
        summary_started_at = time.perf_counter()
        summary_log = self.create_log_message(
            label="FINAL SUMMARY",
            data={},
            metadata={
                LogMetadata.STARTED_AT: perf_to_wall(summary_started_at),
                LogMetadata.PROVIDER: self._model.provider,
            },
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield summary_log

        prompt_messages = self._organize_prompt_messages()
        if self._model.entity and self._model.completion_params:
            self.recalc_llm_max_tokens(
                self._model.entity,
                prompt_messages,
                self._model.completion_params,
            )

        text_messages, response, _, summary_usage = self._call_llm(
            prompt_messages,
            tools=[],
        )
        yield from text_messages

        yield self.finish_log_message(
            log=summary_log,
            data={"response": response},
            metadata=finish_log_metadata(
                summary_started_at,
                provider=self._model.provider,
                usage=summary_usage,
            ),
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        prompt_messages = [*self._history_messages, *self._thoughts]

        supports_vision = (
            ModelFeature.VISION in self._model.entity.features
            if self._model.entity and self._model.entity.features
            else False
        )

        if not supports_vision or self._thoughts:
            prompt_messages = self._clear_user_prompt_image_messages(prompt_messages)

        return prompt_messages

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    _THINK_PATTERN = re.compile(r"<think[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)

    @classmethod
    def _extract_content(cls, content: str | list[Any] | None) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            text = "".join(c.data for c in content)
        else:
            text = str(content)
        return cls._THINK_PATTERN.sub("", text)

    @staticmethod
    def _build_assistant_message(
        response: str,
        tool_calls: list[ToolCall],
    ) -> AssistantPromptMessage | None:
        if tool_calls:
            return AssistantPromptMessage(
                content=response,
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id=tc_id,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=tc_name,
                            arguments=json.dumps(tc_args, ensure_ascii=False),
                        ),
                    )
                    for tc_id, tc_name, tc_args in tool_calls
                ],
            )
        if response.strip():
            return AssistantPromptMessage(content=response, tool_calls=[])
        return None

    @staticmethod
    def _build_tool_messages(
        tool_calls: list[ToolCall],
        tool_responses: list[dict[str, Any]],
    ) -> list[ToolPromptMessage]:
        return [
            ToolPromptMessage(
                content=str(resp["tool_response"]),
                tool_call_id=tc_id,
                name=tc_name,
            )
            for (tc_id, tc_name, _), resp in zip(
                tool_calls, tool_responses, strict=False
            )
            if resp["tool_response"] is not None
        ]

    @staticmethod
    def _clear_user_prompt_image_messages(
        prompt_messages: list[PromptMessage],
    ) -> list[PromptMessage]:
        prompt_messages = deepcopy(prompt_messages)
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, UserPromptMessage) and isinstance(
                prompt_message.content, list
            ):
                prompt_message.content = "\n".join(
                    content.data
                    if content.type == PromptMessageContentType.TEXT
                    else "[image]"
                    if content.type == PromptMessageContentType.IMAGE
                    else "[file]"
                    for content in prompt_message.content
                )
        return prompt_messages
