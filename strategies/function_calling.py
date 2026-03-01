import json
import time
from collections.abc import Generator
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
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import (
    AgentStrategy,
    ToolInvokeMeta,
)
from strategies._base import (
    AgentParams,
    ExecutionMetadata,
    LogMetadata,
    UsageAccumulator,
    build_retriever_resources,
    build_timing_metadata,
    consume_generator,
    create_usage_accumulator,
    extract_tool_calls_from_message,
    process_tool_invoke_responses,
)


class FunctionCallingAgentStrategy(AgentStrategy):
    def _invoke(
        self, parameters: dict[str, Any]
    ) -> Generator[AgentInvokeMessage, None, None]:
        yield from _InvocationRunner(
            self, AgentParams(**parameters)
        ).run()


class _InvocationRunner:
    """Encapsulates state and logic for a single function-calling invocation.

    The strategy instance is used solely for SDK method access (logging, LLM
    calls, tool invocation).  All mutable per-invocation state lives here.
    """

    def __init__(
        self,
        strategy: FunctionCallingAgentStrategy,
        params: AgentParams,
    ):
        self._strategy = strategy
        self.model = params.model
        self.can_stream = (
            ModelFeature.STREAM_TOOL_CALL in params.model.entity.features
            if params.model.entity and params.model.entity.features
            else False
        )
        self.stop_sequences = (
            params.model.completion_params.get("stop", [])
            if params.model.completion_params
            else []
        )
        self.llm_usage: UsageAccumulator = create_usage_accumulator()
        self.history_messages = params.model.history_prompt_messages
        self.history_messages.insert(0, SystemPromptMessage(content=params.instruction))
        self.history_messages.append(UserPromptMessage(content=params.query))
        self.thoughts: list[PromptMessage] = []
        self.tool_instances = (
            {t.identity.name: t for t in params.tools} if params.tools else {}
        )
        self.prompt_tools = strategy._init_prompt_tools(params.tools)
        self._context = params.context
        self._max_iterations = params.maximum_iterations
        self._usd_to_rub = params.usd_to_rub

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Generator[AgentInvokeMessage, None, None]:
        for step in range(1, self._max_iterations + 1):
            round_started_at = time.perf_counter()
            round_log = self._strategy.create_log_message(
                label=f"ROUND {step}",
                data={},
                metadata={LogMetadata.STARTED_AT: round_started_at},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            text_messages, response, tool_calls, usage = (
                yield from self._invoke_model(round_log)
            )

            if not tool_calls:
                yield from text_messages
                yield self._strategy.finish_log_message(
                    log=round_log,
                    data={"output": {"llm_response": response, "tool_responses": []}},
                    metadata=build_timing_metadata(round_started_at, usage=usage, usd_to_rub=self._usd_to_rub),
                )
                break

            tool_responses = yield from self._handle_tool_round(
                tool_calls, response, round_log,
            )

            yield self._strategy.finish_log_message(
                log=round_log,
                data={"output": {"llm_response": response, "tool_responses": tool_responses}},
                metadata=build_timing_metadata(round_started_at, usage=usage, usd_to_rub=self._usd_to_rub),
            )

            if step == self._max_iterations:
                yield from text_messages
                self.thoughts.append(
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

            for prompt_tool in self.prompt_tools:
                self._strategy.update_prompt_message_tool(
                    self.tool_instances[prompt_tool.name], prompt_tool
                )

        yield from self._emit_final_metadata()

    # ------------------------------------------------------------------
    # Model invocation
    # ------------------------------------------------------------------

    def _invoke_model(
        self,
        round_log: AgentInvokeMessage,
    ) -> Generator[
        AgentInvokeMessage,
        None,
        tuple[list[AgentInvokeMessage], str, list[tuple[str, str, dict[str, Any]]], LLMUsage | None],
    ]:
        """Organize prompts, invoke LLM, and log the model call."""
        prompt_messages = self._organize_prompt_messages()
        if self.model.entity and self.model.completion_params:
            self._strategy.recalc_llm_max_tokens(
                self.model.entity, prompt_messages,
                self.model.completion_params,
            )

        model_started_at = time.perf_counter()
        model_log = self._strategy.create_log_message(
            label=f"{self.model.model} Thought",
            data={},
            metadata={
                LogMetadata.STARTED_AT: model_started_at,
                LogMetadata.PROVIDER: self.model.provider,
            },
            parent=round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield model_log

        text_messages, (response, tool_calls, usage) = consume_generator(
            self._call_llm(prompt_messages, tools=self.prompt_tools)
        )

        yield self._strategy.finish_log_message(
            log=model_log,
            data={
                "output": response,
                "tool_name": ";".join(tc[1] for tc in tool_calls),
                "tool_input": [
                    {"name": tc[1], "args": tc[2]} for tc in tool_calls
                ],
            },
            metadata=build_timing_metadata(
                model_started_at,
                provider=self.model.provider,
                usage=usage,
                usd_to_rub=self._usd_to_rub,
            ),
        )

        return text_messages, response, tool_calls, usage

    def _call_llm(
        self,
        prompt_messages: list[PromptMessage],
        *,
        tools: list | None = None,
    ) -> Generator[
        AgentInvokeMessage,
        None,
        tuple[str, list[tuple[str, str, dict[str, Any]]], LLMUsage | None],
    ]:
        """Invoke LLM and dispatch to stream/blocking processor."""
        chunks: Generator[LLMResultChunk, None, None] | LLMResult = (
            self._strategy.session.model.llm.invoke(
                model_config=LLMModelConfig(**self.model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stop=self.stop_sequences,
                stream=self.can_stream,
                tools=tools if tools is not None else [],
            )
        )

        if isinstance(chunks, Generator):
            return (yield from self._process_streaming_response(chunks))
        return (yield from self._process_blocking_response(cast(LLMResult, chunks)))

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_streaming_response(
        self,
        chunks: Generator[LLMResultChunk, None, None],
    ) -> Generator[
        AgentInvokeMessage,
        None,
        tuple[str, list[tuple[str, str, dict[str, Any]]], LLMUsage | None],
    ]:
        response = ""
        tool_calls: list[tuple[str, str, dict[str, Any]]] = []
        last_usage: LLMUsage | None = None

        for chunk in chunks:
            if chunk.delta.message.tool_calls:
                tool_calls.extend(
                    extract_tool_calls_from_message(chunk.delta.message.tool_calls)
                )
            if chunk.delta.message and chunk.delta.message.content:
                if text := self._extract_content(chunk.delta.message.content):
                    response += text
                    yield self._strategy.create_text_message(text)
            if chunk.delta.usage:
                self._strategy.increase_usage(self.llm_usage, chunk.delta.usage)
                last_usage = chunk.delta.usage

        return response, tool_calls, last_usage

    def _process_blocking_response(
        self,
        result: LLMResult,
    ) -> Generator[
        AgentInvokeMessage,
        None,
        tuple[str, list[tuple[str, str, dict[str, Any]]], LLMUsage | None],
    ]:
        tool_calls: list[tuple[str, str, dict[str, Any]]] = []
        last_usage: LLMUsage | None = None

        if result.message.tool_calls:
            tool_calls.extend(
                extract_tool_calls_from_message(result.message.tool_calls)
            )
        if result.usage:
            self._strategy.increase_usage(self.llm_usage, result.usage)
            last_usage = result.usage

        text = self._extract_content(
            result.message.content if result.message else None
        )
        if text:
            yield self._strategy.create_text_message(text)

        return text, tool_calls, last_usage

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _handle_tool_round(
        self,
        tool_calls: list[tuple[str, str, dict[str, Any]]],
        response: str,
        round_log: AgentInvokeMessage,
    ) -> Generator[AgentInvokeMessage, None, list[dict[str, Any]]]:
        """Build LLM context from response, invoke tools, and update thoughts."""
        if msg := self._build_assistant_message(response, tool_calls):
            self.thoughts.append(msg)

        tool_responses = yield from self._invoke_tools(tool_calls, round_log)
        self.thoughts.extend(
            self._build_tool_messages(tool_calls, tool_responses)
        )

        return tool_responses

    def _invoke_tools(
        self,
        tool_calls: list[tuple[str, str, dict[str, Any]]],
        round_log: AgentInvokeMessage,
    ) -> Generator[AgentInvokeMessage, None, list[dict[str, Any]]]:
        tool_responses: list[dict[str, Any]] = []

        for tool_call_id, tool_call_name, tool_call_args in tool_calls:
            tool_instance = self.tool_instances.get(tool_call_name)
            tool_call_started_at = time.perf_counter()
            tool_provider = (
                tool_instance.identity.provider if tool_instance else ""
            )
            tool_call_log = self._strategy.create_log_message(
                label=f"CALL {tool_call_name}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: tool_call_started_at,
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
                        self._strategy.session.tool.invoke(
                            provider_type=ToolProviderType(tool_instance.provider_type),
                            provider=tool_instance.identity.provider,
                            tool_name=tool_instance.identity.name,
                            parameters={**tool_instance.runtime_parameters, **tool_call_args},
                        ),
                        self._strategy,
                    )
                    yield from additional_messages
                except Exception as e:
                    tool_result = f"tool invoke error: {e!s}"
                tool_response = {
                    "tool_call_id": tool_call_id,
                    "tool_call_name": tool_call_name,
                    "tool_call_input": {**tool_instance.runtime_parameters, **tool_call_args},
                    "tool_response": tool_result,
                }

            yield self._strategy.finish_log_message(
                log=tool_call_log,
                data={"output": tool_response},
                metadata=build_timing_metadata(
                    tool_call_started_at, provider=tool_provider, usd_to_rub=self._usd_to_rub
                ),
            )
            tool_responses.append(tool_response)

        return tool_responses

    # ------------------------------------------------------------------
    # Summary call
    # ------------------------------------------------------------------

    def _run_summary_call(self) -> Generator[AgentInvokeMessage, None, None]:
        """Final LLM call when max iterations exceeded."""
        summary_started_at = time.perf_counter()
        summary_round_log = self._strategy.create_log_message(
            label="FINAL SUMMARY",
            data={},
            metadata={LogMetadata.STARTED_AT: summary_started_at},
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield summary_round_log

        summary_model_log = self._strategy.create_log_message(
            label=f"{self.model.model} Summary",
            data={},
            metadata={
                LogMetadata.STARTED_AT: summary_started_at,
                LogMetadata.PROVIDER: self.model.provider,
            },
            parent=summary_round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield summary_model_log

        prompt_messages = self._organize_prompt_messages()
        if self.model.entity and self.model.completion_params:
            self._strategy.recalc_llm_max_tokens(
                self.model.entity, prompt_messages,
                self.model.completion_params,
            )

        response, _, summary_usage = yield from self._call_llm(
            prompt_messages, tools=[],
        )

        yield self._strategy.finish_log_message(
            log=summary_model_log,
            data={"output": response},
            metadata=build_timing_metadata(
                summary_started_at,
                provider=self.model.provider,
                usage=summary_usage,
                usd_to_rub=self._usd_to_rub,
            ),
        )
        yield self._strategy.finish_log_message(
            log=summary_round_log,
            data={"output": {"llm_response": response}},
            metadata=build_timing_metadata(
                summary_started_at, usage=summary_usage, usd_to_rub=self._usd_to_rub
            ),
        )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _emit_final_metadata(self) -> Generator[AgentInvokeMessage, None, None]:
        if isinstance(self._context, list):
            yield self._strategy.create_retriever_resource_message(
                retriever_resources=build_retriever_resources(self._context),
                context="",
            )
        yield self._strategy.create_json_message({
            "execution_metadata": ExecutionMetadata.from_llm_usage(
                self.llm_usage["usage"], usd_to_rub=self._usd_to_rub
            ).model_dump()
        })

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        prompt_messages = [*self.history_messages, *self.thoughts]

        supports_vision = (
            ModelFeature.VISION in self.model.entity.features
            if self.model.entity and self.model.entity.features
            else False
        )

        if not supports_vision or self.thoughts:
            prompt_messages = self._clear_user_prompt_image_messages(prompt_messages)

        return prompt_messages

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(c.data for c in content)
        return str(content)

    @staticmethod
    def _build_assistant_message(
        response: str,
        tool_calls: list[tuple[str, str, dict[str, Any]]],
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
        tool_calls: list[tuple[str, str, dict[str, Any]]],
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
