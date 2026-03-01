import json
import time
from collections.abc import Generator
from typing import Any, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMResultChunk, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import (
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)
from dify_plugin.entities.provider_config import LogMetadata
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentScratchpadUnit,
    AgentStrategy,
    ToolEntity,
)
from output_parser.cot_output_parser import ReactChunk, ReactState, ReactStreamParser
from prompt.template import REACT_PROMPT_TEMPLATE
from strategies._base import (
    ContextItem,
    emit_final_metadata,
    finish_log_metadata,
    perf_to_wall,
    process_tool_invoke_responses,
)

IGNORE_OBSERVATION_PROVIDERS = frozenset(["wenxin"])


class ReActAgentStrategy(AgentStrategy):
    def _invoke(
        self, parameters: dict[str, object]
    ) -> Generator[AgentInvokeMessage, None, None]:
        self._model = AgentModelConfig.model_validate(parameters["model"])
        self._llm_config = LLMModelConfig.model_validate(parameters["model"])
        raw_tools = cast(list[dict[str, Any]] | None, parameters.get("tools"))
        tools = [ToolEntity.model_validate(t) for t in raw_tools] if raw_tools else None

        self._model.completion_params = self._model.completion_params or {}

        self._query = cast(str, parameters["query"])
        self._instruction = cast(str | None, parameters.get("instruction"))
        self._stop_sequences = (
            self._model.completion_params.get("stop", [])
            if self._model.completion_params
            else []
        )
        if (
            "Observation" not in self._stop_sequences
            and self._model.provider not in IGNORE_OBSERVATION_PROVIDERS
        ):
            self._stop_sequences.append("Observation")
        self._llm_usage: dict[str, LLMUsage | None] = {"usage": None}
        self._history_messages = self._model.history_prompt_messages
        self._tool_instances = (
            {t.identity.name: t for t in tools} if tools else {}
        )
        self._prompt_tools = self._init_prompt_tools(tools)
        self._scratchpad: list[AgentScratchpadUnit] = []

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
        final_answer = ""
        answer_streamed = False
        needs_summary = False

        for step in range(1, max_iterations + 1):
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {step}",
                data={},
                metadata={LogMetadata.STARTED_AT: perf_to_wall(round_started_at)},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            unit, round_answer, round_usage = (
                yield from self._invoke_and_parse_model(round_log)
            )
            self._scratchpad.append(unit)
            final_answer += round_answer

            if not unit.action:
                final_answer = unit.thought or ""
                answer_streamed = True
                yield self.finish_log_message(
                    log=round_log,
                    data={"output": final_answer},
                    metadata=finish_log_metadata(round_started_at, usage=round_usage),
                )
                break

            if unit.action.action_name.lower() == "final answer":
                final_answer = self._extract_final_answer(unit.action)
                yield self.finish_log_message(
                    log=round_log,
                    data={"output": final_answer},
                    metadata=finish_log_metadata(round_started_at, usage=round_usage),
                )
                break

            if step == max_iterations:
                needs_summary = True
                yield self.finish_log_message(
                    log=round_log,
                    data={"note": f"Max iterations ({max_iterations}) reached."},
                    metadata=finish_log_metadata(round_started_at, usage=round_usage),
                )
                break

            yield from self._invoke_tool(unit, round_log)
            for pt in self._prompt_tools:
                self.update_prompt_message_tool(
                    self._tool_instances[pt.name], pt
                )
            yield self.finish_log_message(
                log=round_log,
                metadata=finish_log_metadata(round_started_at, usage=round_usage),
            )

        if needs_summary:
            yield from self._run_summary_call()
        elif not answer_streamed and final_answer:
            yield self.create_text_message(final_answer)

        yield from emit_final_metadata(self, context, self._llm_usage, usd_to_rub)

    # ------------------------------------------------------------------
    # Model invocation + parsing
    # ------------------------------------------------------------------

    def _invoke_and_parse_model(
        self,
        round_log: AgentInvokeMessage,
    ) -> Generator[
        AgentInvokeMessage,
        None,
        tuple[AgentScratchpadUnit, str, LLMUsage | None],
    ]:
        prompt_messages = self._organize_prompt_messages()
        if self._model.entity and self._model.completion_params:
            self.recalc_llm_max_tokens(
                self._model.entity, prompt_messages, self._model.completion_params
            )

        chunks: Generator[LLMResultChunk, None, None] = self.session.model.llm.invoke(
            model_config=self._llm_config,
            prompt_messages=prompt_messages,
            stream=True,
            stop=self._stop_sequences,
        )

        usage_dict: dict[str, LLMUsage | None] = {"usage": None}
        react_chunks = ReactStreamParser().parse(chunks, usage_dict)

        unit = AgentScratchpadUnit(
            agent_response="", thought="", action_str="",
            observation="", action=None,
        )
        answer_buffer = ""
        llm_started_at = time.perf_counter()

        for react_chunk in react_chunks:
            if isinstance(react_chunk, AgentScratchpadUnit.Action):
                unit.agent_response = (unit.agent_response or "") + json.dumps(
                    react_chunk.model_dump()
                )
                unit.action_str = json.dumps(react_chunk.model_dump())
                unit.action = react_chunk
            else:
                assert isinstance(react_chunk, ReactChunk)
                if react_chunk.state == ReactState.ANSWER:
                    answer_buffer += react_chunk.content
                elif react_chunk.state == ReactState.THINKING:
                    yield self.create_text_message(react_chunk.content)
                    unit.agent_response = (unit.agent_response or "") + react_chunk.content
                    unit.thought = (unit.thought or "") + react_chunk.content
                else:
                    yield self.create_text_message(react_chunk.content)

        unit.thought = (
            unit.thought.strip() if unit.thought
            else "I am thinking about how to help you"
        )

        usage = usage_dict.get("usage")
        if usage is not None:
            self.increase_usage(self._llm_usage, usage)
        else:
            usage = LLMUsage.empty_usage()  # type: ignore[no-untyped-call]
            usage_dict["usage"] = usage

        # Only create a Thought child log when there's an action (tool call).
        # For the final answer the round log carries the output directly,
        # avoiding the SDK duplicating it via parent-child aggregation.
        if unit.action:
            model_log = self.create_log_message(
                label=f"{self._model.model} Thought",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: perf_to_wall(llm_started_at),
                    LogMetadata.PROVIDER: self._model.provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            action_dict = unit.action.to_dict()
            yield self.finish_log_message(
                log=model_log,
                data={"thought": unit.thought, **action_dict},
                metadata=finish_log_metadata(
                    llm_started_at,
                    provider=self._model.provider,
                    usage=usage_dict["usage"],
                ),
            )

        return unit, answer_buffer, usage_dict["usage"]

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_tool(
        self,
        unit: AgentScratchpadUnit,
        round_log: AgentInvokeMessage,
    ) -> Generator[AgentInvokeMessage, None, None]:
        assert unit.action is not None
        tool_name = unit.action.action_name
        tool_instance = self._tool_instances.get(tool_name)
        tool_started_at = time.perf_counter()
        tool_provider = tool_instance.identity.provider if tool_instance else ""

        tool_log = self.create_log_message(
            label=f"CALL {tool_name}",
            data={},
            metadata={LogMetadata.STARTED_AT: perf_to_wall(tool_started_at), LogMetadata.PROVIDER: tool_provider},
            parent=round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield tool_log

        response, parameters, additional = self._execute_tool(unit.action)
        unit.observation = response
        unit.agent_response = response

        yield from additional
        yield self.finish_log_message(
            log=tool_log,
            data={
                "tool_name": tool_name,
                "tool_call_args": parameters,
                "response": response,
            },
            metadata=finish_log_metadata(tool_started_at, provider=tool_provider),
        )

    def _execute_tool(
        self,
        action: AgentScratchpadUnit.Action,
    ) -> tuple[str, dict[str, Any] | str, list[AgentInvokeMessage]]:
        tool_instance = self._tool_instances.get(action.action_name)
        if not tool_instance:
            return (
                f"there is not a tool named {action.action_name}",
                action.action_input,
                [],
            )

        tool_call_args = action.action_input
        if isinstance(tool_call_args, str):
            try:
                tool_call_args = json.loads(tool_call_args)
            except json.JSONDecodeError as e:
                llm_params = [
                    p.name
                    for p in tool_instance.parameters
                    if p.form == ToolParameter.ToolParameterForm.LLM
                ]
                if len(llm_params) > 1:
                    raise ValueError(
                        "tool call args is not a valid json string"
                    ) from e
                tool_call_args = (
                    {llm_params[0]: tool_call_args} if llm_params else {}
                )

        tool_call_args = cast(dict[str, object], tool_call_args)
        invoke_params = {**tool_instance.runtime_parameters, **tool_call_args}

        try:
            result, additional = process_tool_invoke_responses(
                self.session.tool.invoke(
                    provider_type=ToolProviderType(tool_instance.provider_type),
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters=invoke_params,
                ),
                self,
            )
        except Exception as e:
            result = f"tool invoke error: {e!s}"
            additional = cast(list[AgentInvokeMessage], [])

        return result, invoke_params, additional

    # ------------------------------------------------------------------
    # Summary call
    # ------------------------------------------------------------------

    def _run_summary_call(self) -> Generator[AgentInvokeMessage, None, None]:
        self._scratchpad.append(
            AgentScratchpadUnit(
                agent_response="", thought="", action_str="",
                observation=(
                    "You have reached the maximum number of tool calls. "
                    "You cannot call any more tools. Based on all the information "
                    "you have gathered above, provide your best final answer now."
                ),
                action=None,
            )
        )

        summary_started_at = time.perf_counter()
        summary_log = self.create_log_message(
            label="FINAL SUMMARY",
            data={},
            metadata={LogMetadata.STARTED_AT: perf_to_wall(summary_started_at), LogMetadata.PROVIDER: self._model.provider},
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield summary_log

        prompt_messages = self._organize_prompt_messages()
        if self._model.entity and self._model.completion_params:
            self.recalc_llm_max_tokens(
                self._model.entity, prompt_messages, self._model.completion_params
            )

        summary_stop = [s for s in self._stop_sequences if s != "Observation"]
        chunks: Generator[LLMResultChunk, None, None] = self.session.model.llm.invoke(
            model_config=self._llm_config,
            prompt_messages=prompt_messages,
            stream=True,
            stop=summary_stop,
        )

        response = ""
        usage_dict: dict[str, LLMUsage | None] = {"usage": None}
        for react_chunk in ReactStreamParser().parse(chunks, usage_dict):
            if isinstance(react_chunk, AgentScratchpadUnit.Action):
                continue
            assert isinstance(react_chunk, ReactChunk)
            response += react_chunk.content
            yield self.create_text_message(react_chunk.content)

        summary_usage = usage_dict.get("usage")
        if summary_usage is not None:
            self.increase_usage(self._llm_usage, summary_usage)

        yield self.finish_log_message(
            log=summary_log,
            data={"response": response},
            metadata=finish_log_metadata(
                summary_started_at,
                provider=self._model.provider,
                usage=usage_dict.get("usage"),

            ),
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> SystemPromptMessage:
        system_prompt = (
            REACT_PROMPT_TEMPLATE
            .replace("{{instruction}}", self._instruction or "")
            .replace(
                "{{tools}}",
                json.dumps(
                    [t.model_dump(mode="json") for t in self._prompt_tools]
                ),
            )
            .replace(
                "{{tool_names}}",
                ", ".join(t.name for t in self._prompt_tools),
            )
        )
        return SystemPromptMessage(content=system_prompt)

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        system_msg = self._build_system_prompt()
        query_msg = UserPromptMessage(content=self._query)

        if not self._scratchpad:
            return [system_msg, *self._history_messages, query_msg]

        assistant_content = ""
        for unit in self._scratchpad:
            if unit.is_final():
                assistant_content += f"Final Answer: {unit.agent_response}"
            else:
                assistant_content += f"Thought: {unit.thought}\n\n"
                if unit.action_str:
                    assistant_content += f"Action: {unit.action_str}\n\n"
                if unit.observation:
                    assistant_content += f"Observation: {unit.observation}\n\n"

        return [
            system_msg,
            *self._history_messages,
            query_msg,
            AssistantPromptMessage(content=assistant_content),
            UserPromptMessage(content="continue"),
        ]

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_final_answer(action: AgentScratchpadUnit.Action) -> str:
        if isinstance(action.action_input, dict):
            return json.dumps(action.action_input)
        return str(action.action_input)
