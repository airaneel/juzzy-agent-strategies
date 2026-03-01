import json
import time
from collections.abc import Generator
from typing import Any, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
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
from dify_plugin.interfaces.agent import (
    AgentScratchpadUnit,
    AgentStrategy,
)
from output_parser.cot_output_parser import ReactChunk, ReactState, ReactStreamParser
from prompt.template import REACT_PROMPT_TEMPLATE
from strategies._base import (
    AgentParams,
    ExecutionMetadata,
    LogMetadata,
    UsageAccumulator,
    build_retriever_resources,
    build_timing_metadata,
    create_usage_accumulator,
    process_tool_invoke_responses,
)

IGNORE_OBSERVATION_PROVIDERS = frozenset(["wenxin"])


class ReActAgentStrategy(AgentStrategy):
    def _invoke(
        self, parameters: dict[str, Any]
    ) -> Generator[AgentInvokeMessage, None, None]:
        yield from _InvocationRunner(
            self, AgentParams(**parameters)
        ).run()


class _InvocationRunner:
    """Encapsulates state and logic for a single ReAct invocation.

    The strategy instance is used solely for SDK method access (logging, LLM
    calls, tool invocation).  All mutable per-invocation state lives here.
    """

    def __init__(
        self,
        strategy: ReActAgentStrategy,
        params: AgentParams,
    ):
        self._strategy = strategy
        self.model = params.model
        self.model.completion_params = self.model.completion_params or {}
        self.query = params.query
        self.instruction = params.instruction
        self.stop_sequences = (
            params.model.completion_params.get("stop", [])
            if params.model.completion_params
            else []
        )
        if (
            "Observation" not in self.stop_sequences
            and params.model.provider not in IGNORE_OBSERVATION_PROVIDERS
        ):
            self.stop_sequences.append("Observation")
        self.llm_usage: UsageAccumulator = create_usage_accumulator()
        self.history_messages = params.model.history_prompt_messages
        self.tool_instances = (
            {t.identity.name: t for t in params.tools} if params.tools else {}
        )
        self.prompt_tools = strategy._init_prompt_tools(params.tools)
        self.scratchpad: list[AgentScratchpadUnit] = []
        self._context = params.context
        self._max_iterations = params.maximum_iterations
        self._usd_to_rub = params.usd_to_rub

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Generator[AgentInvokeMessage, None, None]:
        final_answer = ""
        answer_streamed = False
        needs_summary = False

        for step in range(1, self._max_iterations + 1):
            round_started_at = time.perf_counter()
            round_log = self._strategy.create_log_message(
                label=f"ROUND {step}",
                data={},
                metadata={LogMetadata.STARTED_AT: round_started_at},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            unit, round_answer, round_usage = (
                yield from self._invoke_and_parse_model(round_log)
            )
            self.scratchpad.append(unit)
            final_answer += round_answer

            if not unit.action:
                final_answer = unit.thought
                answer_streamed = True
                yield self._finish_round_log(round_log, unit, round_started_at, round_usage)
                break

            if unit.action.action_name.lower() == "final answer":
                final_answer = self._extract_final_answer(unit.action)
                yield self._finish_round_log(round_log, unit, round_started_at, round_usage)
                break

            if step == self._max_iterations:
                needs_summary = True
                yield self._strategy.finish_log_message(
                    log=round_log,
                    data={
                        "thought": unit.thought,
                        "note": f"Max iterations ({self._max_iterations}) reached.",
                    },
                    metadata=build_timing_metadata(round_started_at, usage=round_usage, usd_to_rub=self._usd_to_rub),
                )
                break

            yield from self._invoke_tool(unit, round_log)
            for pt in self.prompt_tools:
                self._strategy.update_prompt_message_tool(
                    self.tool_instances[pt.name], pt
                )
            yield self._finish_round_log(round_log, unit, round_started_at, round_usage)

        if needs_summary:
            yield from self._run_summary_call()
        elif not answer_streamed and final_answer:
            yield self._strategy.create_text_message(final_answer)

        yield from self._emit_final_metadata()

    def _finish_round_log(
        self,
        round_log: AgentInvokeMessage,
        unit: AgentScratchpadUnit,
        started_at: float,
        usage: LLMUsage | None,
    ) -> AgentInvokeMessage:
        return self._strategy.finish_log_message(
            log=round_log,
            data={
                "action_name": unit.action.action_name if unit.action else "",
                "action_input": unit.action.action_input if unit.action else "",
                "thought": unit.thought,
                "observation": unit.observation,
            },
            metadata=build_timing_metadata(started_at, usage=usage, usd_to_rub=self._usd_to_rub),
        )

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
        """Invoke LLM, parse ReAct stream.

        Returns (scratchpad_unit, answer_buffer, usage).
        """
        prompt_messages = self._organize_prompt_messages()
        if self.model.entity and self.model.completion_params:
            self._strategy.recalc_llm_max_tokens(
                self.model.entity, prompt_messages, self.model.completion_params
            )

        chunks = self._strategy.session.model.llm.invoke(
            model_config=LLMModelConfig(**self.model.model_dump(mode="json")),
            prompt_messages=prompt_messages,
            stream=True,
            stop=self.stop_sequences,
        )

        usage_dict: UsageAccumulator = create_usage_accumulator()
        react_chunks = ReactStreamParser().parse(
            chunks, usage_dict
        )

        unit = AgentScratchpadUnit(
            agent_response="", thought="", action_str="",
            observation="", action=None,
        )
        answer_buffer = ""

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
                    yield self._strategy.create_text_message(react_chunk.content)
                    unit.agent_response = (unit.agent_response or "") + react_chunk.content
                    unit.thought = (unit.thought or "") + react_chunk.content
                else:
                    yield self._strategy.create_text_message(react_chunk.content)

        unit.thought = (
            unit.thought.strip() if unit.thought
            else "I am thinking about how to help you"
        )

        if usage_dict.get("usage") is not None:
            self._strategy.increase_usage(self.llm_usage, usage_dict["usage"])
        else:
            usage_dict["usage"] = LLMUsage.empty_usage()

        action_dict = (
            unit.action.to_dict() if unit.action
            else {"action": unit.agent_response}
        )
        yield self._strategy.finish_log_message(
            log=model_log,
            data={"thought": unit.thought, **action_dict},
            metadata=build_timing_metadata(
                model_started_at,
                provider=self.model.provider,
                usage=usage_dict["usage"],
                usd_to_rub=self._usd_to_rub,
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
        """Invoke a tool action and update the scratchpad unit."""
        assert unit.action is not None
        tool_name = unit.action.action_name
        tool_instance = self.tool_instances.get(tool_name)
        tool_started_at = time.perf_counter()
        tool_provider = tool_instance.identity.provider if tool_instance else ""

        tool_log = self._strategy.create_log_message(
            label=f"CALL {tool_name}",
            data={},
            metadata={
                LogMetadata.STARTED_AT: tool_started_at,
                LogMetadata.PROVIDER: tool_provider,
            },
            parent=round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield tool_log

        response, parameters, additional = self._execute_tool(unit.action)
        unit.observation = response
        unit.agent_response = response

        yield from additional
        yield self._strategy.finish_log_message(
            log=tool_log,
            data={
                "tool_name": tool_name,
                "tool_call_args": parameters,
                "output": response,
            },
            metadata=build_timing_metadata(tool_started_at, provider=tool_provider, usd_to_rub=self._usd_to_rub),
        )

    def _execute_tool(
        self,
        action: AgentScratchpadUnit.Action,
    ) -> tuple[str, dict[str, Any] | str, list[ToolInvokeMessage]]:
        """Execute a single tool call."""
        tool_instance = self.tool_instances.get(action.action_name)
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

        tool_call_args = cast(dict[str, Any], tool_call_args)
        invoke_params = {**tool_instance.runtime_parameters, **tool_call_args}

        try:
            result, additional = process_tool_invoke_responses(
                self._strategy.session.tool.invoke(
                    provider_type=ToolProviderType(tool_instance.provider_type),
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters=invoke_params,
                ),
                self._strategy,
            )
        except Exception as e:
            result = f"tool invoke error: {e!s}"
            additional = []

        return result, invoke_params, additional

    # ------------------------------------------------------------------
    # Summary call
    # ------------------------------------------------------------------

    def _run_summary_call(self) -> Generator[AgentInvokeMessage, None, None]:
        """Final LLM call when max iterations exceeded."""
        self.scratchpad.append(
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
                self.model.entity, prompt_messages, self.model.completion_params
            )

        summary_stop = [s for s in self.stop_sequences if s != "Observation"]
        chunks = self._strategy.session.model.llm.invoke(
            model_config=LLMModelConfig(**self.model.model_dump(mode="json")),
            prompt_messages=prompt_messages,
            stream=True,
            stop=summary_stop,
        )

        response = ""
        usage_dict: UsageAccumulator = create_usage_accumulator()
        for react_chunk in ReactStreamParser().parse(
            chunks, usage_dict
        ):
            if isinstance(react_chunk, AgentScratchpadUnit.Action):
                continue
            assert isinstance(react_chunk, ReactChunk)
            response += react_chunk.content
            yield self._strategy.create_text_message(react_chunk.content)

        if usage_dict.get("usage") is not None:
            self._strategy.increase_usage(self.llm_usage, usage_dict["usage"])

        summary_usage = usage_dict.get("usage")
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

    def _build_system_prompt(self) -> SystemPromptMessage:
        system_prompt = (
            REACT_PROMPT_TEMPLATE
            .replace("{{instruction}}", self.instruction or "")
            .replace(
                "{{tools}}",
                json.dumps(
                    [t.model_dump(mode="json") for t in self.prompt_tools]
                ),
            )
            .replace(
                "{{tool_names}}",
                ", ".join(t.name for t in self.prompt_tools),
            )
        )
        return SystemPromptMessage(content=system_prompt)

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        system_msg = self._build_system_prompt()
        query_msg = UserPromptMessage(content=self.query)

        if not self.scratchpad:
            return [system_msg, *self.history_messages, query_msg]

        assistant_content = ""
        for unit in self.scratchpad:
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
            *self.history_messages,
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
