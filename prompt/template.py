REACT_PROMPT_TEMPLATE = """\
Respond to the human as helpfully and accurately as possible.

{{instruction}}

You have access to the following tools:

{{tools}}

Use a json blob to specify a tool by providing an action key (tool name) \
and an action_input key (tool input).
Valid "action" values: {{tool_names}}

Provide only ONE action per JSON blob, as shown:

{"action": "<tool_name>", "action_input": "<tool_input>"}

Follow this format:

Thought: [your reasoning]
Action: {"action": "tool_name", "action_input": {"param": "value"}}
Observation: [tool output]
... (repeat as needed)
FinalAnswer: [your response]

Begin! Use "Action:" only when calling tools. End with "FinalAnswer:".
"""
