# Agent Strategies Plugin for Dify

Function Calling and ReAct agent reasoning strategies for Dify.

## Strategies

### Function Calling

Maps user intent to structured tool calls. The LLM identifies which tool to call and extracts parameters using the model's native function calling API.

### ReAct (Reason + Act)

Alternates between reasoning and action. The LLM analyzes the current state, selects a tool, observes the result, and repeats until the task is complete. Uses text-based `Thought:/Action:/Observation:/FinalAnswer:` format.

## Installation

### From GitHub

In your Dify instance, go to **Plugins > Install from GitHub** and enter this repository URL.

### Manual

1. Clone this repository
2. Package with `dify plugin package ./`
3. Upload the `.difypkg` file via **Plugins > Install from local file**
