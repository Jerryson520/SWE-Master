from r2egym.agenthub.leaf import ToolResult
from rllm.agents.leaf_agent import FunctionCallingAgent, parse_tool_response


def test_parse_multiple_calls_and_natural_finish():
    action = parse_tool_response(
        'inspect\n<tool_call>{"name":"Read","arguments":{"file_path":"a.py"}}</tool_call>'
        '<tool_call>{"name":"Glob","arguments":{"pattern":"**/*.py"}}</tool_call>',
        3,
    )
    assert [call.name for call in action.tool_calls] == ["Read", "Glob"]
    assert [call.id for call in action.tool_calls] == ["call_3_0", "call_3_1"]
    assert parse_tool_response("done", 4).is_terminal


def test_agent_pairs_tool_results_by_id():
    agent = FunctionCallingAgent()
    agent.update_from_env("fix it", 0, False, {})
    action = agent.update_from_model(
        '<tool_call>{"name":"Read","arguments":{"file_path":"a.py"}}</tool_call>'
        '<tool_call>{"name":"Glob","arguments":{"pattern":"*.py"}}</tool_call>'
    ).action
    results = [ToolResult(call.id, call.name, f"result-{index}") for index, call in enumerate(action.tool_calls)]
    agent.update_from_env(results, 0, False, {})
    tool_messages = [message for message in agent.chat_completions if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call_0_0", "call_0_1"]

