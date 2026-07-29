"""Tests for cli_tool_protocol.py's pure functions: the CLI-agnostic half of
the hand-rolled tool-calling bridge shared by claude_cli_llm.py and
copilot_cli_llm.py -- JSON extraction, tool description text, message
flattening. No I/O, no framework backend dependency beyond message classes."""
import json

from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

from bee_bug_hunter.cli_tool_protocol import (
    describe_tools,
    extract_json_object,
    find_balanced_json_objects,
    flatten_messages,
)


class TestFindBalancedJsonObjects:
    def test_single_object(self):
        assert find_balanced_json_objects('{"a": 1}') == ['{"a": 1}']

    def test_multiple_top_level_objects(self):
        text = 'prose {"a": 1} more prose {"b": 2}'
        assert find_balanced_json_objects(text) == ['{"a": 1}', '{"b": 2}']

    def test_nested_objects_return_only_outer_span(self):
        text = '{"outer": {"inner": 1}}'
        assert find_balanced_json_objects(text) == ['{"outer": {"inner": 1}}']

    def test_braces_inside_string_do_not_affect_depth(self):
        # A brace quoted inside a JSON string value must not be treated as a
        # real brace for depth-counting purposes.
        text = '{"msg": "look at this: } weird"}'
        spans = find_balanced_json_objects(text)
        assert spans == ['{"msg": "look at this: } weird"}']

    def test_escaped_quote_inside_string_does_not_end_string_early(self):
        text = r'{"msg": "she said \"hi\" with a } brace"}'
        spans = find_balanced_json_objects(text)
        assert spans == [text]

    def test_no_braces_returns_empty(self):
        assert find_balanced_json_objects("just plain prose") == []

    def test_unbalanced_closing_brace_is_ignored(self):
        text = 'stray } then {"a": 1}'
        assert find_balanced_json_objects(text) == ['{"a": 1}']


class TestExtractJsonObject:
    def test_direct_json(self):
        assert extract_json_object('{"tool": "x", "args": {}}') == {"tool": "x", "args": {}}

    def test_json_in_code_fence(self):
        text = '```json\n{"final_answer": "done"}\n```'
        assert extract_json_object(text) == {"final_answer": "done"}

    def test_malformed_json_in_code_fence_falls_through_to_brace_scan(self):
        # The fenced content itself doesn't parse, but a real protocol-shaped
        # object elsewhere in the text should still be found.
        text = '```json\n{not valid json\n```\nthen {"tool": "x", "args": {}}'
        assert extract_json_object(text) == {"tool": "x", "args": {}}

    def test_json_in_plain_code_fence_no_language_tag(self):
        text = '```\n{"tool": "y", "args": {}}\n```'
        assert extract_json_object(text) == {"tool": "y", "args": {}}

    def test_prose_with_protocol_shaped_object_among_other_json(self):
        # An HTTP error body quoted in the model's prose is also valid JSON,
        # but only the "tool"/"final_answer"-shaped one should be preferred.
        text = 'The API returned {"status": 500, "error": "boom"} so I will call {"tool": "retry", "args": {}}'
        assert extract_json_object(text) == {"tool": "retry", "args": {}}

    def test_falls_back_to_first_parseable_when_nothing_protocol_shaped(self):
        text = 'unrelated json blob: {"status": 500, "error": "boom"}'
        assert extract_json_object(text) == {"status": 500, "error": "boom"}

    def test_malformed_json_repaired_via_parse_broken_json(self):
        # Unescaped newline inside a string breaks strict json.loads on every
        # balanced-brace candidate; parse_broken_json is the last-resort repair.
        text = '{"final_answer": "line one\nline two"}'
        result = extract_json_object(text)
        assert result is not None
        assert result.get("final_answer", "").startswith("line one")

    def test_no_json_at_all_returns_none(self):
        assert extract_json_object("just some plain prose, no braces") is None

    def test_repaired_json_not_protocol_shaped_returns_none(self):
        # Even after repair, an object with neither "tool" nor "final_answer"
        # should not be accepted as a last resort here (unlike the balanced-
        # brace path, which does fall back to first-parseable).
        text = 'garbage {not real json'
        assert extract_json_object(text) is None


class TestDescribeTools:
    def test_empty_tools_list(self):
        assert describe_tools([]) == "(none)"
        assert describe_tools(None) == "(none)"

    def test_lists_name_description_and_schema(self):
        class FakeSchema:
            @staticmethod
            def model_json_schema():
                return {"type": "object", "properties": {"query": {"type": "string"}}}

        class FakeTool:
            name = "run_mysql_query"
            description = "Runs a query"
            input_schema = FakeSchema

        result = describe_tools([FakeTool()])
        assert "run_mysql_query" in result
        assert "Runs a query" in result
        assert '"query"' in result

    def test_schema_generation_failure_falls_back_to_empty_object(self):
        class BrokenSchema:
            @staticmethod
            def model_json_schema():
                raise RuntimeError("schema broke")

        class FakeTool:
            name = "broken_tool"
            description = "desc"
            input_schema = BrokenSchema

        result = describe_tools([FakeTool()])
        assert "broken_tool" in result
        assert "parameters schema: {}" in result

    def test_multiple_tools_each_get_a_line(self):
        class FakeSchema:
            @staticmethod
            def model_json_schema():
                return {}

        class FakeTool:
            def __init__(self, name):
                self.name = name
                self.description = f"{name} desc"
                self.input_schema = FakeSchema

        result = describe_tools([FakeTool("a"), FakeTool("b")])
        assert "- a: a desc" in result
        assert "- b: b desc" in result


class TestFlattenMessages:
    def test_system_messages_go_to_system_prompt(self):
        system, convo = flatten_messages([SystemMessage("be helpful")])
        assert system == "be helpful"
        assert convo == ""

    def test_user_and_assistant_go_to_conversation(self):
        system, convo = flatten_messages([UserMessage("hi"), AssistantMessage("hello")])
        assert system == ""
        assert "USER: hi" in convo
        assert "ASSISTANT: hello" in convo

    def test_tool_call_content_rendered_as_protocol_json(self):
        msg = AssistantMessage([MessageToolCallContent(tool_name="run_query", args="{}", id="1")])
        _system, convo = flatten_messages([msg])
        assert '"tool": "run_query"' in convo

    def test_tool_result_content_rendered_with_label(self):
        msg = ToolMessage([MessageToolResultContent(tool_name="run_query", tool_call_id="1", result="42 rows")])
        _system, convo = flatten_messages([msg])
        assert "Tool 'run_query' result:" in convo
        assert "42 rows" in convo

    def test_multiple_messages_joined_with_blank_line(self):
        _system, convo = flatten_messages([UserMessage("first"), AssistantMessage("second")])
        assert convo == "USER: first\n\nASSISTANT: second"

    def test_empty_message_list_returns_empty_strings(self):
        assert flatten_messages([]) == ("", "")
