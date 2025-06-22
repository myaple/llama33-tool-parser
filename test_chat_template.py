import pytest
from jinja2 import Template


# Load the chat template once for all tests
@pytest.fixture(scope="module")
def chat_template():
    with open("example-chat-template.txt") as f:
        return Template(f.read())


# Test cases
def test_simple_user_message(chat_template):
    """Test that simple user messages get 'detailed thinking on' prefix"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Should contain the prefix before user message
    assert "detailed thinking on What is the capital of France?" in result

    # Extract user message section
    user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "detailed thinking on What is the capital of France?" in user_section


def test_multi_content_user_message(chat_template):
    """Test that multi-content user messages get prefix on text content"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"},
                    {
                        "type": "image",
                        "image_url": {"url": "https://example.com/france.png"},
                    },
                ],
            }
        ],
    )

    # Should contain prefix before text content
    assert "detailed thinking on What is the capital of France?" in result

    # Extract user message section
    user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "detailed thinking on What is the capital of France?" in user_section


def test_system_message_first(chat_template):
    """Test that system messages are preserved and user messages get prefix"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {"role": "system", "content": "You are a geography expert"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )

    # System message should be preserved
    assert "You are a geography expert" in result

    # Extract system message section
    system_section = result.split("<|start_header_id|>system<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]
    assert "You are a geography expert" in system_section

    # Extract user message section
    user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "detailed thinking on What is the capital of France?" in user_section


def test_assistant_message_no_prefix(chat_template):
    """Test that assistant messages don't get the prefix"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    )

    # Extract user message section
    user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "detailed thinking on What is the capital of France?" in user_section

    # Extract assistant message section
    assistant_section = result.split("<|start_header_id|>assistant<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]
    assert "The capital of France is Paris." in assistant_section
    assert "detailed thinking on" not in assistant_section


def test_non_user_messages_no_prefix(chat_template):
    """Test that system/assistant/tool messages don't get prefix"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "tool", "content": "Tool message"},
            {"role": "user", "content": "User message"},
        ],
    )

    # Extract system message section
    system_section = result.split("<|start_header_id|>system<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]
    assert "System message" in system_section
    assert "detailed thinking on" not in system_section

    # Extract assistant message section
    assistant_section = result.split("<|start_header_id|>assistant<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]
    assert "Assistant message" in assistant_section
    assert "detailed thinking on" not in assistant_section

    # Extract tool message section
    tool_section = result.split("<|start_header_id|>ipython<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "Tool message" in tool_section
    assert "detailed thinking on" not in tool_section

    # Extract user message section
    user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "detailed thinking on User message" in user_section


# Tool-related tests
def test_single_tool_call(chat_template):
    """Test formatting of assistant tool calls"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris", "unit": "celsius"}',
                        },
                    }
                ],
            }
        ],
    )

    # Extract assistant tool call section
    tool_call_section = result.split("<|start_header_id|>assistant<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]

    # Should contain proper JSON structure
    assert (
        '{"name": "get_weather", "parameters": {"location": "Paris", "unit": "celsius"}}'
        in tool_call_section
    )
    assert "detailed thinking on" not in tool_call_section


def test_tool_response(chat_template):
    """Test formatting of tool responses"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": "Sunny, 22°C",
            }
        ],
    )

    # Extract tool response section
    tool_response_section = result.split("<|start_header_id|>ipython<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]

    # Should contain proper JSON output
    # Degree symbol might be escaped as \u00b0 in JSON
    assert "Sunny, 22" in tool_response_section and "C" in tool_response_section
    assert "output" in tool_response_section
    assert "detailed thinking on" not in tool_response_section


def test_multi_tool_response(chat_template):
    """Test formatting of tool responses with multi-content"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": [
                    {"type": "text", "text": "Sunny, 22°C"},
                    {
                        "type": "image",
                        "image_url": {"url": "https://example.com/weather.png"},
                    },
                ],
            }
        ],
    )

    # Extract tool response section
    tool_response_section = result.split("<|start_header_id|>ipython<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]

    # Should contain only text content in JSON output
    # Degree symbol might be escaped as \u00b0 in JSON
    assert "Sunny, 22" in tool_response_section and "C" in tool_response_section
    assert "image" not in tool_response_section


def test_full_tool_conversation(chat_template):
    """Test full conversation with tool use"""
    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": "Sunny, 22°C",
            },
            {"role": "assistant", "content": "The weather in Paris is sunny and 22°C."},
        ],
    )

    # Verify user message
    user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "detailed thinking on What is the weather in Paris?" in user_section

    # Verify tool call
    tool_call_section = result.split("<|start_header_id|>assistant<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]
    assert (
        '{"name": "get_weather", "parameters": {"location": "Paris"}}'
        in tool_call_section
    )

    # Verify tool response
    tool_response_section = result.split("<|start_header_id|>ipython<|end_header_id|>")[
        1
    ].split("<|eot_id|>")[0]
    # Degree symbol might be escaped as \u00b0 in JSON
    assert "Sunny, 22" in tool_response_section and "C" in tool_response_section

    # Verify final assistant response
    final_assistant_section = result.split(
        "<|start_header_id|>assistant<|end_header_id|>"
    )[2].split("<|eot_id|>")[0]
    assert "The weather in Paris is sunny and 22°C." in final_assistant_section
    assert "detailed thinking on" not in final_assistant_section


def test_tool_in_user_message(chat_template):
    """Test template when tools are provided in user message"""
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"],
            },
        }
    ]

    result = chat_template.render(
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
        tools=tools,
        tools_in_user_message=True,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    )

    # Verify tools section
    tools_section = result.split("<|start_header_id|>user<|end_header_id|>")[1].split(
        "<|eot_id|>"
    )[0]
    assert "Given the following functions" in tools_section
    assert "get_weather" in tools_section

    # Verify user message has prefix
    assert "detailed thinking on What is the weather in Paris?" in tools_section
