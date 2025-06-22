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
        bos_token='<|begin_of_text|>',
        add_generation_prompt=True,
        messages=[
            {'role': 'user', 'content': 'What is the capital of France?'}
        ]
    )
    
    # Should contain the prefix before user message
    assert 'detailed thinking on What is the capital of France?' in result
    
    # Extract user message section
    user_section = result.split('<|start_header_id|>user<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'detailed thinking on What is the capital of France?' in user_section

def test_multi_content_user_message(chat_template):
    """Test that multi-content user messages get prefix on text content"""
    result = chat_template.render(
        bos_token='<|begin_of_text|>',
        add_generation_prompt=True,
        messages=[
            {'role': 'user', 
             'content': [
                 {'type': 'text', 'text': 'What is the capital of France?'},
                 {'type': 'image', 'image_url': {'url': 'https://example.com/france.png'}}
             ]}
        ]
    )
    
    # Should contain prefix before text content
    assert 'detailed thinking on What is the capital of France?' in result
    
    # Extract user message section
    user_section = result.split('<|start_header_id|>user<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'detailed thinking on What is the capital of France?' in user_section

def test_system_message_first(chat_template):
    """Test that system messages are preserved and user messages get prefix"""
    result = chat_template.render(
        bos_token='<|begin_of_text|>',
        add_generation_prompt=True,
        messages=[
            {'role': 'system', 'content': 'You are a geography expert'},
            {'role': 'user', 'content': 'What is the capital of France?'}
        ]
    )
    
    # System message should be preserved
    assert 'You are a geography expert' in result
    
    # Extract system message section
    system_section = result.split('<|start_header_id|>system<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'You are a geography expert' in system_section
    
    # Extract user message section
    user_section = result.split('<|start_header_id|>user<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'detailed thinking on What is the capital of France?' in user_section

def test_assistant_message_no_prefix(chat_template):
    """Test that assistant messages don't get the prefix"""
    result = chat_template.render(
        bos_token='<|begin_of_text|>',
        add_generation_prompt=True,
        messages=[
            {'role': 'user', 'content': 'What is the capital of France?'},
            {'role': 'assistant', 'content': 'The capital of France is Paris.'}
        ]
    )
    
    # Extract user message section
    user_section = result.split('<|start_header_id|>user<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'detailed thinking on What is the capital of France?' in user_section
    
    # Extract assistant message section
    assistant_section = result.split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'The capital of France is Paris.' in assistant_section
    assert 'detailed thinking on' not in assistant_section

def test_non_user_messages_no_prefix(chat_template):
    """Test that system/assistant/tool messages don't get prefix"""
    result = chat_template.render(
        bos_token='<|begin_of_text|>',
        add_generation_prompt=True,
        messages=[
            {'role': 'system', 'content': 'System message'},
            {'role': 'assistant', 'content': 'Assistant message'},
            {'role': 'tool', 'content': 'Tool message'},
            {'role': 'user', 'content': 'User message'}
        ]
    )
    
    # Extract system message section
    system_section = result.split('<|start_header_id|>system<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'System message' in system_section
    assert 'detailed thinking on' not in system_section
    
    # Extract assistant message section
    assistant_section = result.split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'Assistant message' in assistant_section
    assert 'detailed thinking on' not in assistant_section
    
    # Extract tool message section
    tool_section = result.split('<|start_header_id|>ipython<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'Tool message' in tool_section
    assert 'detailed thinking on' not in tool_section
    
    # Extract user message section
    user_section = result.split('<|start_header_id|>user<|end_header_id|>')[1].split('<|eot_id|>')[0]
    assert 'detailed thinking on User message' in user_section