#!/bin/bash

# Test 1: Simple user message
echo "Testing simple user message:"
.env/bin/python -c "
from jinja2 import Template

with open('example-chat-template.txt') as f:
    template = Template(f.read())

result = template.render(
    bos_token='<|begin_of_text|>',
    add_generation_prompt=True,
    messages=[
        {'role': 'user', 'content': 'What is the capital of France?'}
    ]
)
print(result)
"

echo -e "\n---\n"

# Test 2: Multi-content message
echo "Testing multi-content message:"
.env/bin/python -c "
from jinja2 import Template

with open('example-chat-template.txt') as f:
    template = Template(f.read())

result = template.render(
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
print(result)
"

echo -e "\n---\n"

# Test 3: System message first
echo "Testing system message first:"
.env/bin/python -c "
from jinja2 import Template

with open('example-chat-template.txt') as f:
    template = Template(f.read())

result = template.render(
    bos_token='<|begin_of_text|>',
    add_generation_prompt=True,
    messages=[
        {'role': 'system', 'content': 'You are a geography expert'},
        {'role': 'user', 'content': 'What is the capital of France?'}
    ]
)
print(result)
"

echo -e "\n---\n"

# Test 4: Assistant message should not have prefix
echo "Testing assistant message (should not have prefix):"
.env/bin/python -c "
from jinja2 import Template

with open('example-chat-template.txt') as f:
    template = Template(f.read())

result = template.render(
    bos_token='<|begin_of_text|>',
    add_generation_prompt=True,
    messages=[
        {'role': 'user', 'content': 'What is the capital of France?'},
        {'role': 'assistant', 'content': 'The capital of France is Paris.'}
    ]
)
print(result)
"