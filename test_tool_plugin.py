import unittest
from unittest.mock import MagicMock
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from tool_plugin import Llama33ToolParser

class TestLlama33ToolParser(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.parser = Llama33ToolParser(self.mock_tokenizer)
        
        # Minimal request object
        self.request = ChatCompletionRequest(
            model="test-model",
            messages=[],
            temperature=0.7,
            max_tokens=100
        )
    
    def test_valid_json_block(self):
        output = """Here's my response:
```json
{
    "name": "weather_search",
    "arguments": {
        "location": "New York",
        "unit": "celsius"
    }
}
```"""
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "weather_search")
        self.assertIn('"location": "New York"', result.tool_calls[0].function.arguments)
        # Content should be the text before the tool call
        self.assertEqual(result.content, "Here's my response:")
    
    def test_interspersed_text(self):
        output = "First I think we should call the weather_search tool with location Paris. Here's the call: {\"name\": \"weather_search\", \"arguments\": {\"location\": \"Paris\"}} Then we can do something else."
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "weather_search")
        self.assertEqual(result.content, "First I think we should call the weather_search tool with location Paris. Here's the call:")
    
    def test_special_tokens(self):
        output = "Thinking... <|tool_call|>{\"name\": \"special_tool\", \"arguments\": {}}<|tool_call_end|> Done!"
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "special_tool")
        self.assertEqual(result.content, "Thinking...")
    
    def test_invalid_json(self):
        output = """```json
{'name': 'invalid', 'arguments': {missing_quote}}
```"""
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 0)
        self.assertEqual(result.content, output)
    
    def test_no_tool_call(self):
        output = "This is a normal response without any tool calls."
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 0)
        self.assertEqual(result.content, output)
    
    def test_trailing_comma(self):
        output = """{
    \"name\": \"trailing_comma\",
    \"arguments\": {
        \"key\": \"value\"
    }
}"""
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "trailing_comma")
    
    def test_commented_json(self):
        output = """```json\n{\n    // This is a comment\n    \"name\": \"commented_tool\",\n    \"arguments\": {\n        \"param\": true\n    }\n}\n```"""
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "commented_tool")
    
    def test_multiple_calls_returns_first_valid(self):
        output = """Call 1: 
```json
{"name": "first_tool", "arguments": {"param": 1}}
```
Call 2:
{"name": "second_tool", "arguments": {"param": 2}}"""
        result = self.parser.extract_tool_calls(output, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "first_tool")

if __name__ == "__main__":
    unittest.main()