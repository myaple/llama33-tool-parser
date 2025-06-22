# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import json
import random
import string
from typing import List, Dict, Any, Optional, Union, Sequence

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ToolCall,
    FunctionCall,
    ExtractedToolCallInformation
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager
)
from vllm.logger import init_logger

logger = init_logger(__name__)

@ToolParserManager.register_module("llama33")
class Llama33ToolParser(ToolParser):
    """
    Robust tool call parser for Llama3.3 models with multiple fallback methods
    """
    
    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.tool_use_start = "<|tool_call|>"
        self.tool_use_end = "<|tool_call_end|>"

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Parses tool calls from model output with robust extraction and validation
        Handles:
        - JSON wrapped in markdown code blocks (```json)
        - Tool calls with interspersed text
        - Invalid JSON recovery
        """
        # Special token extraction
        if self.tool_use_start in model_output and self.tool_use_end in model_output:
            start_idx = model_output.index(self.tool_use_start) + len(self.tool_use_start)
            end_idx = model_output.index(self.tool_use_end)
            json_str = model_output[start_idx:end_idx].strip()
            tool_calls = self._parse_json_tool_call(json_str)
            content = model_output[:model_output.index(self.tool_use_start)].strip()
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content or None
            )
        
        # Markdown code block extraction
        json_blocks = self._extract_json_blocks(model_output)
        if json_blocks:
            # Extract content before the markdown block
            content = model_output.split('```json', 1)[0].strip()
            tool_calls = self._parse_json_tool_call(json_blocks[0])
            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content or None
                )
        
        # Fallback: JSON pattern search
        json_match = self._find_json_in_text(model_output)
        if json_match:
            # Extract content before the JSON object
            content = model_output.split(json_match, 1)[0].strip()
            tool_calls = self._parse_json_tool_call(json_match)
            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content or None
                )
        
        # No tool calls found
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output
        )

    def _extract_json_blocks(self, text: str) -> List[str]:
        """Extracts content from markdown code blocks"""
        pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    def _find_json_in_text(self, text: str) -> Optional[str]:
        """Finds first valid JSON object in text"""
        # First try parsing the entire text as JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return text
        except json.JSONDecodeError:
            pass
        
        # Then look for JSON-like patterns
        # Use a more robust regex to capture entire JSON objects
        json_candidates = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}]|\{[^{}]*\})*\}))*\}', text)
        
        for candidate in json_candidates:
            # Try parsing without cleaning first
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    return candidate
            except json.JSONDecodeError:
                # If fails, try cleaning common issues
                cleaned = candidate.strip()
                cleaned = re.sub(r',\s*(?=[}\]])', '', cleaned)  # Trailing commas
                cleaned = re.sub(r'//.*?\n', '', cleaned)  # Comments
                
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        return cleaned
                except json.JSONDecodeError:
                    continue
        return None

    def _parse_json_tool_call(self, json_str: str) -> List[ToolCall]:
        """Parses and validates a JSON tool call"""
        try:
            # Handle common formatting issues
            json_str = json_str.strip()
            json_str = re.sub(r'^\s*```(?:json)?', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'```\s*$', '', json_str, flags=re.IGNORECASE)
            
            # Handle single-quoted JSON by converting to double quotes
            if "'" in json_str:
                json_str = re.sub(r"'(.*?)':", '"\1":', json_str)  # Keys
                json_str = re.sub(r": '(.*?)'", ': "\1"', json_str)  # Values
                json_str = re.sub(r"\['(.*?)'\]", '["\1"]', json_str)  # Array values
                
            # Remove comments
            json_str = re.sub(r'//.*?\n', '', json_str)
            
            # Fix trailing commas
            json_str = re.sub(r',\s*(?=[}\])])', '', json_str)
            
            tool_call = json.loads(json_str)
            
            # Validate structure
            if not isinstance(tool_call, dict):
                return []
            if "name" not in tool_call or "arguments" not in tool_call:
                return []
                
            # Convert to VLLM format
            arguments = tool_call["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
                
            return [ToolCall(
                id=f"call_{random_tool_call_id()}",
                type="function",
                function=FunctionCall(
                    name=tool_call["name"],
                    arguments=arguments
                )
            )]
        except (json.JSONDecodeError, TypeError):
            return []

    # Placeholder for streaming support - not implemented yet
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> None:
        """
        Placeholder for streaming tool call parsing
        (To be implemented for full streaming support)
        """
        return None