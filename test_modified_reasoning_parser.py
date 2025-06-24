import pytest
from example_reasoning_parser import DeepSeekR1ReasoningParser
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.protocol import DeltaMessage
from typing import Dict, List, Union


class MockTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        super().__init__()
        self.vocab: Dict[str, int] = {"<think>": 1000, "</think>": 1001}
        self.model_tokenizer = self

    def __len__(self) -> int:
        return len(self.vocab)

    def get(self, token_str: str) -> int:
        return self.vocab.get(token_str)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(id) for id in ids]

    def _convert_id_to_token(self, id: int) -> str:
        for token, token_id in self.vocab.items():
            if token_id == id:
                return token
        return "[UNK]"

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab


@pytest.fixture
def parser():
    tokenizer = MockTokenizer()
    return DeepSeekR1ReasoningParser(tokenizer)


def test_non_streaming_parsing(parser):
    """Test non-streaming parsing with various formats."""
    # Test with new format (two newlines after </think>)
    output1 = "<think>Reasoning content</think>\n\nFinal content"
    reasoning, content = parser.extract_reasoning_content(output1, None)
    assert reasoning == "Reasoning content"
    assert content == "Final content"

    # Test with old format (no newlines)
    output2 = "<think>Reasoning content</think>Final content"
    reasoning, content = parser.extract_reasoning_content(output2, None)
    assert reasoning == "Reasoning content"
    assert content == "Final content"

    # Test with one newline
    output3 = "<think>Reasoning content</think>\nFinal content"
    reasoning, content = parser.extract_reasoning_content(output3, None)
    assert reasoning == "Reasoning content"
    assert content == "Final content"

    # Test with extra spaces
    output4 = "<think>Reasoning content</think>  \n\nFinal content"
    reasoning, content = parser.extract_reasoning_content(output4, None)
    assert reasoning == "Reasoning content"
    assert content == "Final content"

    # Test empty content
    output5 = "<think>Reasoning content</think>\n\n"
    reasoning, content = parser.extract_reasoning_content(output5, None)
    assert reasoning == "Reasoning content"
    assert content is None


def test_streaming_parsing(parser):
    """Test streaming parsing with various delta formats."""
    # Case 1: End token with two newlines in the same delta
    delta1 = parser.extract_reasoning_content_streaming(
        previous_text="<think>Reasoning",
        current_text="<think>Reasoning</think>\n\nFinal",
        delta_text="</think>\n\nFinal",
        previous_token_ids=[1000, 200],  # <think> + token for "Reasoning"
        current_token_ids=[
            1000,
            200,
            1001,
            300,
            400,
        ],  # <think>, "Reasoning", </think>, "\n", "\n", "Final"
        delta_token_ids=[1001, 300, 400],  # </think>, "\n", "\n", "Final"
    )
    assert isinstance(delta1, DeltaMessage)
    assert (
        delta1.reasoning_content == ""
    )  # Content after previous_text and before </think>
    assert delta1.content == "Final"  # Content after removing \n\n

    # Case 2: End token with two newlines split across deltas
    delta2_1 = parser.extract_reasoning_content_streaming(
        previous_text="<think>Reasoning",
        current_text="<think>Reasoning</think>",
        delta_text="</think>",
        previous_token_ids=[1000, 200],
        current_token_ids=[1000, 200, 1001],
        delta_token_ids=[1001],
    )
    assert isinstance(delta2_1, DeltaMessage)
    assert delta2_1.reasoning_content == ""
    assert delta2_1.content is None

    delta2_2 = parser.extract_reasoning_content_streaming(
        previous_text="<think>Reasoning</think>",
        current_text="<think>Reasoning</think>\n\nFinal",
        delta_text="\n\nFinal",
        previous_token_ids=[1000, 200, 1001],
        current_token_ids=[1000, 200, 1001, 300, 400],
        delta_token_ids=[300, 400],
    )
    assert isinstance(delta2_2, DeltaMessage)
    assert delta2_2.reasoning_content is None
    assert delta2_2.content == "Final"  # Should remove the two newlines

    # Case 3: No newlines after end token
    delta3 = parser.extract_reasoning_content_streaming(
        previous_text="<think>Reasoning",
        current_text="<think>Reasoning</think>Final",
        delta_text="</think>Final",
        previous_token_ids=[1000, 200],
        current_token_ids=[1000, 200, 1001, 500],
        delta_token_ids=[1001, 500],
    )
    assert isinstance(delta3, DeltaMessage)
    assert delta3.reasoning_content == ""
    assert delta3.content == "Final"

    # Case 4: Partial newlines (only one newline)
    delta4 = parser.extract_reasoning_content_streaming(
        previous_text="<think>Reasoning",
        current_text="<think>Reasoning</think>\nFinal",
        delta_text="</think>\nFinal",
        previous_token_ids=[1000, 200],
        current_token_ids=[1000, 200, 1001, 300, 500],
        delta_token_ids=[1001, 300, 500],
    )
    assert isinstance(delta4, DeltaMessage)
    assert delta4.reasoning_content == ""
    assert delta4.content == "\nFinal"  # Should not remove since only one newline


def test_content_extraction(parser):
    """Test extraction of content IDs."""
    # Test with end token and content
    input_ids = [1000, 200, 1001, 300, 400]  # <think>, content, </think>, content
    content_ids = parser.extract_content_ids(input_ids)
    assert content_ids == [300, 400]

    # Test with end token at the end
    input_ids = [1000, 200, 1001]
    content_ids = parser.extract_content_ids(input_ids)
    assert content_ids == []

    # Test without end token
    input_ids = [1000, 200, 300]
    content_ids = parser.extract_content_ids(input_ids)
    assert content_ids == []


if __name__ == "__main__":
    pytest.main()
