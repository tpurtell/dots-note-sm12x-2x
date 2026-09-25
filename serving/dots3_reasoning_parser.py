# SPDX-License-Identifier: Apache-2.0
"""Dots3 thinking parser respecting its bare assistant prompt and request mode.

The checkpoint starts thinking generations with <|assistant|>, not a prefilled
<think>. Earlier assistant turns may contain completed thinking blocks. Their
end markers must not close reasoning in the newly generated assistant turn.
"""
from collections.abc import Iterable, Sequence

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class Dots3ReasoningParser(DeepSeekR1ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Match the checkpoint Jinja condition exactly: only literal false
        # disables thinking; omission defaults to enabled.
        template_kwargs = kwargs.get('chat_template_kwargs') or {}
        self.thinking_enabled = template_kwargs.get('enable_thinking', True) is not False
        self.assistant_token_id = self.vocab.get('<|assistant|>')
        if self.assistant_token_id is None:
            raise ValueError('Dots3 reasoning requires the <|assistant|> tokenizer token')

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self.thinking_enabled:
            return True
        # Called both on the full rendered prompt and on generated token IDs.
        # The last assistant delimiter bounds the current turn in a prompt.
        for index in range(len(input_ids) - 1, -1, -1):
            token = input_ids[index]
            if token == self.assistant_token_id:
                return False
            if token == self.start_token_id:
                return False
            if token == self.end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        if not self.thinking_enabled:
            return True
        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(self, model_output, request):
        request_kwargs = getattr(request, 'chat_template_kwargs', None) or {}
        thinking = request_kwargs.get('enable_thinking', self.thinking_enabled) is not False
        if not thinking:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text,
                                    previous_token_ids, current_token_ids, delta_token_ids):
        if not self.thinking_enabled:
            return DeltaMessage(content=delta_text)
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids)
        # Dots emits its own opener, including potentially in an MTP batch with
        # reasoning text. The inherited parser exposes that opener in this case.
        if (result is not None and result.reasoning is not None
                and self.start_token_id in delta_token_ids
                and self.start_token_id not in previous_token_ids
                and self.end_token_id not in delta_token_ids):
            result.reasoning = delta_text.partition(self.start_token)[2]
        return result

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return super().count_reasoning_tokens(token_ids) if self.thinking_enabled else 0
