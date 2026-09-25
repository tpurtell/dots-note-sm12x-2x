#!/usr/bin/env python3
"""CPU-only parser/template/xgrammar contracts against the cached Dots tokenizer."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tool_parsers.dots_tool_parser import DotsToolParser

if 'Dots3ReasoningParser' not in globals():
    try:
        from vllm.reasoning.dots3_reasoning_parser import Dots3ReasoningParser
    except ModuleNotFoundError:
        from dots3_reasoning_parser import Dots3ReasoningParser


class Combined(DelegatingParser):
    reasoning_parser_cls = Dots3ReasoningParser
    tool_parser_cls = DotsToolParser


TOOL = {'type': 'function', 'function': {'name': 'add', 'description': 'Add integers',
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                       'required': ['a', 'b'], 'additionalProperties': False}}}
XML = '<dots_function_call>\n<invoke name="add">\n<parameter name="a">2</parameter>\n<parameter name="b">3</parameter>\n</invoke>\n</dots_function_call>'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model', default='/root/.cache/huggingface/hub/models--wrldsuksgo2mars--dots3-note-prev-exl3-k4-v1/snapshots/d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da')
    args = cli.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert DotsToolParser.supports_required_and_named, 'Use the recipe\'s existing required/named tool fix'
    tokens = {text: tokenizer.encode(text, add_special_tokens=False) for text in ['<think>', '</think>', '<|assistant|>', '<|endofassistant|>']}
    assert all(len(ids) == 1 for ids in tokens.values())
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    histories = {
        'first': [{'role': 'user', 'content': 'Answer the question.'}],
        'prior_thinking': [{'role': 'user', 'content': 'Earlier question'},
            {'role': 'assistant', 'content': '<think>Previous reasoning</think>Previous answer'},
            {'role': 'user', 'content': 'Next question'}],
        'tool_return': [{'role': 'user', 'content': 'Add 2 and 3'},
            {'role': 'assistant', 'content': '<think>Previous reasoning</think>' + XML},
            {'role': 'tool', 'tool_call_id': 'prior_call', 'content': '5'}],
    }
    baseline = DeepSeekR1ReasoningParser(tokenizer)
    baseline_failures = []
    contracts = []
    schemas = []
    for thinking in (False, True):
        for history_name, messages in histories.items():
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            prompt = encode(prompt_text)
            parser = Dots3ReasoningParser(tokenizer, chat_template_kwargs={'enable_thinking': thinking})
            assert parser.is_reasoning_end(prompt) is (not thinking), (thinking, history_name)
            if baseline.is_reasoning_end(prompt) != (not thinking):
                baseline_failures.append({'case': history_name, 'thinking': thinking, 'baseline_prompt_reasoning_end': baseline.is_reasoning_end(prompt)})
            for mode in ('none', 'json', 'auto', 'named', 'required'):
                choice = {'type': 'function', 'function': {'name': 'add'}} if mode == 'named' else mode
                with_tools = mode in ('auto', 'named', 'required')
                req = ChatCompletionRequest(model='dots3-note-exl3-k4', messages=messages,
                    chat_template_kwargs={'enable_thinking': thinking},
                    tools=[TOOL] if with_tools else None, tool_choice=choice if with_tools else 'none',
                    response_format={'type': 'json_object'} if mode == 'json' else None)
                body = {'none': '42', 'json': '{"answer":42}', 'auto': XML,
                        'named': '{"a":2,"b":3}',
                        'required': '[{"name":"add","parameters":{"a":2,"b":3}}]'}[mode]
                text = ('<think>Let me check.</think>' if thinking else '') + body
                expected_reasoning = 'Let me check.' if thinking else ''
                nonstream = Combined(tokenizer, tools=req.tools, chat_template_kwargs={'enable_thinking': thinking})
                reason, content, tools = nonstream.parse(text, req, enable_auto_tools=with_tools)
                assert (reason or '') == expected_reasoning, (thinking, history_name, mode, reason)
                if with_tools:
                    assert len(tools) == 1 and tools[0].name == 'add', (mode, tools)
                    assert json.loads(tools[0].arguments) == {'a': 2, 'b': 3}
                else:
                    assert content == body and not tools, (mode, content, tools)
                ids = encode(text)
                for chunk_size in (1, 2, 5, len(ids)):
                    stream = Combined(tokenizer, tools=req.tools, chat_template_kwargs={'enable_thinking': thinking})
                    reasoning = content = previous = ''
                    calls = {}
                    for offset in range(0, len(ids), chunk_size):
                        end = min(offset + chunk_size, len(ids))
                        current = tokenizer.decode(ids[:end], skip_special_tokens=False)
                        delta = stream.parse_delta(current[len(previous):], ids[offset:end], req,
                            prompt_token_ids=prompt, finished=end == len(ids))
                        previous = current
                        if delta is None:
                            continue
                        reasoning += delta.reasoning or ''
                        content += delta.content or ''
                        for call in delta.tool_calls or []:
                            entry = calls.setdefault(call.index, {'name': '', 'arguments': ''})
                            if call.function:
                                entry['name'] += call.function.name or ''
                                entry['arguments'] += call.function.arguments or ''
                    assert reasoning == expected_reasoning, (thinking, history_name, mode, chunk_size, reasoning)
                    if with_tools:
                        assert len(calls) == 1, (mode, chunk_size, calls, content)
                        call = next(iter(calls.values()))
                        assert call['name'] == 'add' and json.loads(call['arguments']) == {'a': 2, 'b': 3}, call
                        assert not content.strip(), (mode, chunk_size, content)
                    else:
                        assert content == body and not calls, (mode, chunk_size, content, calls)
                    contracts.append({'thinking': thinking, 'history': history_name, 'mode': mode, 'chunk_tokens': chunk_size})
                if mode == 'json':
                    schemas.append((parser, prompt, text, body, thinking))
    false_req = ChatCompletionRequest(model='dots', messages=histories['first'], chat_template_kwargs={'enable_thinking': False})
    assert baseline.extract_reasoning('Hello', false_req) == ('Hello', None)
    baseline_failures.append({'case': 'nonthinking_nonstream', 'baseline': ['Hello', None], 'expected': [None, 'Hello']})

    # Exercise the actual CPU xgrammar tokenizer/compiler/matcher. The parser's
    # gate controls WHEN the grammar starts; no reasoning text is schema-masked.
    import xgrammar as xgr
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(info)
    schema = {'type': 'object', 'properties': {'answer': {'type': 'integer'}},
              'required': ['answer'], 'additionalProperties': False}
    compiled = compiler.compile_json_schema(json.dumps(schema))
    for parser, prompt, text, body, thinking in schemas:
        matcher = xgr.GrammarMatcher(compiled)
        ended = parser.is_reasoning_end(prompt)
        generated = []
        accepted = []
        for token in encode(text):
            generated.append(token)
            if ended:
                assert matcher.accept_token(token), (thinking, tokenizer.decode(accepted + [token]))
                accepted.append(token)
            elif parser.is_reasoning_end_streaming(generated, [token]):
                ended = True
        assert tokenizer.decode(accepted) == body
        assert parser.count_reasoning_tokens(encode(text)) == (len(encode('Let me check.')) if thinking else 0)
    # Exercise vLLM0.30's actual speculative boundary validation and matcher
    # rollback/reset code, not a reimplementation of its gate.
    from types import SimpleNamespace
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager.enable_in_reasoning = False
    manager.reasoner_cls = Dots3ReasoningParser
    manager.tokenizer = tokenizer
    reasoner = Dots3ReasoningParser(tokenizer, chat_template_kwargs={'enable_thinking': True})
    prompt = encode(tokenizer.apply_chat_template(histories['tool_return'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True))
    grammar = XgrammarGrammar(vocab_size=len(tokenizer), ctx=compiled,
                              matcher=xgr.GrammarMatcher(compiled, max_rollback_tokens=128))
    prefix = prompt + encode('<think>Brief check.')
    end_id = tokens['</think>'][0]
    valid_json = encode('{"answer":42}')
    bad_json = encode('not_json')
    for repeat in range(3):
        structured = SimpleNamespace(reasoner=reasoner, reasoning_ended=False, grammar=grammar)
        request = SimpleNamespace(use_structured_output=True, structured_output_request=structured,
                                  prompt_token_ids=prompt, all_token_ids=prefix.copy(), request_id='cpu-boundary')
        assert manager.validate_tokens(request, [end_id] + bad_json) == [end_id]
        draft = [end_id] + valid_json
        assert manager.validate_tokens(request, draft) == draft
        assert manager.validate_tokens(request, draft) == draft  # Validation rolled back.
        assert grammar.num_processed_tokens == 0
        request.all_token_ids.extend(draft)
        assert manager.accept_tokens(request, draft)
        assert structured.reasoning_ended is True
        assert grammar.num_processed_tokens == len(valid_json)
        assert grammar.accept_tokens(request.request_id, [tokenizer.eos_token_id])
        assert grammar.is_terminated()
        grammar.reset()
        assert not grammar.is_terminated() and grammar.num_processed_tokens == 0
    print(json.dumps({'event': 'dots3_reasoning_cpu_qualified', 'tokens': tokens,
        'stream_contracts_passed': len(contracts), 'nonstream_contracts_passed': 30,
        'xgrammar_gating_cases_passed': len(schemas), 'vllm_spec_boundary_rollback_reset_repeats': 3, 'baseline_defects': baseline_failures,
        'limitations': ['Synthetic parser/token tests only; live decoding, tool roundtrips and speculative structured output still require API qualification.']}, indent=2))


if __name__ == '__main__':
    main()
