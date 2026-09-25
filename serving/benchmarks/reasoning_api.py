#!/usr/bin/env python3
"""Live reasoning/tool/JSON gate; requests are sent only when main is run.

Each repeat sends forty short deterministic requests covering thinking on/off, stream/nonstream,
first/second assistant turns, JSON, four tool modes and actual tool returns.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

TOOL = {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.',
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                       'required': ['a', 'b'], 'additionalProperties': False}}}


def request(base, payload, timeout):
    endpoint = base.rstrip('/')
    if not endpoint.endswith('/v1'):
        endpoint += '/v1'
    req = urllib.request.Request(endpoint + '/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    result = {'content': '', 'reasoning': '', 'tool_calls': [], 'finish_reason': None,
              'usage': None, 'raw_response': None, 'stream_complete': not payload['stream'],
              'token_chunks': [], 'reasoning_boundary_chunks': []}
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if not payload['stream']:
                body = json.load(response)
                result['raw_response'] = body
                message = body['choices'][0]['message']
                result.update(content=message.get('content') or '',
                    reasoning=message.get('reasoning') or message.get('reasoning_content') or '',
                    tool_calls=message.get('tool_calls') or [],
                    finish_reason=body['choices'][0].get('finish_reason'), usage=body.get('usage'))
            else:
                events, calls = [], {}
                result['raw_response'] = events
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == b'[DONE]':
                        result['stream_complete'] = True
                        break
                    event = json.loads(raw)
                    events.append(event)
                    result['usage'] = event.get('usage') or result['usage']
                    for choice in event.get('choices', []):
                        ids = choice.get('token_ids') or []
                        if ids:
                            result['token_chunks'].append(ids)
                            if 151722 in ids and ids.index(151722) + 1 < len(ids):
                                result['reasoning_boundary_chunks'].append(ids)
                        delta = choice.get('delta', {})
                        result['content'] += delta.get('content') or ''
                        result['reasoning'] += delta.get('reasoning') or delta.get('reasoning_content') or ''
                        result['finish_reason'] = choice.get('finish_reason') or result['finish_reason']
                        for call in delta.get('tool_calls') or []:
                            entry = calls.setdefault(call['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                            entry['id'] = call.get('id') or entry['id']
                            for field in ('name', 'arguments'):
                                entry['function'][field] += (call.get('function') or {}).get(field) or ''
                result['tool_calls'] = [calls[index] for index in sorted(calls)]
    except Exception as exc:
        result['error'] = repr(exc)
        if isinstance(exc, urllib.error.HTTPError):
            result['error_body'] = exc.read().decode(errors='replace')
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def metrics(base):
    endpoint = base.rstrip('/')
    if endpoint.endswith('/v1'):
        endpoint = endpoint[:-3]
    try:
        with urllib.request.urlopen(endpoint + '/metrics', timeout=30) as response:
            lines = response.read().decode().splitlines()
        raw, totals = [], {}
        for line in lines:
            if not line.startswith('vllm:spec_decode_'):
                continue
            name = line.split('{', 1)[0].split()[0]
            if not name.endswith('_total'):
                continue
            raw.append(line)
            totals[name] = totals.get(name, 0) + float(line.rsplit(' ', 1)[1])
        return {'raw_lines': raw, 'totals': totals}
    except Exception as error:
        return {'error': repr(error), 'raw_lines': [], 'totals': {}}


def violations(result, thinking, *, expected=None, expect_tool=False, json_answer=False):
    issues = []
    if result.get('error'):
        issues.append('request_error')
    if not result['stream_complete'] or result['usage'] is None:
        issues.append('incomplete_response')
    if result['finish_reason'] not in (('stop', 'tool_calls') if expect_tool else ('stop',)):
        issues.append('unexpected_finish_reason')
    if any(tag in result['content'] or tag in result['reasoning'] for tag in ('<think>', '</think>')):
        issues.append('thinking_marker_leak')
    # Enabled thinking must exercise the reasoning field in this gate, rather
    # than silently accepting a parser-disabled response or empty reasoning path.
    if thinking and not result['reasoning'].strip():
        issues.append('reasoning_missing')
    if not thinking and result['reasoning'].strip():
        issues.append('nonthinking_has_reasoning')
    calls = result['tool_calls']
    if expect_tool:
        if len(calls) != 1:
            issues.append('expected_one_tool')
        else:
            try:
                if not calls[0].get('id') or calls[0]['function']['name'] != 'add' or json.loads(calls[0]['function']['arguments']) != {'a': 2, 'b': 3}:
                    issues.append('wrong_tool_or_arguments')
            except (ValueError, KeyError, TypeError):
                issues.append('invalid_tool_arguments')
    else:
        if calls:
            issues.append('unexpected_tool_call')
        if json_answer:
            try:
                if json.loads(result['content']) != {'answer': 5}:
                    issues.append('incorrect_json_answer')
            except (ValueError, TypeError):
                issues.append('invalid_json')
        elif expected is not None and result['content'].strip() != expected:
            issues.append('incorrect_final_answer')
    return issues


def assistant_message(result):
    message = {'role': 'assistant', 'content': result['content']}
    # vLLM accepts the canonical API history field and aliases it to the
    # checkpoint template's reasoning_content internally.
    if result['reasoning']:
        message['reasoning'] = result['reasoning']
    if result['tool_calls']:
        message['tool_calls'] = result['tool_calls']
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='dots3-note-exl3-k4')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--require-mtp', action='store_true', help='Require speculative draft counter increase during the gate')
    parser.add_argument('--require-boundary-chunk', action='store_true', help='Require a structured thinking stream chunk with </think> followed by content token(s)')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0 or args.repeats <= 0:
        parser.error('Token limit, timeout and repeats must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    before_metrics = metrics(args.base_url)
    with args.output.open('x') as output:
        output.write(json.dumps({'record': 'meta', 'schema': 'dots3-reasoning-api-v1',
            'args': vars(args) | {'output': str(args.output)},
            'scope': 'Short functional gate, not a performance benchmark; include_reasoning stays true.',
            'speculative_metrics_before': before_metrics}) + '\n')

        def run(case, thinking, streaming, messages, *, mode=None, schema=False, expected=None, dependency=None):
            payload = {'model': args.model, 'messages': messages, 'temperature': 0,
                'max_tokens': args.max_tokens, 'stream': streaming, 'include_reasoning': True,
                'return_token_ids': True,
                'chat_template_kwargs': {'enable_thinking': thinking}}
            if streaming:
                payload['stream_options'] = {'include_usage': True}
            if mode is not None:
                payload.update(tools=[TOOL], tool_choice=mode if mode != 'named' else
                    {'type': 'function', 'function': {'name': 'add'}})
            if schema:
                payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'addition', 'strict': True, 'schema': {'type': 'object',
                    'properties': {'answer': {'type': 'integer', 'const': 5}},
                    'required': ['answer'], 'additionalProperties': False}}}
            if dependency is not None:
                row = {'record': 'case', 'case': case, 'thinking': thinking, 'streaming': streaming,
                    'passed': False, 'issues': ['dependency_failed'], 'dependency': dependency,
                    'request': payload, 'result': None}
            else:
                result = request(args.base_url, payload, args.timeout)
                issues = violations(result, thinking, expected=expected,
                                    expect_tool=mode in ('auto', 'named', 'required'), json_answer=schema)
                row = {'record': 'case', 'case': case, 'thinking': thinking, 'streaming': streaming,
                       'passed': not issues, 'issues': issues, 'request': payload, 'result': result}
            row['repeat'] = repeat
            row['grammar_constrained'] = schema or mode in ('named', 'required')
            records.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({k: row[k] for k in ('case', 'repeat', 'thinking', 'streaming', 'passed', 'issues')}), flush=True)
            return row

        for repeat in range(args.repeats):
            for thinking in (False, True):
                for streaming in (False, True):
                    first_messages = [{'role': 'user', 'content': 'What is 2 + 3? Keep any reasoning brief and give only the number as your final answer.'}]
                    first = run('first_turn', thinking, streaming, first_messages, expected='5')
                    second_messages = first_messages + ([assistant_message(first['result'])] if first['passed'] else []) + [
                        {'role': 'user', 'content': 'Add 1 to your previous answer. Give only the number as your final answer.'}]
                    run('second_turn', thinking, streaming, second_messages, expected='6',
                        dependency=None if first['passed'] else 'first_turn')
                    run('json', thinking, streaming, [{'role': 'user', 'content': 'What is 2 + 3? Keep any reasoning brief. Return JSON with the integer answer in the answer field.'}], schema=True)
                    for mode in ('auto', 'named', 'required', 'none'):
                        text = ('Do not call any tools. What is 2 + 3? Give only the number as your final answer.' if mode == 'none' else
                                'Call add exactly once with a=2 and b=3. Keep any reasoning brief. After receiving its result, give only the resulting number as your final answer.')
                        messages = [{'role': 'user', 'content': text}]
                        row = run('tool_' + mode, thinking, streaming, messages, mode=mode,
                                  expected='5' if mode == 'none' else None)
                        if mode != 'none':
                            followup = list(messages)
                            if row['passed']:
                                followup += [assistant_message(row['result']), {'role': 'tool',
                                    'tool_call_id': row['result']['tool_calls'][0]['id'], 'content': '{"result":5}'}]
                            run('tool_' + mode + '_return', thinking, streaming, followup, mode='none', expected='5',
                                dependency=None if row['passed'] else 'tool_' + mode)
        after_metrics = metrics(args.base_url)
        draft_counter = 'vllm:spec_decode_num_draft_tokens_total'
        draft_delta = after_metrics['totals'].get(draft_counter, 0) - before_metrics['totals'].get(draft_counter, 0)
        boundaries = [{'case': row['case'], 'repeat': row['repeat'], 'chunks': row['result']['reasoning_boundary_chunks']}
                      for row in records if row['result'] is not None and row['thinking']
                      and row['grammar_constrained'] and row['result']['reasoning_boundary_chunks']]
        evidence_issues = []
        if args.require_mtp and (before_metrics.get('error') or after_metrics.get('error') or draft_delta <= 0):
            evidence_issues.append('no_verified_speculative_activity')
        if args.require_boundary_chunk and not boundaries:
            evidence_issues.append('no_observed_think_to_structured_token_chunk')
        summary = {'record': 'summary', 'passed': all(row['passed'] for row in records) and not evidence_issues,
            'cases_passed': sum(row['passed'] for row in records), 'cases_total': len(records),
            'failed_cases': [{k: row[k] for k in ('case', 'repeat', 'thinking', 'streaming', 'issues')} for row in records if not row['passed']],
            'evidence_issues': evidence_issues, 'speculative_draft_token_delta': draft_delta,
            'structured_boundary_chunks': boundaries, 'speculative_metrics_after': after_metrics,
            'boundary_evidence_scope': 'An SSE token chunk crossing </think> shows an accepted output burst; it does not expose rejected drafts or prove their exact contents.'}
        output.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
