"""Opt-in Glimmer video, ATEM constraints and DFlash checks with real weights.

Set SUROGATE_MUSE_TEST_URL to a running Glimmer server with vision, the
muse_glimmer tool parser and DFlash enabled. No tools are actually executed.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os

import pytest
import requests
from PIL import Image

pytestmark = pytest.mark.skipif(not os.getenv('SUROGATE_MUSE_TEST_URL'), reason='requires real Glimmer server')


def ask(messages, *, _url=None, **kwargs):
    response = requests.post((_url or os.environ['SUROGATE_MUSE_TEST_URL']) + '/v1/chat/completions', json={
        'model': 'muse', 'messages': messages or [{'role': 'user', 'content': 'unused raw token prompt'}], 'max_tokens': 1024, 'temperature': 0,
        'return_token_ids': True, **kwargs}, timeout=180)
    assert response.ok, response.text
    return response.json()


def weather_tools():
    return [{'type': 'function', 'function': {'name': 'weather', 'description': 'Get the weather.', 'strict': True,
        'parameters': {'type': 'object', 'properties': {
            'city': {'type': 'string', 'enum': ['Paris']},
            'units': {'type': 'string', 'enum': ['celsius']},
            'days': {'type': 'integer', 'minimum': 1, 'maximum': 3},
            'options': {'type': 'object', 'properties': {'rain': {'type': 'boolean'}},
                        'required': ['rain'], 'additionalProperties': False}},
            'required': ['city', 'units', 'days', 'options'], 'additionalProperties': False}}}]


@pytest.mark.parametrize('choice', ['required', {'type': 'function', 'function': {'name': 'weather'}}])
def test_native_atem_constraints(choice):
    result = ask([{'role': 'user', 'content': 'Use weather to check Paris, celsius, 2 days, rain true.'}],
                 tools=weather_tools(), tool_choice=choice, parallel_tool_calls=False)
    choice = result['choices'][0]
    assert choice['finish_reason'] == 'tool_calls'
    message = choice['message']
    assert len(message['tool_calls']) == 1
    call = message['tool_calls'][0]['function']
    assert call['name'] == 'weather'
    args = json.loads(call['arguments'])
    assert set(args) == {'city', 'units', 'days', 'options'}
    assert args['city'] == 'Paris' and args['units'] == 'celsius'
    assert type(args['days']) is int and 1 <= args['days'] <= 3
    assert type(args['options']['rain']) is bool
    assert '<atem:' not in (message.get('content') or '')


def media(kind, colors):
    frames = [Image.new('RGB', (112, 112), color) for color in colors]
    data = io.BytesIO()
    if kind == 'video':
        frames[0].save(data, format='GIF', save_all=True, append_images=frames[1:], duration=500, loop=0)
        mime = 'image/gif'
    else:
        frames[0].save(data, format='PNG')
        mime = 'image/png'
    return {'type': kind + '_url', kind + '_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(data.getvalue()).decode()}}


def test_temporal_video_and_mixed_requests():
    video = media('video', ['red', 'green', 'blue', 'yellow', 'purple', 'cyan'])
    prompts = [
        [{'role': 'user', 'content': [video, {'type': 'text', 'text': 'Describe the changing colors in this video.'}]}],
        [{'role': 'user', 'content': [media('image', ['red']), {'type': 'text', 'text': 'What color is the image?'}]}],
        [{'role': 'user', 'content': 'What is the capital of France?'}],
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(ask, prompts))
    for result in results:
        message = result['choices'][0]['message']
        assert result['choices'][0]['finish_reason'] != 'length'
        assert message.get('content')
        assert '<|message|>' not in message['content']
    assert 'red' in results[1]['choices'][0]['message']['content'].lower()
    assert 'paris' in results[2]['choices'][0]['message']['content'].lower()
    assert len(results[0]['prompt_token_ids']) > len(results[1]['prompt_token_ids'])


def test_strict_streaming_preserves_reasoning_and_call():
    response = requests.post(os.environ['SUROGATE_MUSE_TEST_URL'] + '/v1/chat/completions', json={
        'model': 'muse', 'messages': [{'role': 'user', 'content': 'Check the weather in Paris using the tool.'}],
        'tools': weather_tools(), 'tool_choice': 'required', 'temperature': 0,
        'max_tokens': 768, 'parallel_tool_calls': False, 'stream': True}, stream=True, timeout=180)
    assert response.ok, response.text
    content, reasoning, arguments, names, finishes = [], [], [], [], []
    for line in response.iter_lines():
        if not line.startswith(b'data: ') or line == b'data: [DONE]':
            continue
        chunk = json.loads(line[6:])
        assert 'error' not in chunk, chunk
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            content.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            for call in delta.get('tool_calls', []):
                function = call.get('function', {})
                names.append(function.get('name') or '')
                arguments.append(function.get('arguments') or '')
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
    assert finishes == ['tool_calls']
    assert ''.join(names) == 'weather'
    assert json.loads(''.join(arguments))['city'] == 'Paris'
    assert '<atem:' not in ''.join(content)
    assert '<|message|>' not in ''.join(reasoning)


@pytest.mark.parametrize('length,temperature', [(64, 0), (2300, .7)])
def test_dflash_scores_and_cached_replay(length, temperature):
    prefix = [200000] + [100 + i % 23 for i in range(length - 1)]
    first = ask([], tokens=prefix, max_tokens=32, temperature=temperature,
                ignore_eos=True, logprobs=True, top_logprobs=3)
    assert first['prompt_token_ids'] == prefix
    generated = first['choices'][0]['token_ids']
    scores = first['choices'][0]['logprobs']['content']
    assert len(generated) == len(scores) == 32
    for endpoint in [os.getenv('SUROGATE_MUSE_REFERENCE_URL'), None]:
        replay = ask([], _url=endpoint, tokens=prefix + generated, max_tokens=1, prompt_logprobs=3)
        expected = replay['prompt_logprobs']
        assert replay['prompt_token_ids'] == prefix + generated
        assert len(expected) == length + len(generated)
        differences = [abs(expected[length+i][str(token)]['logprob'] - scores[i]['logprob'])
                       for i, token in enumerate(generated)]
        print(f'Glimmer score replay length={length} temperature={temperature}: max difference {max(differences):.6f}')
        assert max(differences) < .12


@pytest.mark.skipif(not os.getenv('SUROGATE_MUSE_REFERENCE_URL'), reason='requires an ordinary decode reference')
def test_dflash_preserves_greedy_target_output():
    prompt = [{'role': 'user', 'content': 'What is the capital of France?'}]
    expected = ask(prompt, _url=os.environ['SUROGATE_MUSE_REFERENCE_URL'], max_tokens=128)
    actual = ask(prompt, max_tokens=128)
    assert actual['prompt_token_ids'] == expected['prompt_token_ids']
    assert actual['choices'][0]['token_ids'] == expected['choices'][0]['token_ids']
    assert actual['choices'][0]['message'] == expected['choices'][0]['message']


@pytest.mark.parametrize('use_tool', [False, True])
def test_strict_auto_keeps_tool_choice_automatic(use_tool):
    prompt = ('Use weather to check Paris in celsius for 2 days with rain true.' if use_tool
              else 'What is 2+2? Answer directly without using any tool.')
    result = ask([{'role': 'user', 'content': prompt}], tools=weather_tools(),
                 tool_choice='auto', parallel_tool_calls=False)
    choice = result['choices'][0]
    message = choice['message']
    if use_tool:
        assert choice['finish_reason'] == 'tool_calls'
        assert len(message['tool_calls']) == 1
        assert message['tool_calls'][0]['function']['name'] == 'weather'
        assert json.loads(message['tool_calls'][0]['function']['arguments'])['city'] == 'Paris'
    else:
        assert not message.get('tool_calls')
        assert '4' in message['content']
