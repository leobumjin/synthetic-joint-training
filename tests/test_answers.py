"""Scoring regressions: final-answer extraction must not inflate or suppress scores."""
from decimal import localcontext

import pytest

from synthetic_joint_training.answers import numeric_answer, parse_completion, PARSER_VERSION


def task(name='gsm8k', gold='42'):
    return {'benchmark': name, 'input': {'options': ['one', 'two', 'three', 'four']}, 'gold': gold}


@pytest.mark.parametrize('text', [
    '42', '42.', 'Final answer: 42', 'The answer is 42.',
    'Therefore, the final answer is 42.', '**Final answer:** **42**.',
    'Final answer:\n42', 'Final answer: \\[\n42\n\\]',
    r'\boxed{42}', r'Final answer: $\boxed{42}$.',
    '{"answer":"42"}', '{"answer":42}', '{"answer":42.0}',
    'Final answer: {"answer":"42"}', 'Answer:\n{"answer":"42"}',
    '```json\n{"answer":"42"}\n```', 'Final answer:\n```json\n{"answer":"42"}\n```',
    'Reasoning has numbers 10 and 32.\n{"answer":"42"}',
    '<think>Answer: 17\n{"answer":"17"}</think>\n{"answer":"42"}',
    '42 + 0 = 42.\n#### 42',
    '{"reasoning":"10 + 32", "answer":"42"}',
])
def test_common_explicit_outputs_extract(text):
    result = parse_completion(text, task(), 'CoT')
    assert result['parsed_value'] == '42'
    assert result['correct'] and result['answer_extracted']
    assert result['extraction_error'] is None


@pytest.mark.parametrize('text', [
    '', 'Some numbers are 17 and 42.', '42 or 43',
    '{"answer":"4,2"}', '{"answer":"42 units"}',
    '{"answer":true}', '{"answer":null}', '{"answer":NaN}',
    '{"answer":{"answer":"42"}}', '[{"answer":"42"}]',
    '{"answer":"42","answer":"42"}', '{"answer":"42","answer":"43"}',
    '{"answer":"42"}\n{"answer":"invalid"}', '{"answer":"42"}\n{"answer":"43"}',
    '{"answer":"42"}\n{"answer":',
    'Final answer: 42\nFinal answer: 43',
    'Final answer: 42\n{"answer":"invalid"}',
    r'\boxed{41}' + '\n{"answer":"42"}',
    '<think>{"answer":"42"}', '{"answer":"42"}<think>unfinished',
    'Final answer: 4,2',
])
def test_invalid_ambiguous_and_unfinished_outputs_fail(text):
    result = parse_completion(text, task(), 'CoT')
    assert not result['answer_extracted'] and not result['correct']
    assert result['parsed_value'] is None
    assert result['extraction_error'] is not None
    assert not result['requested_format_valid']


@pytest.mark.parametrize('text,style,expected', [
    ('{"answer":"42"}', 'Direct', True),
    ('  {"answer":"42"}\n', 'Direct', True),
    ('Reasoning\n{"answer":"42"}', 'CoT', True),
    ('Reasoning\n{"answer":"42"}', 'Direct', False),
    ('Reasoning {"answer":"42"}', 'CoT', False),
    ('Final answer: {"answer":"42"}', 'CoT', False),
    ('{"answer":42}', 'Direct', False),
    ('{"reasoning":"steps", "answer":"42"}', 'CoT', False),
    ('{"answer":"**42**"}', 'Direct', False),
    ('{"answer":"42"}\n{"answer":"42"}', 'CoT', False),
    ('```json\n{"answer":"42"}\n```', 'CoT', False),
    ('<think>scratch</think>{"answer":"42"}', 'CoT', False),
    ('<think>scratch</think>\n{"answer":"42"}', 'CoT', True),
    ('<think>scratch</think>\n{"answer":"42"}', 'Direct', False),
    ('42', 'Direct', False),
    ('{"answer":"42"}\nExplanation after answer.', 'CoT', False),
])
def test_format_compliance_is_separate(text, style, expected):
    result = parse_completion(text, task(), style)
    assert result['correct']
    assert result['requested_format_valid'] is expected


@pytest.mark.parametrize('name,text,gold', [
    ('mmlu', '{"answer":"B"}', 'B'),
    ('mmlu', 'Final answer: (B).', 'B'),
    ('mmlu', '**B**', 'B'),
    ('mmlu_pro', 'Reasoning\n{"answer":"b"}', 'B'),
    ('folio', '{"answer":"Unknown"}', 'Unknown'),
    ('folio', 'Final answer: **False**.', 'False'),
    ('folio', '\\boxed{True}', 'True'),
])
def test_multiple_choice_and_logic(name, text, gold):
    assert parse_completion(text, task(name, gold), 'CoT')['correct']


@pytest.mark.parametrize('name,text', [
    ('mmlu', '{"answer":"E"}'), ('mmlu', '{"answer":1}'),
    ('mmlu', 'B because this is correct'), ('mmlu', '{"answer":"B or C"}'),
    ('folio', 'Not True'), ('folio', '{"answer":false}'),
    ('folio', '{"answer":"True or Unknown"}'),
])
def test_choice_and_logic_do_not_guess(name, text):
    assert not parse_completion(text, task(name), 'CoT')['answer_extracted']


@pytest.mark.parametrize('value,expected', [
    ('1,234.500', '1234.5'), ('-0.00', '0'), ('+0', '0'),
    ('−42', '-42'), ('.50', '0.5'), ('1e3', '1000'),
    ('1.234e-3', '0.001234'), ('123456789012345678901234567890123456789', '123456789012345678901234567890123456789'),
])
def test_exact_numeric_normalization(value, expected):
    with localcontext() as context:
        context.prec = 4
        assert numeric_answer(value) == expected


@pytest.mark.parametrize('value', ['4,2', '12,34', '1,,000', '1 000', '1,000,00',
                                   'NaN', 'Infinity', '1e999999', '42 dollars', '1/2'])
def test_malformed_numbers_rejected(value):
    with pytest.raises(ValueError):
        numeric_answer(value)


def test_extraction_is_independent_of_gold_and_style():
    for text in ['{"answer":42}', 'Final answer: 42', '17 + 25 = 42.',
                 '{"answer":"42"}\n{"answer":"43"}']:
        a = parse_completion(text, task(gold='42'), 'Direct')
        b = parse_completion(text, task(gold='17'), 'CoT')
        assert a['parsed_value'] == b['parsed_value']
        assert a['answer_extracted'] == b['answer_extracted']
    assert PARSER_VERSION == 'explicit-answer-v3'


def test_legacy_extraction_and_format():
    assert parse_completion('42', task(), 'Direct', 'legacy')['requested_format_valid']
    assert parse_completion('Final answer: 42', task(), 'CoT', 'legacy')['requested_format_valid']
    assert not parse_completion('42', task(), 'CoT', 'legacy')['requested_format_valid']
