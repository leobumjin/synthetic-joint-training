"""Gold-independent answer extraction, separate from requested-format compliance."""
from decimal import Decimal, InvalidOperation
import json
import re

PARSER_VERSION = 'explicit-answer-v3'


def numeric_answer(value):
    """Canonical exact decimal, with validated thousands separators and no rounding."""
    if not isinstance(value, str):
        raise ValueError('Numeric answer must be text')
    value = value.strip().replace('\N{MINUS SIGN}', '-')
    if len(value) > 1024 or not re.fullmatch(
            r'[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)'
            r'(?:[eE][+-]?[0-9]+)?', value):
        raise ValueError(f'Invalid numeric answer: {value!r}')
    try:
        number = Decimal(value.replace(',', ''))
    except InvalidOperation as exc:
        raise ValueError('Invalid number') from exc
    if not number.is_finite() or abs(number.as_tuple().exponent) > 1000:
        raise ValueError('Invalid or excessive numeric exponent')
    if number.is_zero():
        return '0'
    result = format(number, 'f')
    return result.rstrip('0').rstrip('.') if '.' in result else result


class _Object(dict):
    def __init__(self, pairs):
        super().__init__(pairs)
        self.duplicate_keys = len(pairs) != len(self)


def _atom(value, row):
    value = value.strip()
    # Remove matched wrappers only. Never remove arbitrary commas, units, or prose.
    pairs = [('**', '**'), ('__', '__'), ('`', '`'),
             (r'\[', r'\]'), (r'\(', r'\)'), ('$$', '$$'), ('$', '$'), ('(', ')')]
    for _ in range(12):
        previous = value
        if value.endswith('.') and not value.endswith('..'):
            value = value[:-1].strip()
        for left, right in pairs:
            if len(value) > len(left) + len(right) and value.startswith(left) and value.endswith(right):
                value = value[len(left):-len(right)].strip()
                break
        boxed = re.fullmatch(r'\\boxed\{([^{}]+)\}', value)
        if boxed:
            value = boxed.group(1).strip()
        if value == previous:
            break
    if row['benchmark'] == 'gsm8k':
        return numeric_answer(value)
    if row['benchmark'] in ('mmlu', 'mmlu_pro'):
        value = value.upper()
        if len(value) != 1 or not 'A' <= value <= chr(64 + len(row['input']['options'])):
            raise ValueError('Invalid option')
        return value
    try:
        return {'true': 'True', 'false': 'False', 'unknown': 'Unknown'}[value.lower()]
    except KeyError as exc:
        raise ValueError('Invalid FOLIO label') from exc


def parse_completion(text, row, style, answer_format_version='json'):
    """No gold-dependent extraction or guessing from an arbitrary last number.

    JSON objects, explicit answer markers, boxes, and standalone answers are
    accepted. All explicit candidates must be valid and agree. Reasoning inside
    completed <think> blocks is excluded from extraction, never format checks.
    """
    if style not in ('Direct', 'CoT') or answer_format_version not in ('json', 'legacy'):
        raise ValueError('Unsupported prompt style or answer format')
    if not isinstance(text, str):
        raise ValueError('Completion must be text')
    raw = text.strip()
    visible = re.sub(r'<think>.*?</think>', '', raw, flags=re.S | re.I).strip()
    unfinished_thinking = bool(re.search(r'</?think>', visible, re.I))
    candidates, containers, json_answers = [], [], []

    def add(value, source):
        try:
            parsed = _atom(value, row)
        except (ValueError, TypeError):
            parsed = None
        candidates.append((parsed, source))

    decoder = json.JSONDecoder(object_pairs_hook=_Object, parse_float=Decimal)
    position = 0
    while position < len(visible):
        if visible[position] not in '{[':
            position += 1
            continue
        try:
            obj, end = decoder.raw_decode(visible, position)
        except (ValueError, RecursionError):
            # An explicit but malformed answer must not disappear when another
            # valid answer occurs later in the same completion.
            if re.match(r'\{\s*"answer"\s*:', visible[position:]):
                candidates.append((None, 'json'))
            position += 1
            continue
        containers.append((position, end))
        if isinstance(obj, _Object) and 'answer' in obj:
            value = obj['answer']
            if obj.duplicate_keys or isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
                candidates.append((None, 'json'))
            elif isinstance(value, str):
                add(value, 'json')
            elif row['benchmark'] == 'gsm8k':
                add(str(value), 'json')
            else:
                candidates.append((None, 'json'))
            schema_valid = not obj.duplicate_keys and set(obj) == {'answer'} and isinstance(value, str)
            json_answers.append((position, end, schema_valid))
        position = end

    def inside_container(start):
        return any(left <= start < right for left, right in containers)

    # Generic 'answer' is anchored to a line; 'final answer' can also follow
    # concluding prose. JSON keys and quoted strings are handled by the decoder.
    marker_pattern = re.compile(
        r'(?im)(?:(?<!\w)(?:the[ \t]+)?final[ \t]+answer|'
        r'^[ \t]*(?:\#{1,6}[ \t]+)?(?:\*\*|__)?(?:the[ \t]+)?answer)'
        r'[ \t]*(?:\*\*|__)?[ \t]*(?::|\bis\b)[ \t]*(?:(?:\*\*|__)(?=[ \t]))?[ \t]*')
    markers = [m for m in marker_pattern.finditer(visible) if not inside_container(m.start())]
    if row['benchmark'] == 'gsm8k':
        markers += [m for m in re.finditer(r'(?m)^[ \t]*\#{4}[ \t]*', visible)
                    if not inside_container(m.start())]
    for marker in markers:
        start = marker.end()
        while start < len(visible) and visible[start].isspace():
            start += 1
        tail = visible[start:]
        fence = re.match(r'```(?:json)?[ \t]*\n\s*', tail, re.I)
        if fence:
            start += fence.end()
            tail = visible[start:]
        if any(left == start for left, _, _ in json_answers):
            continue  # Same answer, not a second invalid scalar candidate.
        if tail.startswith((r'\[', '$$')):
            right = r'\]' if tail.startswith(r'\[') else '$$'
            closing = tail.find(right, 2)
            value = tail[:closing + len(right)] if closing >= 0 else tail
        else:
            value = tail.split('\n', 1)[0]
        add(value, 'marker')
    for match in re.finditer(r'\\boxed\{([^{}]+)\}', visible):
        if not inside_container(match.start()):
            add(match.group(), 'boxed')
    if not candidates:
        try:
            standalone = _atom(visible, row)
        except ValueError:
            pass
        else:
            candidates.append((standalone, 'standalone'))
    values = [value for value, _ in candidates]
    valid = not unfinished_thinking and bool(values) and None not in values and len(set(values)) == 1
    parsed = values[0] if valid else None

    def requested_value(value):
        try:
            if row['benchmark'] == 'gsm8k':
                return numeric_answer(value) == parsed
            if row['benchmark'] in ('mmlu', 'mmlu_pro'):
                return value.strip().upper() == parsed
            return value.strip().lower() == parsed.lower()
        except (ValueError, AttributeError):
            return False

    # Check the ORIGINAL completion, not the text after stripping reasoning tags.
    json_valid = False
    if valid and len(json_answers) == 1 and json_answers[0][2]:
        final_line = raw.splitlines()[-1].strip() if raw else ''
        try:
            obj, end = decoder.raw_decode(final_line)
            exact_line = (isinstance(obj, _Object) and not obj.duplicate_keys and
                          set(obj) == {'answer'} and isinstance(obj['answer'], str) and
                          not final_line[end:].strip() and requested_value(obj['answer']))
        except (ValueError, TypeError, RecursionError):
            exact_line = False
        json_valid = bool(exact_line and (style == 'CoT' or raw == final_line))
    if answer_format_version == 'legacy':
        if style == 'Direct':
            try:
                compliant = valid and _atom(raw, row) == parsed
            except ValueError:
                compliant = False
        else:
            compliant = valid and bool(markers)
    else:
        compliant = json_valid
    if unfinished_thinking:
        error = 'unfinished_thinking'
    elif not candidates:
        error = 'missing_answer'
    elif None in values:
        error = 'invalid_explicit_answer'
    elif len(set(values)) > 1:
        error = 'conflicting_answers'
    else:
        error = None
    return {'parsed_value': parsed, 'format_valid': valid, 'answer_extracted': valid,
            'requested_format_valid': compliant, 'correct': valid and parsed == row['gold'],
            'extraction_source': '+'.join(sorted({source for _, source in candidates})) if valid else None,
            'extraction_error': error, 'parser_version': PARSER_VERSION}
