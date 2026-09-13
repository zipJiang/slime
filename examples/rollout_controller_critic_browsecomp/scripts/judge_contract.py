"""A malformed or incomplete judge result is an error, never a negative label."""
import re


def verdict(text, finish_reason):
    if finish_reason != 'stop':
        raise ValueError('Judge did not finish normally')
    match = re.fullmatch(r'\s*(EQUIVALENT|DIFFERENT)\s*', text.rsplit('</think>', 1)[-1])
    if not match:
        raise ValueError('Missing or ambiguous judge verdict')
    return match[1] == 'EQUIVALENT'
