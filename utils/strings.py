from typing import Any, List, Tuple

class ConcatSafeStr(str):
    # For a string to be safe to directly concatenate to,
    # it should either be empty or end with a line break.
    def __new__(cls, val: Any) -> 'ConcatSafeStr':
        str_val = str(val)
        if str_val and not str_val.endswith('\n'):
            str_val += '\n'
        return super().new(cls, str_val)

def replace_at_indices(
    s: str,
    replacements: List[Tuple[Tuple[int, int], str]]
) -> str:
    """
    Replaces each `s[replacements[i][0][0]:replacements[i][0][1]]` (right exclusive)
    with `replacements[i][1]`.
    `replacements[i][0][0] == replacements[i][0][1]` achieves insertion.
    Assumes `replacements` is sorted and non-overlapping.
    """
    last_end = 0
    res = ""
    for (bgn, end), text in replacements:
        assert bgn >= last_end
        res += s[last_end:bgn] + text
        last_end = end

    res += s[last_end:]

    return res 