from typing import List, Tuple

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