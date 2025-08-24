from __future__ import annotations

import shutil
import textwrap


def pretty_echo(text: str, title: str = "Jinx") -> None:
    """Render model output in a neat ASCII box with a title.

    - Uses word-wrapping (no mid-word splits) for readability.
    - Preserves blank lines from the original text.
    - Avoids ANSI so it won't clash with prompt rendering.
    """
    width = shutil.get_terminal_size((80, 24)).columns
    width = max(50, min(width, 120))
    inner_w = width - 2

    # Title bar
    title_str = f" {title} " if title else ""
    title_len = len(title_str)
    if title_len and title_len + 2 < inner_w:
        top = "+-" + title_str + ("-" * (inner_w - title_len - 2)) + "+"
    else:
        top = "+" + ("-" * inner_w) + "+"
    bot = "+" + ("-" * inner_w) + "+"

    print(top)
    lines = text.splitlines() if text else [""]
    for ln in lines:
        wrapped = (
            textwrap.wrap(
                ln,
                width=inner_w,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            if ln.strip() != ""
            else [""]
        )
        for chunk in wrapped:
            pad = inner_w - len(chunk)
            print(f"|{chunk}{' ' * pad}|")
    print(bot + "\n")
