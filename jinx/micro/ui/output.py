from __future__ import annotations

import shutil
import textwrap
import asyncio
import importlib


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


async def pretty_echo_async(text: str, title: str = "Jinx") -> None:
    """Async variant of pretty_echo with cooperative yields and PTK-safe stdout.

    - Uses prompt_toolkit.patch_stdout to avoid TTY contention with the active prompt.
    - Yields to the event loop every few lines to keep input responsive.
    """
    try:
        patch_stdout = importlib.import_module("prompt_toolkit.patch_stdout").patch_stdout  # type: ignore[assignment]
        print_formatted_text = importlib.import_module("prompt_toolkit").print_formatted_text  # type: ignore[assignment]
        FormattedText = importlib.import_module("prompt_toolkit.formatted_text").FormattedText  # type: ignore[assignment]
    except Exception:
        # Fallback to sync printing in a thread if PTK unavailable
        await asyncio.to_thread(pretty_echo, text, title)
        return

    width = shutil.get_terminal_size((80, 24)).columns
    width = max(50, min(width, 120))
    inner_w = width - 2

    title_str = f" {title} " if title else ""
    title_len = len(title_str)
    if title_len and title_len + 2 < inner_w:
        top = "+-" + title_str + ("-" * (inner_w - title_len - 2)) + "+"
    else:
        top = "+" + ("-" * inner_w) + "+"
    bot = "+" + ("-" * inner_w) + "+"

    ft = FormattedText
    with patch_stdout(raw=True):
        print_formatted_text(ft([("", top)]))
        if not text:
            print_formatted_text(ft([("", f"|{' ' * inner_w}|")]))
        else:
            count = 0
            for ln in text.splitlines():
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
                    print_formatted_text(ft([("", f"|{chunk}{' ' * pad}|")]))
                    count += 1
                    if (count % 20) == 0:
                        # yield every 20 lines
                        await asyncio.sleep(0)
        print_formatted_text(ft([("", bot + "\n")]))
