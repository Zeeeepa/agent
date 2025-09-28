from __future__ import annotations

from typing import Dict, List


def _looks_like_question(line: str) -> bool:
    t = (line or "").strip()
    if not t:
        return False
    # Language-agnostic: common question mark characters (ASCII '?', CJK '？', Arabic '؟')
    if any(ch in t for ch in ("?", "？", "؟")):
        return True
    # Prompt-like: ends with ':' and not obviously code-like
    if t.endswith(":") and len(t) <= 200:
        if not any(x in t for x in "(){}[];="):
            return True
    return False


def _extract_strings_from_code(code: str) -> List[str]:
    """Extract string literals from Python-like code, including triple quotes and prefixes.

    Lightweight parser to avoid heavy imports; good enough for question prompts.
    """
    s = code or ""
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ('"', "'") or (i + 1 < n and s[i] in "rRuUfFbB" and s[i+1] in ('"', "'")) or (i + 2 < n and s[i] in "rRuUfFbB" and s[i+1] in "rRuUfFbB" and s[i+2] in ('"', "'")):
            # detect prefix length
            j = i
            while j < n and s[j] in "rRuUfFbB":
                j += 1
            if j >= n:
                break
            quote = s[j]
            if quote not in ('"', "'"):
                i += 1
                continue
            # triple or single
            triple = (j + 2 < n and s[j] == s[j+1] == s[j+2])
            qlen = 3 if triple else 1
            start = j + qlen
            k = start
            esc = False
            while k < n:
                c = s[k]
                if not triple:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == quote:
                        break
                else:
                    if c == quote and k + 2 < n and s[k+1] == quote and s[k+2] == quote:
                        break
                k += 1
            # capture
            if k < n:
                out.append(s[start:k])
                i = (k + qlen)
            else:
                # unterminated; stop
                break
        else:
            i += 1
    return out


def _extract_question_from_python_blocks(text: str) -> str:
    # very lightweight scan of <python_...></python_...> bodies for print/input prompts
    s = text or ""
    out: List[str] = []
    pos = 0
    while True:
        i = s.find("<python_", pos)
        if i == -1:
            break
        j = s.find("</python_", i)
        if j == -1:
            break
        body = s[i:j]
        # grab inside after first '>'
        k = body.find(">");
        if k != -1:
            code = body[k + 1 :]
            out.extend(_extract_strings_from_code(code))
        pos = j + 9
    for cand in reversed(out):
        if _looks_like_question(cand):
            return cand
    return ""


def _last_question_from_agent(synth: str) -> str:
    text = synth or ""
    # Try to locate <python_question_...> blocks quickly
    i = text.rfind("<python_question_")
    if i != -1:
        j = text.find("</python_question_", i)
        if j != -1:
            body = text[i:j]
            k = body.find(">")
            if k != -1:
                return body[k + 1 :].strip()
    q = _extract_question_from_python_blocks(text)
    if q:
        return q
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _looks_like_question(ln):
            return ln
    return ""


def last_agent_question(synth: str) -> str:
    return _last_question_from_agent(synth)


def _last_user_query(synth: str) -> str:
    text = synth or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if len(ln) >= 12 and not ln.startswith("<python_") and not ln.startswith("<plan_"):
            return ln
    return ""


def last_user_query(synth: str) -> str:
    return _last_user_query(synth)


def extract_anchors(synth: str) -> Dict[str, List[str]]:
    text = synth or ""
    anchors: Dict[str, List[str]] = {"questions": [], "symbols": [], "paths": []}
    q = _last_question_from_agent(text)
    if q:
        anchors["questions"].append(q)

    for token in ["def ", "class "]:
        pos = 0
        while True:
            i = text.find(token, pos)
            if i == -1:
                break
            j = i + len(token)
            name = []
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                name.append(text[j])
                j += 1
            if name:
                anchors["symbols"].append("".join(name))
            pos = j

    pos = 0
    while True:
        i = text.find("(", pos)
        if i == -1:
            break
        j = i - 1
        name = []
        while j >= 0 and (text[j].isalnum() or text[j] == "_"):
            name.append(text[j])
            j -= 1
        if name:
            nm = "".join(reversed(name)).strip("_")
            if len(nm) >= 3 and nm[0].isalpha():
                anchors["symbols"].append(nm)
        pos = i + 1

    pos = 0
    while True:
        i = text.find(":\\", pos)
        if i == -1:
            break
        j = i - 1
        if j >= 0 and text[j].isalpha():
            k = i + 2
            path = [text[j], ":\\"]
            while k < len(text) and not text[k].isspace():
                path.append(text[k])
                k += 1
            anchors["paths"].append("".join(map(str, path)))
        pos = i + 2

    for token in ["/jinx/", "/agent/", "/src/", "/app/"]:
        pos = 0
        while True:
            i = text.find(token, pos)
            if i == -1:
                break
            k = i
            buf = []
            while k < len(text) and not text[k].isspace():
                buf.append(text[k])
                k += 1
            anchors["paths"].append("".join(buf))
            pos = k

    for k in anchors:
        seen = set()
        uniq: List[str] = []
        for v in anchors[k]:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        anchors[k] = uniq[:10]
    return anchors
