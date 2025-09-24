from __future__ import annotations


def lang_for_file(path: str) -> str:
    p = path.lower()
    if p.endswith('.py'): return 'python'
    if p.endswith('.js'): return 'javascript'
    if p.endswith('.ts'): return 'typescript'
    if p.endswith('.tsx'): return 'tsx'
    if p.endswith('.jsx'): return 'jsx'
    if p.endswith('.go'): return 'go'
    if p.endswith('.java'): return 'java'
    if p.endswith('.cs'): return 'csharp'
    if p.endswith('.cpp') or p.endswith('.cc') or p.endswith('.cxx'): return 'cpp'
    if p.endswith('.c'): return 'c'
    if p.endswith('.rs'): return 'rust'
    if p.endswith('.php'): return 'php'
    if p.endswith('.rb'): return 'ruby'
    if p.endswith('.sh') or p.endswith('.bash'): return 'bash'
    if p.endswith('.ps1'): return 'powershell'
    if p.endswith('.json'): return 'json'
    if p.endswith('.yaml') or p.endswith('.yml'): return 'yaml'
    if p.endswith('.toml'): return 'toml'
    if p.endswith('.ini'): return 'ini'
    if p.endswith('.md'): return 'markdown'
    return ''
