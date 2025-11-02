from __future__ import annotations

import os
import asyncio
from typing import Dict

from jinx.micro.embeddings.project_paths import PROJECT_INDEX_DIR


async def _read_text_safe(path: str, max_chars: int = 60000) -> str:
    try:
        def _read() -> str:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()[-max_chars:]
            except Exception:
                return ''
        return await asyncio.to_thread(_read)
    except Exception:
        return ''


async def _read_json(path: str) -> dict:
    try:
        def _load() -> dict:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        return await asyncio.to_thread(_load)
    except Exception:
        return {}


async def scan_framework_markers(project_root: str) -> Dict[str, float]:
    nodes: Dict[str, float] = {}

    # NodeJS: package.json
    try:
        pkg = os.path.join(project_root, 'package.json')
        if os.path.exists(pkg):
            obj = await _read_json(pkg)
            deps = {}
            deps.update(obj.get('dependencies') or {})
            deps.update(obj.get('devDependencies') or {})
            scripts = obj.get('scripts') or {}
            keys = {str(k).strip().lower() for k in deps.keys()}
            dep_map = {
                'react': 2.0, 'next': 2.5, 'vue': 2.0, 'nuxt': 2.5, 'svelte': 2.0,
                'angular': 2.0, 'express': 1.8, 'koa': 1.6, 'nestjs': 2.2, 'vite': 1.6,
                'webpack': 1.4, 'astro': 1.8, 'gatsby': 1.8, 'storybook': 1.4,
            }
            for k in list(keys)[:2000]:
                if k in dep_map:
                    nodes[f"framework: {k}"] = nodes.get(f"framework: {k}", 0.0) + dep_map[k]
                    nodes['lang: node'] = nodes.get('lang: node', 0.0) + 1.0
            # Scripts
            s_low = {str(a).lower(): str(b).lower() for a, b in (scripts or {}).items()}
            script_map = {
                'next': ('next dev', 'next build', 'next start'),
                'react': ('react-scripts',),
                'vue': ('vue-cli-service',),
                'nuxt': ('nuxt',),
                'svelte': ('svelte-kit', 'vite dev'),
                'angular': ('ng ', 'angular-cli'),
                'nestjs': ('nest ',),
                'vite': ('vite ',),
                'webpack': ('webpack ',),
                'storybook': ('storybook ',),
            }
            for fw, patt in script_map.items():
                for name, cmd in s_low.items():
                    if any(p in cmd for p in patt):
                        nodes[f"framework: {fw}"] = nodes.get(f"framework: {fw}", 0.0) + 1.2
                        nodes['lang: node'] = nodes.get('lang: node', 0.0) + 0.6
                        break
            # Configs
            cfg_files = {
                'next': ('next.config.js', 'next.config.mjs', 'next.config.ts'),
                'nuxt': ('nuxt.config.js', 'nuxt.config.ts'),
                'svelte': ('svelte.config.js', 'svelte.config.ts'),
                'vite': ('vite.config.js', 'vite.config.ts'),
                'webpack': ('webpack.config.js',),
                'angular': ('angular.json',),
                'nestjs': ('nest-cli.json',),
                'storybook': ('.storybook',),
            }
            for fw, files in cfg_files.items():
                for fname in files:
                    path = os.path.join(project_root, fname)
                    if os.path.exists(path):
                        nodes[f"framework: {fw}"] = nodes.get(f"framework: {fw}", 0.0) + 0.8
                        nodes['lang: node'] = nodes.get('lang: node', 0.0) + 0.4
                        break
    except Exception:
        pass

    # Python: pyproject/requirements
    try:
        pyproj = os.path.join(project_root, 'pyproject.toml')
        reqs = os.path.join(project_root, 'requirements.txt')
        text = ''
        if os.path.exists(pyproj):
            text += await _read_text_safe(pyproj)
        if os.path.exists(reqs):
            text += '\n' + await _read_text_safe(reqs)
        if text:
            low = text.lower()
            for lib, fw in (
                ('django','django'), ('fastapi','fastapi'), ('flask','flask'), ('starlette','starlette'),
                ('pydantic','pydantic'), ('celery','celery'), ('sqlalchemy','sqlalchemy'),
            ):
                if lib in low:
                    nodes[f"framework: {fw}"] = nodes.get(f"framework: {fw}", 0.0) + 2.0
                    nodes['lang: python'] = nodes.get('lang: python', 0.0) + 1.0
    except Exception:
        pass

    # Imports hint
    try:
        entries = [os.path.join(PROJECT_INDEX_DIR, f) for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith('.json')]
    except Exception:
        entries = []
    seen_fw: set[str] = set()
    for p in entries[:120]:
        try:
            obj = await _read_json(p)
            imports = obj.get('imports') or []
            if isinstance(imports, dict):
                imports = list(imports.keys())
            s = {str(x).lower() for x in imports[:128]}
            for lib, fw in (('django','django'),('fastapi','fastapi'),('flask','flask'),('starlette','starlette'),('celery','celery'),('sqlalchemy','sqlalchemy')):
                if fw in seen_fw:
                    continue
                if lib in s:
                    nodes[f"framework: {fw}"] = nodes.get(f"framework: {fw}", 0.0) + 1.2
                    nodes['lang: python'] = nodes.get('lang: python', 0.0) + 0.6
                    seen_fw.add(fw)
        except Exception:
            continue

    # Tooling
    try:
        if os.path.exists(os.path.join(project_root, 'Pipfile')):
            nodes['framework: pipenv'] = nodes.get('framework: pipenv', 0.0) + 0.8
            nodes['lang: python'] = nodes.get('lang: python', 0.0) + 0.4
        pyproj = os.path.join(project_root, 'pyproject.toml')
        if os.path.exists(pyproj):
            text = await _read_text_safe(pyproj)
            if '[tool.poetry]' in text:
                nodes['framework: poetry'] = nodes.get('framework: poetry', 0.0) + 0.8
                nodes['lang: python'] = nodes.get('lang: python', 0.0) + 0.4
        if os.path.exists(os.path.join(project_root, 'environment.yml')) or os.path.exists(os.path.join(project_root, 'environment.yaml')):
            nodes['framework: conda'] = nodes.get('framework: conda', 0.0) + 0.8
            nodes['lang: python'] = nodes.get('lang: python', 0.0) + 0.4
        if os.path.exists(os.path.join(project_root, 'Dockerfile')):
            nodes['framework: docker'] = nodes.get('framework: docker', 0.0) + 1.2
        if os.path.exists(os.path.join(project_root, 'docker-compose.yml')) or os.path.exists(os.path.join(project_root, 'docker-compose.yaml')):
            nodes['framework: docker-compose'] = nodes.get('framework: docker-compose', 0.0) + 1.0
        for fn in ('k8s', 'deployment.yaml', 'deployment.yml'):
            if os.path.exists(os.path.join(project_root, fn)):
                nodes['framework: kubernetes'] = nodes.get('framework: kubernetes', 0.0) + 1.0
                break
    except Exception:
        pass

    return nodes


__all__ = ["scan_framework_markers"]
