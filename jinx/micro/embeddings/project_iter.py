from __future__ import annotations

import os
from typing import Iterable, Iterator, Tuple

from .project_util import file_should_include


def iter_candidate_files(
    root: str,
    *,
    include_exts: list[str],
    exclude_dirs: list[str],
    max_file_bytes: int,
) -> Iterator[Tuple[str, str]]:
    """Yield (abs_path, rel_path) for files under root that pass filters.

    - Prunes excluded directories in-place for efficiency.
    - Applies extension and size filters.
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            abs_p = os.path.join(dirpath, fn)
            if not file_should_include(abs_p, include_exts=include_exts, exclude_dirs=exclude_dirs):
                continue
            try:
                if os.path.getsize(abs_p) > max_file_bytes:
                    continue
            except Exception:
                continue
            rel_p = os.path.relpath(abs_p, start=root)
            yield abs_p, rel_p
