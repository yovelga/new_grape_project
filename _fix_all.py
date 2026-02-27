"""
Comprehensive final fixer for all remaining syntax issues in the transformed files.

Universal approach: for EVERY file with _PROJECT_ROOT -
  - Strip BOM
  - Remove _PROJECT_ROOT (and its paired pathlib import if added by us)
  - Re-insert both at the canonical top position (after __future__, after docstring)
  - Ensure from __future__ stays before from pathlib import Path
  - Remove duplicate pathlib imports
"""
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROJ_ROOT_PAT = re.compile(r'^_PROJECT_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]\r?\n', re.MULTILINE)
BOM = '\ufeff'


def fix_file(path: Path) -> tuple[bool, str]:
    raw = path.read_bytes()
    text = raw.decode('utf-8', errors='replace')
    original = text

    reason_parts = []

    # ---- 0. Normalize line endings to \n ----
    if '\r\n' in text or '\r' in text:
        text = text.replace('\r\n', '\n').replace('\r', '\n')

    # ---- 1. Strip ALL BOM characters ----
    if BOM in text:
        text = text.replace(BOM, '')
        reason_parts.append('BOM stripped')

    # ---- 2. Find and extract _PROJECT_ROOT ----
    m = PROJ_ROOT_PAT.search(text)
    if not m:
        # No _PROJECT_ROOT — just fix BOM if needed
        if text != original:
            path.write_text(text, encoding='utf-8', newline='\n')
            return True, ', '.join(reason_parts)
        return False, ''

    depth = int(m.group(1))
    proj_root_stmt = f'_PROJECT_ROOT = Path(__file__).resolve().parents[{depth}]\n'

    # ---- 3. Remove ALL occurrences of _PROJECT_ROOT ----
    text = PROJ_ROOT_PAT.sub('', text)
    reason_parts.append('removed _PROJECT_ROOT')

    # ---- 4. Remove duplicate `from pathlib import Path` lines — keep only first ----
    lines = text.splitlines(keepends=True)
    pathlib_indices = [i for i, l in enumerate(lines) if l.strip() == 'from pathlib import Path']
    if len(pathlib_indices) > 1:
        for idx in reversed(pathlib_indices[1:]):
            lines.pop(idx)
        text = ''.join(lines)
        reason_parts.append('removed dup pathlib import')

    # ---- 5. Fix `from __future__` ordering ----
    # from __future__ must appear BEFORE `from pathlib import Path`
    lines = text.splitlines(keepends=True)
    pathlib_idx = next((i for i, l in enumerate(lines) if l.strip() == 'from pathlib import Path'), None)
    future_idx = next((i for i, l in enumerate(lines) if l.strip() == 'from __future__ import annotations'), None)

    if pathlib_idx is not None and future_idx is not None and future_idx > pathlib_idx:
        future_line = lines.pop(future_idx)
        lines.insert(pathlib_idx, future_line)
        text = ''.join(lines)
        reason_parts.append('moved from __future__ before pathlib')

    # ---- 6. Re-insert _PROJECT_ROOT at canonical top position ----
    lines = text.splitlines(keepends=True)
    insert_pos = _canonical_insert_pos(lines)

    # Ensure `from pathlib import Path` is present at or before insert_pos
    has_pathlib = any(
        re.match(r'^from pathlib import.*\bPath\b', l) or re.match(r'^import pathlib', l)
        for l in lines
    )
    pathlib_before_insert = any(
        re.match(r'^from pathlib import.*\bPath\b', l) or re.match(r'^import pathlib', l)
        for l in lines[:insert_pos]
    )
    if not pathlib_before_insert:
        lines.insert(insert_pos, 'from pathlib import Path\n')
        insert_pos += 1
        if not has_pathlib:
            reason_parts.append('added pathlib import')

    lines.insert(insert_pos, proj_root_stmt)
    text = ''.join(lines)
    reason_parts.append(f're-inserted _PROJECT_ROOT at line {insert_pos + 1}')

    path.write_text(text, encoding='utf-8', newline='\n')
    return True, ', '.join(reason_parts)


def _canonical_insert_pos(lines: list[str]) -> int:
    """
    Return the line index at which to INSERT the _PROJECT_ROOT statement.
    Ordering:
      1. shebang / encoding comments (#!)
      2. from __future__ imports  ← must stay before everything
      3. module docstring          ← skip past it
      4. ← INSERT HERE
    """
    i = 0
    n = len(lines)

    # Skip shebang / encoding comments at the very top
    while i < n and re.match(r'^\s*#', lines[i]):
        i += 1

    # Skip from __future__ imports (must precede all other statements)
    while i < n and re.match(r'^from __future__\s+import', lines[i]):
        i += 1

    # Skip blank lines
    while i < n and lines[i].strip() == '':
        i += 1

    # Skip module docstring
    if i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            rest = stripped[3:]
            if rest.rstrip().endswith(quote) and len(rest.rstrip()) >= 3:
                i += 1  # single-line docstring
            else:
                i += 1
                while i < n:
                    if quote in lines[i]:
                        i += 1
                        break
                    i += 1

    return i


def main():
    fixed = 0
    errors = 0
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {
            '.venv', 'venv', '__pycache__', '.git',
            'backup_before_temporal_removal', 'backup_full_project',
        }]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fp = Path(dirpath) / fname
            if fp.name in ('_fix_bad_insertions.py', '_fix_paths_to_relative.py', '_fix_all.py'):
                continue
            try:
                ok, reason = fix_file(fp)
                if ok:
                    print(f'  FIXED [{reason}]: {fp.relative_to(ROOT)}')
                    fixed += 1
            except Exception as e:
                import traceback
                print(f'  ERROR: {fp.relative_to(ROOT)}: {e}')
                traceback.print_exc()
                errors += 1
    print(f'\nDone. Fixed: {fixed}, Errors: {errors}')


if __name__ == '__main__':
    main()
