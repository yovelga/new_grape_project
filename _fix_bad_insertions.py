"""
Fix script: the previous transformation inserted _PROJECT_ROOT in wrong positions
(inside try/except blocks). This script finds the misplaced line, removes it, and
re-inserts it correctly at the top of each file (after docstring/encoding comments).
"""
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROJ_ROOT_PAT = re.compile(
    r'^_PROJECT_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]\n',
    re.MULTILINE
)
PATHLIB_IMPORT_PAT = re.compile(r'^from pathlib import Path\n', re.MULTILINE)


def top_insertion_index(lines: list[str]) -> int:
    """
    Return the index (in `lines`) AFTER which we should insert the
    _PROJECT_ROOT line: i.e. skip any coding declarations on line 0/1
    and any opening module docstring.
    """
    i = 0
    n = len(lines)

    # Skip coding / shebang declarations on first two lines
    while i < min(2, n) and (lines[i].startswith('#') or lines[i].strip() == ''):
        i += 1

    # Skip opening module docstring (triple-quoted)
    if i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            # single-line docstring?
            rest = stripped[3:]
            if rest.rstrip().endswith(quote) and len(rest.rstrip()) >= 3:
                i += 1  # whole docstring on one line
            else:
                # multi-line: advance until closing quote
                i += 1
                while i < n:
                    if quote in lines[i]:
                        i += 1
                        break
                    i += 1

    return i  # insert *before* lines[i]


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding='utf-8-sig')  # strips BOM if present
    lines = text.splitlines(keepends=True)

    # Find the misplaced _PROJECT_ROOT line
    m = PROJ_ROOT_PAT.search(text)
    if not m:
        return False  # nothing to fix

    depth = int(m.group(1))
    proj_root_line = f'_PROJECT_ROOT = Path(__file__).resolve().parents[{depth}]\n'

    # Check if it is already at a module-level safe position:
    # i.e. the line is not preceded by indented code
    line_start = m.start()
    # Find which line number this is
    line_idx = text[:line_start].count('\n')
    if line_idx < len(lines):
        prev_lines = lines[:line_idx]
        # If previous non-empty line is indented, this is misplaced
        for prev in reversed(prev_lines):
            if prev.strip():
                if prev.startswith('    ') or prev.startswith('\t'):
                    break  # misplaced
                else:
                    return False  # already at top level, no fix needed
    
    # --- Remove the misplaced _PROJECT_ROOT line ---
    # Also remove the `from pathlib import Path` that was inserted just before it
    # only if it appears to be the one we added (no content between import and _PROJECT_ROOT)
    new_text = PROJ_ROOT_PAT.sub('', text, count=1)

    # Check if there's a dangling `from pathlib import Path` right before where it was
    # (the script added it only if not present; check if the one that's there now is ours)
    # Strategy: remove duplicate `from pathlib import Path` if it appears twice
    pathlib_matches = list(PATHLIB_IMPORT_PAT.finditer(new_text))
    if len(pathlib_matches) >= 2:
        # Keep only the first, remove subsequent ones that are not at module top level
        for pm in reversed(pathlib_matches[1:]):
            pm_line = new_text[:pm.start()].count('\n')
            new_lines = new_text.splitlines(keepends=True)
            # check if the preceding non-empty line is indented
            for prev in reversed(new_lines[:pm_line]):
                if prev.strip():
                    if prev.startswith('    ') or prev.startswith('\t'):
                        # this pathlib import is inside a block - remove it
                        new_text = new_text[:pm.start()] + new_text[pm.end():]
                    break

    # --- Insert properly at top ---
    new_lines = new_text.splitlines(keepends=True)
    insert_pos = top_insertion_index(new_lines)

    # Ensure `from pathlib import Path` is present somewhere before insert_pos
    has_pathlib_before = any(
        'from pathlib import Path' in l or 'import pathlib' in l
        for l in new_lines[:insert_pos]
    )
    has_pathlib_after = any(
        re.match(r'^from pathlib import.*\bPath\b', l) or re.match(r'^import pathlib', l)
        for l in new_lines
    )

    if not has_pathlib_before:
        if has_pathlib_after:
            # pathlib is imported later - that's fine, we'll add our own before it
            pass
        new_lines.insert(insert_pos, 'from pathlib import Path\n')
        insert_pos += 1

    new_lines.insert(insert_pos, proj_root_line)

    path.write_text(''.join(new_lines), encoding='utf-8')
    return True


def main():
    fixed = 0
    skipped = 0
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
            if fp.name in ('_fix_bad_insertions.py', '_fix_paths_to_relative.py'):
                continue
            try:
                if fix_file(fp):
                    print(f'  FIXED: {fp.relative_to(ROOT)}')
                    fixed += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f'  ERROR: {fp.relative_to(ROOT)}: {e}')
                errors += 1
    print(f'\nDone. Fixed: {fixed}, Skipped: {skipped}, Errors: {errors}')


if __name__ == '__main__':
    main()
