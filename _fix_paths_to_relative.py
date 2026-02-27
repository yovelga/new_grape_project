"""
Transformation script: converts all hardcoded absolute project-root paths
in Python files to use Path(__file__).resolve().parents[N], making them
portable regardless of where the project folder is located.

Run once from the project root, then delete this file.
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ABSOLUTE_PATH = str(PROJECT_ROOT)          # C:\Users\yovel\OneDrive\Desktop\Grape_Project
ABSOLUTE_PATH_FWD = ABSOLUTE_PATH.replace("\\", "/")  # forward-slash variant

VAR_NAME = "_PROJECT_ROOT"

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def get_depth(py_file: Path) -> int:
    """Number of parent-steps from py_file up to PROJECT_ROOT."""
    return len(py_file.resolve().relative_to(PROJECT_ROOT).parts) - 1


def escape_for_regex(path: str) -> str:
    return re.escape(path)


def build_replacement(rest_of_path: str, raw_prefix: str) -> str:
    """
    Given the part of the string AFTER project root,
    build the replacement expression.
    """
    rest = rest_of_path.lstrip("\\/")
    if not rest:
        return f"str({VAR_NAME})"
    # normalise to forward-slash for Path() portability
    rest_normalized = rest.replace("\\", "/")
    return f'str({VAR_NAME} / {raw_prefix}"{rest_normalized}")'


def transform_content(content: str, depth: int) -> str:
    """
    1. Ensure `from pathlib import Path` is present.
    2. Insert `_PROJECT_ROOT = Path(__file__).resolve().parents[depth]` 
       after the last top-level import statement.
    3. Replace literal strings that start with the absolute project root
       with `str(_PROJECT_ROOT / "rest/of/path")`.
    """

    # Skip if already converted
    if VAR_NAME in content:
        # Still do the string replacements in case some were missed
        pass
    else:
        # ---- 1. Ensure pathlib import ----------------------------------------
        has_pathlib = bool(re.search(r'from pathlib import.*\bPath\b', content)
                           or re.search(r'import pathlib', content))

        # ---- 2. Find insertion point (after last top-level import) -----------
        lines = content.splitlines(keepends=True)
        last_import_line = -1
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            # skip docstrings/comments at top
            if re.match(r'^(import |from )\S', stripped):
                last_import_line = i

        pathlib_line = "" if has_pathlib else "from pathlib import Path\n"
        proj_root_line = f"{VAR_NAME} = Path(__file__).resolve().parents[{depth}]\n"
        insert_block = pathlib_line + proj_root_line

        if last_import_line >= 0:
            lines.insert(last_import_line + 1, insert_block)
        else:
            lines.insert(0, insert_block)

        content = "".join(lines)

    # ---- 3. Replace absolute path strings ------------------------------------
    #
    # Matches patterns like:
    #   r"C:\Users\...\Grape_Project\some\path"
    #   r"C:\Users\...\Grape_Project"
    #   "C:/Users/.../Grape_Project/some/path"
    #   Path(r"C:\Users\...\Grape_Project\rest")
    #   rf"C:\Users\...\Grape_Project\rest\{var}"  ← leave f-strings alone,
    #                                                 they're handled below
    #
    # We only replace plain r"..." and "..." literals (not f-strings) because
    # f-string rewrites need manual inspection.

    def replacer(m: re.Match) -> str:
        raw = m.group(1)        # 'r' or ''
        quote = m.group(2)      # " or '
        rest_path = m.group(3)  # everything after the project root, could be ''
        # If it was already converted somehow, skip
        if VAR_NAME in m.group(0):
            return m.group(0)
        rest_path = rest_path.lstrip("\\/")
        if not rest_path:
            replacement = f"str({VAR_NAME})"
        else:
            rest_normalized = rest_path.replace("\\", "/")
            # Keep raw prefix if needed; forward slashes are safe in Path()
            replacement = f'str({VAR_NAME} / r"{rest_normalized}")'
        return replacement

    # Backslash variant (raw strings): r"C:\Users\...\Grape_Project\something"
    back_pattern = (
        r'\b(r?)'                          # optional raw prefix
        + r'(")'                           # opening quote
        + re.escape(ABSOLUTE_PATH)         # the absolute path
        + r'((?:\\[^"\\]*)*)'              # optional \rest\of\path
        + r'"'                             # closing quote
    )
    content = re.sub(back_pattern, replacer, content)

    # Forward-slash variant: "C:/Users/.../Grape_Project/something"
    def replacer_fwd(m: re.Match) -> str:
        raw = m.group(1)
        rest_path = m.group(2)
        rest_path = rest_path.lstrip("/")
        if not rest_path:
            return f"str({VAR_NAME})"
        rest_normalized = rest_path.replace("\\", "/")
        return f'str({VAR_NAME} / "{rest_normalized}")'

    fwd_pattern = (
        r'\b(r?)'
        + r'"'
        + re.escape(ABSOLUTE_PATH_FWD)
        + r'((?:/[^"]*)*)'
        + r'"'
    )
    content = re.sub(fwd_pattern, replacer_fwd, content)

    # Single-quote backslash variant
    def replacer_sq(m: re.Match) -> str:
        raw = m.group(1)
        rest_path = m.group(2)
        rest_path = rest_path.lstrip("\\/")
        if not rest_path:
            return f"str({VAR_NAME})"
        rest_normalized = rest_path.replace("\\", "/")
        return f"str({VAR_NAME} / r'{rest_normalized}')"

    sq_pattern = (
        r"\b(r?)"
        + r"'"
        + re.escape(ABSOLUTE_PATH)
        + r"((?:\\[^'\\]*)*)"
        + r"'"
    )
    content = re.sub(sq_pattern, replacer_sq, content)

    # Single-quote forward-slash variant
    def replacer_sq_fwd(m: re.Match) -> str:
        raw = m.group(1)
        rest_path = m.group(2)
        rest_path = rest_path.lstrip("/")
        if not rest_path:
            return f"str({VAR_NAME})"
        rest_normalized = rest_path.replace("\\", "/")
        return f"str({VAR_NAME} / '{rest_normalized}')"

    sq_fwd_pattern = (
        r"\b(r?)"
        + r"'"
        + re.escape(ABSOLUTE_PATH_FWD)
        + r"((?:/[^']*)*)"
        + r"'"
    )
    content = re.sub(sq_fwd_pattern, replacer_sq_fwd, content)

    return content


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    changed = 0
    skipped = 0
    errors = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip virtual environments and hidden dirs
        dirs[:] = [d for d in dirs if d not in {
            "venv", ".venv", "__pycache__", ".git", "node_modules",
            "backup_before_temporal_removal", "backup_full_project",
        }]

        for fname in files:
            if not fname.endswith(".py"):
                continue
            py_file = Path(root) / fname

            # Skip this very script
            if py_file.resolve() == Path(__file__).resolve():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"  [ERROR reading] {py_file.relative_to(PROJECT_ROOT)}: {e}")
                errors += 1
                continue

            if ABSOLUTE_PATH not in content and ABSOLUTE_PATH_FWD not in content:
                skipped += 1
                continue

            depth = get_depth(py_file)
            new_content = transform_content(content, depth)

            if new_content != content:
                try:
                    py_file.write_text(new_content, encoding="utf-8")
                    print(f"  [OK] {py_file.relative_to(PROJECT_ROOT)}  (depth={depth})")
                    changed += 1
                except Exception as e:
                    print(f"  [ERROR writing] {py_file.relative_to(PROJECT_ROOT)}: {e}")
                    errors += 1
            else:
                print(f"  [unchanged] {py_file.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. Changed: {changed}, Unchanged: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
