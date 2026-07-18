#!/usr/bin/env python3
"""
Citation audit builder for a LaTeX thesis.

Scans active .tex files for citation commands, cross-references each key
with bibliography/references.bib, and writes a rich citation_audit.csv
designed for downstream validation (by an LLM or a human reviewer) of
whether each cited paper actually supports the claim it is attached to.

Dependencies: Python 3.8+ standard library only.
"""

import re
import csv
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────

THESIS_DIR = Path(__file__).parent
BIB_FILE = THESIS_DIR / "bibliography" / "references.bib"

# Active .tex files compiled via main.tex \input commands.
ACTIVE_TEX_FILES = [
    THESIS_DIR / "chapters" / "abstract.tex",
    THESIS_DIR / "chapters" / "introduction.tex",
    THESIS_DIR / "chapters" / "literature_review.tex",
    THESIS_DIR / "chapters" / "objectives_hypotheses.tex",
    THESIS_DIR / "chapters" / "materials_methods.tex",
    THESIS_DIR / "chapters" / "results_new.tex",
    THESIS_DIR / "chapters" / "discussion.tex",
    THESIS_DIR / "chapters" / "conclusions.tex",
    THESIS_DIR / "chapters" / "appendix_results.tex",
]

OUTPUT_CSV = THESIS_DIR / "citation_audit.csv"

# ── Regex patterns ───────────────────────────────────────────────────────

# Citation command regex.
# Captures:
#   group 'cmd'      – the command name without backslash (e.g. "parencite")
#   group 'opts'     – the raw optional-argument block (may contain [pre][post])
#   group 'keys'     – the brace-delimited key list
CITE_RE = re.compile(
    r'\\(?P<cmd>(?:par(?:en)?cite|textcite|autocite|cite)\*?)'
    r'(?P<opts>(?:\s*\[[^\]]*\])*)'   # zero or more [optional] blocks
    r'\s*\{(?P<keys>[^}]+)\}',
)

# Extracts up to two optional arguments from the opts group.
# BibLaTeX convention: \parencite[prenote][postnote]{key}
# When only one bracket pair is present it is the postnote.
OPT_RE = re.compile(r'\[([^\]]*)\]')

# Section / chapter heading.
SECTION_RE = re.compile(
    r'\\(chapter|section|subsection|subsubsection)\s*'
    r'(?:\[[^\]]*\])?\s*'
    r'\{([^}]+)\}'
)

# ── CSV column order ─────────────────────────────────────────────────────

CSV_FIELDS = [
    "occurrence_id",
    "citation_key",
    "rendered_citation_text",
    "citation_command_type",
    "citation_command_full",
    "prenote",
    "postnote",
    "all_citations_in_same_command",
    "citation_position_in_cluster",
    "cluster_size",
    "is_multi_citation_cluster",
    "bib_entry_found",
    "doi_found",
    "doi",
    "title",
    "authors",
    "year",
    "journal_or_booktitle",
    "volume",
    "issue",
    "pages",
    "publisher",
    "url",
    "chapter_section",
    "source_tex_file",
    "source_line_number",
    "paragraph_text",
    "sentence_with_citation",
    "previous_sentence",
    "next_sentence",
    "local_context_window",
    "claim_summary",
    "claim_type",
    "citation_role_guess",
    "needs_manual_review",
    "manual_review_reason",
]


# ── BibTeX parser ────────────────────────────────────────────────────────

def parse_bib(bib_path: Path) -> dict:
    """Parse *bib_path* into ``{key: {field: value, ...}, ...}``.

    Handles nested braces (one level) in field values and the three common
    BibTeX value syntaxes: ``{braced}``, ``"quoted"``, and bare integers.
    """
    entries: dict = {}
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entry_re = re.compile(r'@(\w+)\s*\{([^,]+),', re.IGNORECASE)
    # Field value: handles one level of nested braces inside {}.
    field_re = re.compile(
        r'(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)"|(\d+))',
        re.IGNORECASE,
    )

    pos = 0
    while pos < len(text):
        m = entry_re.search(text, pos)
        if not m:
            break
        key = m.group(2).strip()
        # Walk forward to find the matching closing brace.
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        body = text[start:i - 1]
        pos = i

        entry: dict = {}
        for fm in field_re.finditer(body):
            fname = fm.group(1).lower()
            fval = fm.group(2) if fm.group(2) is not None else (
                fm.group(3) if fm.group(3) is not None else fm.group(4)
            )
            if fval:
                entry[fname] = re.sub(r'\s+', ' ', fval).strip()
        entries[key] = entry

    return entries


def bib_get(bib: dict | None, field: str) -> str:
    """Safely retrieve a field from a bib entry dict, returning '' on miss."""
    if bib is None:
        return ""
    return bib.get(field, "")


# ── Author rendering ────────────────────────────────────────────────────

def render_author_short(author_str: str) -> str:
    """Return a short citation-style author string (e.g. 'Smith et al.')."""
    if not author_str:
        return ""
    authors = re.split(r'\s+and\s+', author_str)
    if not authors:
        return ""

    def _last(name: str) -> str:
        name = name.strip()
        if ',' in name:
            return name.split(',')[0].strip().replace('{', '').replace('}', '')
        parts = name.split()
        return parts[-1].replace('{', '').replace('}', '') if parts else name

    if len(authors) == 1:
        return _last(authors[0])
    if len(authors) == 2:
        return f"{_last(authors[0])} and {_last(authors[1])}"
    return f"{_last(authors[0])} et al."


# ── LaTeX text cleaning ─────────────────────────────────────────────────

def clean_latex(text: str) -> str:
    """Produce a human-readable version of *text* by stripping LaTeX markup.

    This is intentionally approximate.  It removes citation commands,
    common formatting macros (\\textbf, \\textit, \\emph, \\gls, etc.),
    stray braces, and collapses whitespace.  Scientific content is
    preserved as much as possible.
    """
    s = text
    # Remove citation commands entirely.
    s = re.sub(
        r'\\(?:par(?:en)?cite|textcite|autocite|cite)\*?'
        r'(?:\s*\[[^\]]*\])*\s*\{[^}]+\}',
        '', s,
    )
    # \textbf{X}, \textit{X}, \emph{X}, \gls{X}, \textsubscript{X},
    # \textsuperscript{X}, \hyperref[...]{X}  →  X
    s = re.sub(r'\\(?:textbf|textit|emph|gls|textsubscript|textsuperscript)\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\hyperref\[[^\]]*\]\{([^}]*)\}', r'\1', s)
    # \textit{X} variants that may be nested – second pass.
    s = re.sub(r'\\(?:textbf|textit|emph)\{([^}]*)\}', r'\1', s)
    # Other common commands with one argument we don't care about.
    s = re.sub(r'\\(?:label|ref|eqref|Cref|cref)\{[^}]*\}', '', s)
    # Strip remaining backslash-commands that take no argument (e.g. \noindent).
    s = re.sub(r'\\(?:noindent|medskip|bigskip|smallskip|newpage|clearpage|FloatBarrier)\b', '', s)
    # \, \; \! \~ spacing
    s = re.sub(r'\\[,;!~]', ' ', s)
    # Strip leftover braces.
    s = s.replace('{', '').replace('}', '')
    # Collapse whitespace.
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── Paragraph extraction ────────────────────────────────────────────────

def extract_paragraph(lines: list[str], line_idx: int) -> str:
    """Return the full paragraph containing *line_idx*.

    A paragraph is a contiguous run of non-blank, non-comment-only lines.
    LaTeX blank lines (or lines that are only ``%`` comments) act as
    paragraph separators.
    """

    def _is_blank_or_comment(ln: str) -> bool:
        s = ln.strip()
        return s == '' or s.startswith('%')

    # Walk upward to find paragraph start.
    start = line_idx
    while start > 0 and not _is_blank_or_comment(lines[start - 1]):
        start -= 1

    # Walk downward to find paragraph end.
    end = line_idx
    while end < len(lines) - 1 and not _is_blank_or_comment(lines[end + 1]):
        end += 1

    para = ' '.join(lines[start:end + 1])
    para = re.sub(r'\s+', ' ', para).strip()
    # Cap at a reasonable length for CSV readability.
    if len(para) > 3000:
        para = para[:3000] + " [TRUNCATED]"
    return para


# ── Sentence splitting ───────────────────────────────────────────────────

# Heuristic sentence splitter.
# Splits on ". ", "! ", "? " after sentence-ending punctuation.
# Known limitation: abbreviations with periods that do not appear in our
# merge-back list will cause false splits.  This is a deliberate trade-off.


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a conservative heuristic.

    Strategy: split on punctuation followed by whitespace, then merge back
    fragments that end with common abbreviations.
    """
    raw = re.split(r'(?<=[.!?])\s+', text)
    # Merge fragments ending with known abbreviations back onto the next fragment.
    abbrev_re = re.compile(r'\b(?:e\.g|i\.e|et al|vs|Fig|Eq|Tab|Sec|Ch|Ref|cf|approx|Dr|Mr|Mrs|Prof|Vol|No)\.$')
    merged: list[str] = []
    carry = ""
    for frag in raw:
        if carry:
            frag = carry + " " + frag
            carry = ""
        if abbrev_re.search(frag):
            carry = frag
        else:
            merged.append(frag)
    if carry:
        merged.append(carry)
    return [s for s in merged if s.strip()]


def find_sentence_with_key(sentences: list[str], key: str):
    """Return (index, sentence) for the sentence containing *key*.

    Returns the first match.  Falls back to the first sentence that
    contains any citation command if *key* is not found literally.
    """
    for i, s in enumerate(sentences):
        if key in s:
            return i, s
    # Fallback: first sentence with any citation command.
    for i, s in enumerate(sentences):
        if re.search(r'\\(?:par(?:en)?cite|textcite|autocite|cite)', s):
            return i, s
    return 0, sentences[0] if sentences else ""


# ── Section heading lookup ───────────────────────────────────────────────

def find_section_heading(lines: list[str], line_idx: int) -> str:
    """Walk upward from *line_idx* to find the nearest heading."""
    for i in range(line_idx, -1, -1):
        if lines[i].lstrip().startswith('%'):
            continue
        m = SECTION_RE.search(lines[i])
        if m:
            return f"\\{m.group(1)}{{{m.group(2)}}}"
    return "(unknown)"


# ── Claim-type heuristic ─────────────────────────────────────────────────

_CLAIM_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("definition",          re.compile(r'\b(defin|is defined as|refers to|denotes)\b', re.I)),
    ("empirical_result",    re.compile(r'\b(achiev|result|found|demonstrat|observ|measur|yield|obtain|report|record)\b', re.I)),
    ("methodological_claim", re.compile(r'\b(method|pipeline|algorithm|implement|approach|framework|technique|protocol|procedure)\b', re.I)),
    ("comparison",          re.compile(r'\b(compar|outperform|superior|inferior|versus|while .+ achiev|contrast)\b', re.I)),
    ("motivation",          re.compile(r'\b(necessit|essential|crucial|important|critical|motivat|requir|need for|underscores)\b', re.I)),
    ("interpretation",      re.compile(r'\b(suggest|indicat|imply|interpret|consistent with|attribut|explain)\b', re.I)),
    ("dataset_or_materials", re.compile(r'\b(dataset|cultivar|vineyard|sensor|camera|instrument|sample|material)\b', re.I)),
    ("background_fact",     re.compile(r'\b(known|established|well.document|prior|previous|literature|review)\b', re.I)),
]


def classify_claim(text: str) -> str:
    """Return a heuristic claim-type label based on keyword matching."""
    for label, pat in _CLAIM_PATTERNS:
        if pat.search(text):
            return label
    return "unsupported_other"


# ── Citation-role heuristic ──────────────────────────────────────────────

_ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("contrast_or_comparison", re.compile(r'\b(however|contrast|whereas|unlike|although|conversely|while)\b', re.I)),
    ("method_reference",       re.compile(r'\b(using|via|follow|adopt|based on|implement|applied)\b', re.I)),
    ("direct_support",         re.compile(r'\\(?:parencite|cite)\{[^,}]+\}', re.I)),  # single-key cite
]


def classify_citation_role(text: str, cluster_size: int) -> str:
    """Guess how the citation is used in context."""
    for label, pat in _ROLE_PATTERNS:
        if label == "direct_support":
            # Only applies when the citation is alone.
            if cluster_size == 1 and pat.search(text):
                return "direct_support"
            continue
        if pat.search(text):
            return label
    if cluster_size > 2:
        return "grouped_support"
    if cluster_size == 1:
        return "direct_support"
    return "general_background"


# ── Main logic ───────────────────────────────────────────────────────────

def main():
    print("Parsing bibliography...")
    bib_entries = parse_bib(BIB_FILE)
    print(f"  Found {len(bib_entries)} BibTeX entries")

    rows: list[dict] = []
    stats = {
        "total": 0,
        "unique_keys": set(),
        "missing_bib": 0,
        "missing_doi": 0,
        "multi_clusters": 0,
        "needs_review": 0,
    }

    # Counter per (file, line) to disambiguate occurrence_id.
    occurrence_counter: dict[tuple, int] = {}

    for tex_path in ACTIVE_TEX_FILES:
        if not tex_path.exists():
            print(f"  WARNING: {tex_path} not found, skipping")
            continue

        rel_path = str(tex_path.relative_to(THESIS_DIR)).replace("\\", "/")
        print(f"  Scanning {rel_path}...")

        text = tex_path.read_text(encoding="utf-8", errors="replace")
        lines = text.split('\n')

        for line_idx, line in enumerate(lines):
            # Skip comment-only lines.
            if line.lstrip().startswith('%'):
                continue

            for m in CITE_RE.finditer(line):
                cmd_type = m.group("cmd")
                cmd_full = m.group(0)
                opts_raw = m.group("opts")
                keys = [k.strip() for k in m.group("keys").split(',') if k.strip()]
                cluster_size = len(keys)
                is_multi = cluster_size > 1

                # Parse prenote / postnote from optional args.
                opt_matches = OPT_RE.findall(opts_raw)
                if len(opt_matches) == 2:
                    prenote, postnote = opt_matches
                elif len(opt_matches) == 1:
                    prenote, postnote = "", opt_matches[0]
                else:
                    prenote, postnote = "", ""

                # ── Context extraction (once per command) ────────────
                paragraph = extract_paragraph(lines, line_idx)
                sentences = split_sentences(paragraph)
                # We match the first key to anchor the sentence lookup;
                # all keys share the same sentence context.
                anchor_key = keys[0] if keys else ""
                sent_idx, sent_text = find_sentence_with_key(sentences, anchor_key)
                prev_sent = sentences[sent_idx - 1] if sent_idx > 0 else ""
                next_sent = sentences[sent_idx + 1] if sent_idx < len(sentences) - 1 else ""

                # ±2 line context window.
                ctx_start = max(0, line_idx - 2)
                ctx_end = min(len(lines), line_idx + 3)
                ctx_window = ' '.join(lines[ctx_start:ctx_end])
                ctx_window = re.sub(r'\s+', ' ', ctx_window).strip()
                if len(ctx_window) > 1500:
                    ctx_window = ctx_window[:1500] + " [TRUNCATED]"

                section = find_section_heading(lines, line_idx)

                # Claim analysis (computed once, reused for each key).
                claim = clean_latex(sent_text)
                if not claim:
                    claim = "(extraction failed)"
                elif len(claim) > 300:
                    claim = claim[:300] + "..."
                claim_type = classify_claim(claim)

                if is_multi:
                    stats["multi_clusters"] += 1

                # ── Per-key row generation ───────────────────────────
                for pos_1based, key in enumerate(keys, start=1):
                    stats["total"] += 1
                    stats["unique_keys"].add(key)

                    bib = bib_entries.get(key)
                    bib_found = bib is not None
                    doi = bib_get(bib, "doi")
                    doi_found = bool(doi)
                    authors = bib_get(bib, "author")
                    year = bib_get(bib, "year")

                    author_short = render_author_short(authors)
                    rendered = f"{author_short}, {year}" if (author_short and year) else key

                    role = classify_citation_role(sent_text, cluster_size)

                    # ── Review flagging ───────────────────────────────
                    reasons: list[str] = []
                    if not bib_found:
                        reasons.append("missing_bib_entry")
                        stats["missing_bib"] += 1
                    if not doi_found:
                        reasons.append("missing_doi")
                        stats["missing_doi"] += 1
                    if is_multi:
                        reasons.append("multi_citation_cluster")
                    if not sent_text.strip():
                        reasons.append("short_context")
                    if "(extraction failed)" in claim or len(claim) < 15:
                        reasons.append("claim_extraction_unclear")
                    if role == "unclear":
                        reasons.append("unclear_role")
                    if claim_type == "unsupported_other":
                        reasons.append("unsupported_claim_type")

                    review = len(reasons) > 0
                    if review:
                        stats["needs_review"] += 1

                    # Build deterministic occurrence_id.
                    occ_key = (rel_path, line_idx + 1, key)
                    occurrence_counter[occ_key] = occurrence_counter.get(occ_key, 0) + 1
                    occ_idx = occurrence_counter[occ_key]
                    occurrence_id = f"{rel_path}:{line_idx + 1}:{key}:{occ_idx}"

                    rows.append({
                        "occurrence_id": occurrence_id,
                        "citation_key": key,
                        "rendered_citation_text": rendered,
                        "citation_command_type": cmd_type,
                        "citation_command_full": cmd_full,
                        "prenote": prenote,
                        "postnote": postnote,
                        "all_citations_in_same_command": ", ".join(keys),
                        "citation_position_in_cluster": pos_1based,
                        "cluster_size": cluster_size,
                        "is_multi_citation_cluster": str(is_multi).upper(),
                        "bib_entry_found": str(bib_found).upper(),
                        "doi_found": str(doi_found).upper(),
                        "doi": doi,
                        "title": bib_get(bib, "title"),
                        "authors": authors,
                        "year": year,
                        "journal_or_booktitle": bib_get(bib, "journal") or bib_get(bib, "booktitle"),
                        "volume": bib_get(bib, "volume"),
                        "issue": bib_get(bib, "number"),
                        "pages": bib_get(bib, "pages"),
                        "publisher": bib_get(bib, "publisher"),
                        "url": bib_get(bib, "url"),
                        "chapter_section": section,
                        "source_tex_file": rel_path,
                        "source_line_number": line_idx + 1,
                        "paragraph_text": paragraph,
                        "sentence_with_citation": sent_text,
                        "previous_sentence": prev_sent,
                        "next_sentence": next_sent,
                        "local_context_window": ctx_window,
                        "claim_summary": claim,
                        "claim_type": claim_type,
                        "citation_role_guess": role,
                        "needs_manual_review": str(review).upper(),
                        "manual_review_reason": ";".join(reasons),
                    })

    # ── Write CSV ────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Terminal summary ─────────────────────────────────────────────────
    sep = '=' * 60
    print(f"\n{sep}")
    print("CITATION AUDIT SUMMARY")
    print(sep)
    print(f"Total citation occurrences:        {stats['total']}")
    print(f"Unique citation keys:              {len(stats['unique_keys'])}")
    print(f"Rows with missing BibTeX entry:    {stats['missing_bib']}")
    print(f"Rows with missing DOI:             {stats['missing_doi']}")
    print(f"Multi-citation cluster commands:   {stats['multi_clusters']}")
    print(f"Rows flagged for manual review:    {stats['needs_review']}")
    print(f"Output written to: {OUTPUT_CSV}")
    print(sep)


if __name__ == "__main__":
    main()
