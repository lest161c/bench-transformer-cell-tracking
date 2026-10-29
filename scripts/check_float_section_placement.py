#!/usr/bin/env python3
"""Check that every float in the compiled report stays within its own section.

SPEC 0036 acceptance check (A1/A2): every figure/table caption must be
typeset on a page that belongs to the top-level section (or appendix section)
in which the float environment is declared. A float on the page where the
*next* section starts counts as a violation, because that page carries the
next section's heading and running head.

How it works:
1. Parse ``main.toc`` for the start page of every numbered section and build
   page ranges ``[start_i, start_{i+1} - 1]`` per top-level section.
2. Walk ``main.tex`` (recursing into ``\\input`` files) in document order,
   tracking the current ``\\section`` and numbering ``figure``/``table``
   environments exactly as LaTeX does (global counters).
3. Extract the page of each printed caption ("Figure N:", "Table N:") from
   the PDF with ``pdftotext``.
4. Report any caption whose page falls outside its owning section's range.

Usage (from ``report/``)::

    python3 scripts/check_float_section_placement.py [main]

The first argument is the PDF basename (default ``main``). Exit status is 0
when no float drifted, 1 otherwise.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_RE = re.compile(r"^\\input\{(.+?)\}", re.M)
SECTION_RE = re.compile(r"^\\section\*?\{(.+?)\}", re.M)
BEGIN_FLOAT_RE = re.compile(r"^\\begin\{(figure|table)\}", re.M)
END_FLOAT_RE = re.compile(r"^\\end\{(figure|table)\}", re.M)
LABEL_RE = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")
CAPTION_RE = re.compile(r"^(Figure|Table)\s+(\d+)\s*:", re.M)


def parse_toc(toc_path):
    """Return the ordered [(number, title, start_page)] of numbered sections.

    Only entries with ``\\numberline`` are returned; starred sections
    (Acknowledgment) are skipped because floats are barred from them anyway.
    """
    entries = []
    for line in toc_path.read_text(encoding="utf-8").splitlines():
        match = re.search(
            r"\\contentsline \{section\}\{\\numberline \{([^}]*)\}([^{}]*)\}\{(\d+)\}",
            line,
        )
        if match:
            entries.append((match.group(1), match.group(2).strip(), int(match.group(3))))
    return entries


def scan_source(main_tex):
    """Return {float_number -> (kind, label, owning_section_title)} in document order.

    Walks ``main.tex`` and recurses into ``\\input`` files, mirroring the
    document order. Float numbering follows the global LaTeX counters
    (figures and tables are numbered independently).
    """
    assignments = {}
    counters = {"figure": 0, "table": 0}
    current_section = None

    def walk(path):
        """Recurse through a LaTeX source file, recording every float.

        Follows `\\input` includes in document order (outside float
        environments), tracks the owning `\\section` heading, and records each
        float's global number, label, and section the first time its `\\label`
        appears, so `assignments` ends up keyed as `Figure<N>`/`Table<N>`.
        """
        nonlocal current_section
        text = path.read_text(encoding="utf-8")
        # Sections outside any float environment set the owner.
        inside_float = False
        pending_label = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            input_match = INPUT_RE.match(line)
            if input_match and not inside_float:
                input_name = input_match.group(1)
                if not input_name.endswith(".tex"):
                    input_name += ".tex"
                walk(path.parent / input_name)
                continue
            if not inside_float and SECTION_RE.match(line):
                current_section = SECTION_RE.match(line).group(1)
            begin_match = BEGIN_FLOAT_RE.match(line)
            if begin_match and not inside_float:
                inside_float = True
                kind = begin_match.group(1)
                counters[kind] += 1
                pending_label = None
                float_kind = kind
                float_section = current_section
            if inside_float and pending_label is None:
                label_match = LABEL_RE.search(line)
                if label_match:
                    pending_label = label_match.group(1)
                    kind_word = "Figure" if float_kind == "figure" else "Table"
                    assignments[f"{kind_word}{counters[float_kind]}"] = (
                        float_kind,
                        pending_label,
                        float_section,
                    )
            if inside_float and END_FLOAT_RE.match(line):
                inside_float = False

    walk(main_tex)
    return assignments


def caption_pages(pdf_base):
    """Return {caption_key -> [pages]} by scanning the PDF text page by page."""
    result = {}
    page_number = 1
    while True:
        proc = subprocess.run(
            ["pdftotext", "-f", str(page_number), "-l", str(page_number),
             f"{pdf_base}.pdf", "-"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            break
        for kind, number in CAPTION_RE.findall(proc.stdout):
            result.setdefault(f"{kind}{number}", []).append(page_number)
        page_number += 1
    return result, page_number - 1


def main():
    """Check that every float in the compiled PDF sits inside its own section.

    Takes the PDF base name as an optional positional argument (default
    "main") and `--report-dir` for the report tree to check (default: the
    live `report-shorten`). Reads the `.toc` for section page ranges, scans
    the LaTeX sources for float labels and their owning sections, locates
    each caption in the PDF text, and reports floats found outside their
    section's page range. Returns 1 when the `.toc` is missing or empty
    (run latexmk first).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_base", nargs="?", default="main")
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "report-shorten"))
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    pdf_base = args.pdf_base
    sections = parse_toc(report_dir / f"{pdf_base}.toc")
    if not sections:
        print("ERROR: no sections found in .toc (run latexmk first)")
        return 1

    # Page range of each top-level section: from its own start page up to
    # (excluding) the next section's start page. A float on the next
    # section's start page would carry that section's heading/running head,
    # so the boundary page belongs to the next section.
    ranges = []
    for index, (number, title, page) in enumerate(sections):
        # When two sections share a start page, the range must not invert:
        # the shared page counts for both sections.
        end = max(page, sections[index + 1][2] - 1) if index + 1 < len(sections) else 10_000
        ranges.append((number, title, page, end))

    assignments = scan_source(report_dir / f"{pdf_base}.tex")
    pages_by_caption, total_pages = caption_pages(report_dir / pdf_base)

    print(f"PDF pages: {total_pages}")
    print(f"{'caption':<10} {'pages':<14} {'owning section'}")
    violations = 0
    for caption in sorted(
        assignments, key=lambda key: (key.startswith("Table"), int(re.sub(r"\D", "", key)))
    ):
        _, _, section_title = assignments[caption]
        printed_pages = pages_by_caption.get(caption, [])
        owner_range = next(
            (r for r in ranges if r[1].strip() == section_title.strip()), None
        )
        if owner_range is None:
            status = "UNRESOLVED SECTION"
            violations += 1
        elif printed_pages and all(
            owner_range[2] <= page <= owner_range[3] for page in printed_pages
        ):
            status = "ok"
        else:
            status = "DRIFTED"
            violations += 1
        marker = "" if status == "ok" else f"  <-- {status}"
        print(
            f"{caption:<10} {str(printed_pages):<14} "
            f"{owner_range[1] if owner_range else section_title[:40]}{marker}"
        )

    if violations:
        print(f"\nFAIL: {violations} float(s) outside their section")
        return 1
    print("\nPASS: all floats are within their section's pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
