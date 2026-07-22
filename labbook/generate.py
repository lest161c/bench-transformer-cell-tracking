#!/usr/bin/env python3
"""
Generate labbook/index.html — a dashboard-style labbook viewer.

Reads all .md files, extracts metadata, renders markdown,
and produces a self-contained HTML file with search, filtering,
timeline, tag cloud, and an interactive reading view.

Usage:
    python generate.py              # Build index.html
    python generate.py --serve      # Build + serve via HTTP
"""

import os, re, html, json, sys, collections, math
from pathlib import Path
from datetime import datetime, timedelta

HERE = Path(__file__).resolve().parent
OUT = HERE / "index.html"

try:
    import markdown as _md
    MD_EXTENSIONS = ["markdown.extensions.extra", "markdown.extensions.smarty"]
except ImportError:
    _md = None


def extract_date(name):
    m = re.match(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else "9999-99-99"


def extract_tags(name, text):
    """Derive tags from filename and content keywords."""
    stem = name.replace(".md", "")
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)
    parts = re.split(r"[_\-]", stem)
    stopwords = {"a", "an", "the", "and", "or", "of", "in", "to", "for",
                 "with", "on", "at", "by", "from", "is", "was", "are",
                 "we", "our", "per", "via", "not", "no", "vs"}
    tags = set()
    for p in parts:
        p = p.strip().lower()
        p = re.sub(r"[^a-z0-9]", "", p)
        if p and p not in stopwords:
            tags.add(p)

    # Extract hashtags from content
    for m in re.finditer(r"#([A-Za-z][A-Za-z0-9_]+)", text):
        tags.add(m.group(1).lower())

    # Extract capitalized multi-word phrases as potential tags
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text):
        tags.add(m.group(0).lower())

    return sorted(tags)


def extract_excerpt(text, max_words=40):
    """Extract first meaningful paragraph as excerpt."""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---") and not line.startswith("```"):
            words = line.split()
            if len(words) > 3:
                if len(words) <= max_words:
                    return line
                return " ".join(words[:max_words]) + "..."
    return ""


_ULINE_PLACEHOLDER = "⁅ULINE⁆"

def _protect_latex(text):
    # Protect \(...\) inline math: replace delimiters AND underscores inside
    text = re.sub(
        r"\\\((.+?)\\\)",
        lambda m: "⁅LIO⁆" + m.group(1).replace("_", _ULINE_PLACEHOLDER) + "⁅LIC⁆",
        text, flags=re.DOTALL
    )
    # Protect \[...\] display math
    text = re.sub(
        r"\\\[(.+?)\\\]",
        lambda m: "⁅LDO⁆" + m.group(1).replace("_", _ULINE_PLACEHOLDER) + "⁅LDC⁆",
        text, flags=re.DOTALL
    )
    return text

def _restore_latex(text):
    text = text.replace(_ULINE_PLACEHOLDER, "_")
    text = text.replace("⁅LIO⁆", "\\(").replace("⁅LIC⁆", "\\)")
    text = text.replace("⁅LDO⁆", "\\[").replace("⁅LDC⁆", "\\]")
    return text

def render_md(text):
    text = _protect_latex(text)
    if _md is not None:
        html = _md.markdown(text, extensions=MD_EXTENSIONS)
        return _restore_latex(html)

    # Fallback basic renderer (see prior implementation)
    lines = text.split("\n")
    out = []
    i = 0
    in_code = False
    code_buf = []

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return t

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                out.append(f"<pre><code>{html.escape(chr(10).join(code_buf))}</code></pre>")
                code_buf = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        hm = re.match(r"^(#{1,6})\s+(.+)$", line)
        if hm:
            out.append(f"<h{len(hm.group(1))}>{hm.group(2)}</h{len(hm.group(1))}>")
            i += 1
            continue

        if re.match(r"^---+$", line):
            out.append("<hr>")
            i += 1
            continue

        if line.strip() == "":
            out.append("")
            i += 1
            continue

        # Table
        if "|" in line and line.strip().startswith("|"):
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            tbl_rows = []
            for ri, r in enumerate(rows):
                if ri >= 1 and re.match(r"^[\s|:\-]+$", rows[ri - 1]):
                    continue
                cells = [c.strip() for c in r.split("|")[1:-1]]
                tag = "th" if ri == 0 else "td"
                tbl_rows.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            out.append("<table>" + "".join(tbl_rows) + "</table>")
            continue

        # List
        lm = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if lm:
            items = []
            indent = len(lm.group(1))
            while i < len(lines):
                m = re.match(r"^(\s*)[-*+]\s+(.*)", lines[i])
                if not m or len(m.group(1)) != indent:
                    break
                items.append(f"<li>{inline(m.group(2))}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue

        # Ordered list
        om = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if om:
            items = []
            indent = len(om.group(1))
            while i < len(lines):
                m = re.match(r"^(\s*)\d+\.\s+(.*)", lines[i])
                if not m or len(m.group(1)) != indent:
                    break
                items.append(f"<li>{inline(m.group(2))}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue

        # Blockquote
        if line.startswith(">"):
            bq_lines = []
            while i < len(lines) and lines[i].startswith(">"):
                bq_lines.append(lines[i][1:].strip())
                i += 1
            inner = render_md("\n".join(bq_lines))
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # Paragraph
        para = []
        while i < len(lines):
            l = lines[i]
            if l.strip() == "" or l.startswith("#") or l.startswith("```") or l.startswith("---") or \
               l.strip().startswith("|") or l.strip().startswith("- ") or l.strip().startswith("* ") or \
               re.match(r"^\s*\d+\.\s+", l) or l.startswith(">"):
                break
            para.append(inline(l))
            i += 1
        if para:
            out.append(f"<p>{' '.join(para)}</p>")
        if not para:
            i += 1

    return _restore_latex("\n".join(out))


def build_data():
    files = sorted(HERE.glob("*.md"), key=lambda f: extract_date(f.name))
    if not files:
        print("No .md files found.")
        return []

    entries = []
    for f in files:
        if f.name == "index.md" or f.name == "generate.py":
            continue
        text = f.read_text(encoding="utf-8")
        date = extract_date(f.name)
        title = ""
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            stem = f.name.replace(".md", "")
            stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)
            title = stem.replace("_", " ").title()

        entries.append({
            "id": f.stem,
            "date": date,
            "title": title,
            "tags": extract_tags(f.name, text),
            "excerpt": extract_excerpt(text),
            "html": render_md(text),
            "wordCount": len(text.split()),
        })
    return entries


def build_html(entries):
    data_json = json.dumps(entries, ensure_ascii=False)

    # Compute dashboard stats
    dates = [e["date"] for e in entries if e["date"] != "9999-99-99"]
    date_range = f"{dates[0]} — {dates[-1]}" if len(dates) >= 2 else dates[0] if dates else ""
    total_words = sum(e["wordCount"] for e in entries)

    # Timeline data: entries per week
    timeline = []
    if dates:
        d_objs = sorted(datetime.strptime(d, "%Y-%m-%d") for d in dates)
        start = d_objs[0] - timedelta(days=d_objs[0].weekday())
        end = d_objs[-1] + timedelta(days=6 - d_objs[-1].weekday())
        current = start
        while current <= end:
            week_start = current
            week_end = current + timedelta(days=6)
            count = sum(1 for d in d_objs if week_start <= d <= week_end)
            timeline.append({
                "week": week_start.strftime("%Y-%m-%d"),
                "count": count,
            })
            current += timedelta(days=7)
    max_count = max((t["count"] for t in timeline), default=1)

    # Tag cloud
    all_tags = collections.Counter()
    for e in entries:
        for t in e["tags"]:
            all_tags[t] += 1

    # Recent entries HTML (pre-computed to avoid nested f-string issues)
    recent_html = ""
    for e in entries[-5:]:
        tag_html = "".join(f'<span class="tag">{t}</span>' for t in e["tags"][:3])
        recent_html += (
            f'<div class="recent-item" data-id="{e["id"]}">'
            f'<span class="rdate">{e["date"]}</span>'
            f'<span class="rtitle">{html.escape(e["title"])}</span>'
            f'<span class="rtags">{tag_html}</span>'
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Labbook Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
/* ============ RESET & BASE ============ */
:root {{
  --bg: #f5f5f7;
  --fg: #1d1d1f;
  --surface: #ffffff;
  --surface2: #f0f0f2;
  --border: #d2d2d7;
  --accent: #2563eb;
  --accent2: #3b82f6;
  --accent-light: #eff6ff;
  --code-bg: #f4f4f5;
  --code-fg: #1f2937;
  --tag-bg: #e8e8ed;
  --tag-fg: #555;
  --muted: #86868b;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-lg: 0 4px 16px rgba(0,0,0,0.1);
  --radius: 10px;
  --radius-sm: 6px;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0a0a0f;
    --fg: #e5e5e7;
    --surface: #15151c;
    --surface2: #1c1c25;
    --border: #2c2c35;
    --accent: #60a5fa;
    --accent2: #3b82f6;
    --accent-light: #0f1729;
    --code-bg: #1a1a24;
    --code-fg: #e5e5e7;
    --tag-bg: #252530;
    --tag-fg: #999;
    --muted: #6b6b72;
    --shadow: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.4);
  }}
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.6;
  padding: 0; margin: 0;
}}

/* ============ APP SHELL ============ */
#app {{ 
  display: flex; flex-direction: column; min-height: 100vh; 
  max-width: 1200px; margin: 0 auto; padding: 0 1.5rem;
}}

/* ============ HEADER ============ */
.app-header {{
  padding: 1.5rem 0 0.75rem 0; margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem;
}}
.app-header h1 {{
  font-size: 1.4rem; font-weight: 700; margin-right: auto;
  background: linear-gradient(135deg, var(--accent), #8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.app-header .subtitle {{ font-size: 0.8rem; color: var(--muted); }}

/* ============ TOOLBAR ============ */
.toolbar {{
  display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
  padding: 0.75rem 0; margin-bottom: 1rem;
}}
.search-box {{
  flex: 1 1 260px; position: relative;
}}
.search-box input {{
  width: 100%; padding: 0.55rem 0.75rem 0.55rem 2.2rem;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--surface); color: var(--fg); font-size: 0.9rem;
  outline: none; transition: border-color 0.2s;
}}
.search-box input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }}
.search-box .icon {{
  position: absolute; left: 0.65rem; top: 50%; transform: translateY(-50%);
  color: var(--muted); font-size: 0.9rem; pointer-events: none;
}}
.filter-group {{
  display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap;
}}
.filter-group label {{ font-size: 0.78rem; color: var(--muted); white-space: nowrap; }}
.filter-group input[type="date"] {{
  padding: 0.4rem 0.5rem; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface); color: var(--fg);
  font-size: 0.8rem; outline: none;
}}
.filter-group input[type="date"]:focus {{ border-color: var(--accent); }}
.btn {{
  padding: 0.45rem 0.85rem; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface); color: var(--fg);
  cursor: pointer; font-size: 0.82rem; transition: all 0.15s;
}}
.btn:hover {{ background: var(--surface2); }}
.btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.btn-icon {{ padding: 0.45rem 0.6rem; }}

/* ============ VIEW SWITCHER ============ */
.view-tabs {{
  display: flex; gap: 0; margin-bottom: 1rem;
  border-bottom: 2px solid var(--border);
}}
.view-tab {{
  padding: 0.5rem 1.2rem; cursor: pointer; font-size: 0.85rem;
  font-weight: 500; color: var(--muted); border-bottom: 2px solid transparent;
  margin-bottom: -2px; transition: all 0.2s;
}}
.view-tab:hover {{ color: var(--fg); }}
.view-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.view-tab .count {{
  display: inline-block; margin-left: 0.35rem; padding: 0.05rem 0.45rem;
  font-size: 0.7rem; background: var(--tag-bg); border-radius: 8px;
  color: var(--muted);
}}

/* ============ DASHBOARD ============ */
#view-dashboard {{ display: block; }}
#view-entries {{ display: none; }}
#view-timeline {{ display: none; }}

.dashboard-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem; margin-bottom: 1.5rem;
}}
.stat-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1rem;
  box-shadow: var(--shadow);
}}
.stat-card .label {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
.stat-card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }}

.dashboard-section {{ margin-bottom: 1.5rem; }}
.dashboard-section h2 {{ 
  font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em;
}}

/* Timeline bars */
.timeline {{ display: flex; align-items: flex-end; gap: 2px; height: 120px; padding: 0.5rem 0; }}
.timeline-bar {{
  flex: 1; min-width: 6px; border-radius: 2px 2px 0 0;
  background: var(--accent); opacity: 0.7; transition: opacity 0.2s;
  position: relative; cursor: pointer;
}}
.timeline-bar:hover {{ opacity: 1; }}
.timeline-bar .tooltip {{
  display: none; position: absolute; bottom: 100%; left: 50%;
  transform: translateX(-50%); background: var(--surface);
  border: 1px solid var(--border); padding: 0.25rem 0.5rem;
  border-radius: 4px; font-size: 0.7rem; white-space: nowrap;
  z-index: 10; box-shadow: var(--shadow);
}}
.timeline-bar:hover .tooltip {{ display: block; }}

/* Tag cloud */
.tag-cloud {{
  display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
}}
.tag {{
  display: inline-block; padding: 0.2rem 0.55rem;
  background: var(--tag-bg); color: var(--tag-fg);
  border-radius: 12px; font-size: 0.78rem; cursor: pointer;
  transition: all 0.15s; line-height: 1.5;
}}
.tag:hover {{ background: var(--accent); color: #fff; }}
.tag.active {{ background: var(--accent); color: #fff; }}
.tag-s1 {{ font-size: 0.72rem; }}
.tag-s2 {{ font-size: 0.78rem; }}
.tag-s3 {{ font-size: 0.85rem; }}
.tag-s4 {{ font-size: 0.92rem; }}
.tag-s5 {{ font-size: 1rem; font-weight: 600; }}

/* Recent entries */
.recent-list {{
  display: flex; flex-direction: column; gap: 0.4rem;
}}
.recent-item {{
  display: flex; gap: 0.75rem; align-items: baseline;
  padding: 0.5rem 0.75rem; background: var(--surface);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  cursor: pointer; transition: all 0.15s;
}}
.recent-item:hover {{ border-color: var(--accent); box-shadow: var(--shadow); }}
.recent-item .rdate {{ font-size: 0.78rem; color: var(--muted); white-space: nowrap; }}
.recent-item .rtitle {{ font-size: 0.9rem; }}
.recent-item .rtags {{ display: flex; gap: 0.25rem; margin-left: auto; }}

/* ============ TIMELINE VIEW ============ */
.tl-wrap {{
  position: relative; padding: 0.5rem 0 1rem 2rem;
}}
.tl-wrap::before {{
  content: ''; position: absolute; left: 8px; top: 0; bottom: 0;
  width: 2px; background: var(--border);
}}
.tl-month {{
  position: relative; font-size: 0.78rem; font-weight: 600;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  padding: 0.75rem 0 0.5rem 1rem; margin-top: 0.5rem;
}}
.tl-month::before {{
  content: ''; position: absolute; left: -1.5rem; top: 50%;
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--surface2); border: 2px solid var(--border);
  transform: translateY(-50%);
}}
.tl-month:first-child {{ margin-top: 0; }}
.tl-item {{
  position: relative; margin-bottom: 0.6rem;
}}
.tl-item::before {{
  content: ''; position: absolute; left: -1.5rem; top: 1rem;
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--surface);
  box-shadow: 0 0 0 2px var(--accent); z-index: 1;
}}
.tl-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 0.75rem 1rem;
  cursor: pointer; transition: all 0.15s;
  box-shadow: var(--shadow);
}}
.tl-card:hover {{ border-color: var(--accent); box-shadow: var(--shadow-lg); }}
.tl-card-header {{
  display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap;
}}
.tl-date {{
  font-size: 0.75rem; font-weight: 600; color: var(--accent);
  white-space: nowrap; letter-spacing: 0.02em;
}}
.tl-title {{
  font-size: 0.95rem; font-weight: 600; flex: 1; line-height: 1.3;
}}
.tl-tags {{
  display: flex; gap: 0.25rem; flex-wrap: wrap;
}}
.tl-tags .tag {{ font-size: 0.68rem; cursor: pointer; }}
.tl-excerpt {{
  font-size: 0.83rem; color: var(--muted); margin-top: 0.35rem;
  line-height: 1.45; display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden;
}}
.tl-card.open .tl-excerpt {{ -webkit-line-clamp: unset; overflow: visible; }}
.tl-body {{
  display: none; padding-top: 0.75rem; border-top: 1px solid var(--border);
  margin-top: 0.6rem;
}}
.tl-card.open .tl-body {{ display: block; }}

/* ============ ENTRIES VIEW ============ */
.controls-bar {{
  display: flex; align-items: center; gap: 0.5rem;
  margin-bottom: 0.75rem; font-size: 0.82rem; color: var(--muted);
}}
.controls-bar .spacer {{ flex: 1; }}

.entry-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); margin-bottom: 0.6rem;
  box-shadow: var(--shadow); transition: box-shadow 0.2s;
  overflow: hidden;
}}
.entry-card:hover {{ box-shadow: var(--shadow-lg); }}
.entry-card-header {{
  padding: 0.75rem 1rem; cursor: pointer;
  display: flex; align-items: flex-start; gap: 0.75rem;
}}
.entry-card-header .edate {{
  font-size: 0.78rem; color: var(--muted); white-space: nowrap;
  padding-top: 0.1rem; min-width: 80px;
}}
.entry-card-header .etitle {{
  font-size: 1rem; font-weight: 600; line-height: 1.3; flex: 1;
}}
.entry-card-header .expand-icon {{
  color: var(--muted); font-size: 0.8rem; transition: transform 0.2s;
  padding-top: 0.15rem;
}}
.entry-card.open .expand-icon {{ transform: rotate(90deg); }}
.entry-card-body {{
  display: none; padding: 0 1rem 1rem 1rem;
  border-top: 1px solid var(--border);
}}
.entry-card.open .entry-card-body {{ display: block; }}
.entry-tags {{
  display: flex; gap: 0.25rem; flex-wrap: wrap; margin-top: 0.25rem;
}}
.entry-tags .tag {{ font-size: 0.7rem; }}

.entry-card-body .entry-content {{
  padding-top: 1rem;
}}
.entry-content > * {{ margin-bottom: 0.75rem; }}
.entry-content h2 {{
  font-size: 1.15rem; margin-top: 1.25rem; margin-bottom: 0.5rem;
  padding-bottom: 0.2rem; border-bottom: 1px solid var(--border);
}}
.entry-content h3 {{ font-size: 1.05rem; margin-top: 1rem; margin-bottom: 0.35rem; }}
.entry-content h4 {{ font-size: 1rem; margin-top: 0.75rem; margin-bottom: 0.25rem; }}
.entry-content p {{ margin-bottom: 0.6rem; }}
.entry-content strong {{ font-weight: 600; }}
.entry-content em {{ font-style: italic; }}
.entry-content ul, .entry-content ol {{ margin: 0.3rem 0 0.5rem 1.35rem; }}
.entry-content li {{ margin-bottom: 0.15rem; }}
.entry-content blockquote {{
  margin: 0.6rem 0; padding: 0.4rem 0.75rem;
  border-left: 3px solid var(--accent); background: var(--surface2);
  border-radius: 0 4px 4px 0;
}}
.entry-content blockquote p {{ margin-bottom: 0.25rem; }}
.entry-content hr {{ margin: 1rem 0; border: none; border-top: 2px solid var(--border); }}
.entry-content code {{
  font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
  font-size: 0.85em; padding: 0.15em 0.3em;
  background: var(--code-bg); color: var(--code-fg); border-radius: 3px;
}}
.entry-content pre {{
  margin: 0.6rem 0; padding: 0.85rem; overflow-x: auto;
  background: var(--code-bg); border-radius: var(--radius-sm);
  font-size: 0.82rem; line-height: 1.45; border: 1px solid var(--border);
}}
.entry-content pre code {{ padding: 0; background: none; border-radius: 0; font-size: inherit; }}
.entry-content table {{
  width: 100%; border-collapse: collapse; margin: 0.6rem 0;
  font-size: 0.85rem; display: block; overflow-x: auto;
}}
.entry-content th, .entry-content td {{
  padding: 0.35rem 0.6rem; text-align: left;
  border: 1px solid var(--border); white-space: nowrap;
}}
.entry-content th {{ background: var(--surface2); font-weight: 600; }}
.entry-content tr:nth-child(even) {{ background: var(--surface2); }}
.entry-content img {{ max-width: 100%; height: auto; border-radius: 4px; }}
.entry-content a {{ color: var(--accent); text-decoration: none; }}
.entry-content a:hover {{ text-decoration: underline; }}

/* Search highlight */
mark {{
  background: #fde68a; color: #1f2937; padding: 0 0.1em; border-radius: 2px;
}}
@media (prefers-color-scheme: dark) {{
  mark {{ background: #92400e; color: #fef3c7; }}
}}

/* No results */
.no-results {{
  text-align: center; padding: 3rem 1rem; color: var(--muted);
}}

/* ============ SCROLL TO TOP ============ */
.scroll-top {{
  position: fixed; bottom: 1.5rem; right: 1.5rem; width: 40px; height: 40px;
  border-radius: 50%; background: var(--accent); color: #fff;
  border: none; cursor: pointer; font-size: 1.2rem;
  box-shadow: var(--shadow-lg); opacity: 0; pointer-events: none;
  transition: opacity 0.3s; z-index: 100;
}}
.scroll-top.visible {{ opacity: 1; pointer-events: auto; }}
.scroll-top:hover {{ transform: scale(1.05); }}

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {{
  #app {{ padding: 0 0.75rem; }}
  .toolbar {{ flex-direction: column; align-items: stretch; }}
  .search-box {{ flex: auto; }}
  .entry-card-header {{ flex-wrap: wrap; }}
  .entry-card-header .edate {{ min-width: auto; }}
  .dashboard-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>
<div id="app">
  <header class="app-header">
    <h1>Labbook</h1>
    <span class="subtitle">{date_range} &middot; {len(entries)} entries &middot; {total_words:,} words</span>
  </header>

  <div class="toolbar">
    <div class="search-box">
      <span class="icon">&#x1F50D;</span>
      <input type="text" id="searchInput" placeholder="Search entries..." autocomplete="off">
    </div>
    <div class="filter-group">
      <label>From</label>
      <input type="date" id="dateFrom">
      <label>To</label>
      <input type="date" id="dateTo">
    </div>
    <div class="filter-group">
      <button class="btn btn-icon" id="sortBtn" title="Toggle sort order">&#x2193; Date</button>
    </div>
  </div>

  <div class="view-tabs">
    <div class="view-tab active" data-view="dashboard">Dashboard</div>
    <div class="view-tab" data-view="timeline">Timeline</div>
    <div class="view-tab" data-view="entries">Entries <span class="count">{len(entries)}</span></div>
  </div>

  <div id="view-dashboard">
    <div class="dashboard-grid">
      <div class="stat-card">
        <div class="label">Total Entries</div>
        <div class="value">{len(entries)}</div>
      </div>
      <div class="stat-card">
        <div class="label">Date Range</div>
        <div class="value" style="font-size:1rem;">{date_range}</div>
      </div>
      <div class="stat-card">
        <div class="label">Total Words</div>
        <div class="value">{total_words:,}</div>
      </div>
      <div class="stat-card">
        <div class="label">Avg Words / Entry</div>
        <div class="value">{total_words // max(len(entries), 1):,}</div>
      </div>
    </div>

    <div class="dashboard-section">
      <h2>Activity Timeline</h2>
      <div class="timeline" id="timeline">
        {''.join(
          f'<div class="timeline-bar" style="height:{max(4, t["count"] / max_count * 100)}%" '
          f'title="{t["week"]}: {t["count"]} entries">'
          f'<span class="tooltip">Week of {t["week"]}: {t["count"]} entries</span></div>'
          for t in timeline
        )}
      </div>
    </div>

    <div class="dashboard-section">
      <h2>Topics</h2>
      <div class="tag-cloud" id="tagCloud">
        {''.join(
          f'<span class="tag tag-s{min(5, max(1, freq // max(1, max(all_tags.values()) // 5) + 1))}" '
          f'data-tag="{tag}">{tag}</span>'
          for tag, freq in all_tags.most_common(40)
        )}
      </div>
    </div>

    <div class="dashboard-section">
      <h2>Recent Entries</h2>
      <div class="recent-list" id="recentList">
        {recent_html}
      </div>
    </div>
  </div>

  <div id="view-entries">
    <div class="controls-bar">
      <span id="resultCount">{len(entries)} entries</span>
      <span class="spacer"></span>
      <span id="activeTags"></span>
    </div>
    <div id="entryList"></div>
  </div>

  <div id="view-timeline">
    <div class="controls-bar">
      <span id="tlResultCount">{len(entries)} entries</span>
      <span class="spacer"></span>
      <span id="tlActiveTags"></span>
    </div>
    <div class="tl-wrap" id="timelineContainer"></div>
  </div>
</div>

<button class="scroll-top" id="scrollTop" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">&#x2191;</button>

<script>
// ===================== KATEX RENDER =====================
const KATEX_DELIMS = [{{left:'\\(',right:'\\)',display:false}},{{left:'\\[',right:'\\]',display:true}}];
function renderKaTeX(el) {{
  if (typeof renderMathInElement === 'function') {{
    try {{ renderMathInElement(el, {{delimiters:KATEX_DELIMS}}); }} catch(e) {{}}
  }}
}}

// ===================== DATA =====================
const DATA = {data_json};

// ===================== STATE =====================
let state = {{
  view: 'dashboard',
  query: '',
  sortAsc: false,
  dateFrom: '',
  dateTo: '',
  activeTag: null,
  openEntries: new Set(),
}};

// ===================== UTILITY =====================
function escapeHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

function highlightText(text, query) {{
  if (!query) return text;
  const parts = query.split(/\\s+/).filter(Boolean);
  let result = text;
  for (const p of parts) {{
    const escaped = p.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
    result = result.replace(new RegExp(escaped, 'gi'), m => `<mark>${{m}}</mark>`);
  }}
  return result;
}}

function stripHtml(html) {{
  const d = document.createElement('div');
  d.innerHTML = html;
  return d.textContent || '';
}}

// ===================== FILTERING =====================
function getFiltered() {{
  let items = [...DATA];

  // Search
  if (state.query) {{
    const q = state.query.toLowerCase();
    items = items.filter(e =>
      e.title.toLowerCase().includes(q) ||
      stripHtml(e.html).toLowerCase().includes(q) ||
      e.tags.some(t => t.includes(q))
    );
  }}

  // Date range
  if (state.dateFrom) {{
    items = items.filter(e => e.date >= state.dateFrom);
  }}
  if (state.dateTo) {{
    items = items.filter(e => e.date <= state.dateTo);
  }}

  // Tag filter
  if (state.activeTag) {{
    items = items.filter(e => e.tags.includes(state.activeTag));
  }}

  // Sort
  items.sort((a, b) => {{
    const cmp = a.date.localeCompare(b.date);
    return state.sortAsc ? cmp : -cmp;
  }});

  return items;
}}

// ===================== RENDER ENTRIES =====================
function renderEntries() {{
  const items = getFiltered();
  const container = document.getElementById('entryList');
  const countEl = document.getElementById('resultCount');

  const tagsEl = document.getElementById('activeTags');
  if (state.activeTag) {{
    tagsEl.innerHTML = `<span class="tag active" style="cursor:pointer" onclick="clearTag()">${{state.activeTag}} &times;</span>`;
  }} else {{
    tagsEl.innerHTML = '';
  }}

  countEl.textContent = `${{items.length}} of ${{DATA.length}} entries`;

  if (items.length === 0) {{
    container.innerHTML = '<div class="no-results">No entries match your filters</div>';
    return;
  }}

  container.innerHTML = items.map(e => {{
    const isOpen = state.openEntries.has(e.id);
    const excerptHtml = isOpen ? e.html : highlightText(escapeHtml(e.excerpt), state.query);
    return `
      <div class="entry-card ${{isOpen ? 'open' : ''}}" data-id="${{e.id}}">
        <div class="entry-card-header" onclick="toggleEntry('${{e.id}}')">
          <span class="edate">${{e.date}}</span>
          <span class="etitle">${{isOpen ? escapeHtml(e.title) : highlightText(escapeHtml(e.title), state.query)}}</span>
          <span class="entry-tags">${{e.tags.slice(0, 4).map(t => `<span class="tag" onclick="event.stopPropagation();filterTag('${{t}}')">${{t}}</span>`).join('')}}</span>
          <span class="expand-icon">&#x25B6;</span>
        </div>
        <div class="entry-card-body">
          <div class="entry-content">${{isOpen ? e.html : `<p>${{excerptHtml}}</p>`}}</div>
        </div>
      </div>
    `;
  }}).join('');
  renderKaTeX(container);
}}

// ===================== RENDER TIMELINE =====================
function renderTimeline() {{
  const items = getFiltered();
  const container = document.getElementById('timelineContainer');
  const countEl = document.getElementById('tlResultCount');
  const tagsEl = document.getElementById('tlActiveTags');

  if (state.activeTag) {{
    tagsEl.innerHTML = `<span class="tag active" style="cursor:pointer" onclick="clearTag()">${{state.activeTag}} &times;</span>`;
  }} else {{
    tagsEl.innerHTML = '';
  }}
  countEl.textContent = `${{items.length}} of ${{DATA.length}} entries`;

  if (items.length === 0) {{
    container.innerHTML = '<div class="no-results">No entries match your filters</div>';
    return;
  }}

  // Group by year-month
  const groups = [];
  let currentGroup = null;
  for (const e of items) {{
    const ym = e.date.slice(0, 7);
    if (!currentGroup || currentGroup.ym !== ym) {{
      currentGroup = {{ ym, label: formatMonth(ym), entries: [] }};
      groups.push(currentGroup);
    }}
    currentGroup.entries.push(e);
  }}

  container.innerHTML = groups.map(g => `
    <div class="tl-month">${{g.label}}</div>
    ${{g.entries.map(e => {{
      const isOpen = state.openEntries.has(e.id);
      return `
        <div class="tl-item">
          <div class="tl-card ${{isOpen ? 'open' : ''}}" onclick="toggleTimelineEntry('${{e.id}}')">
            <div class="tl-card-header">
              <span class="tl-date">${{e.date}}</span>
              <span class="tl-title">${{highlightText(escapeHtml(e.title), state.query)}}</span>
              <span class="tl-tags">${{e.tags.slice(0, 3).map(t => `<span class="tag" onclick="event.stopPropagation();filterTag('${{t}}')">${{t}}</span>`).join('')}}</span>
            </div>
            <div class="tl-excerpt">${{isOpen ? '' : highlightText(escapeHtml(e.excerpt), state.query)}}</div>
            <div class="tl-body">${{isOpen ? e.html : ''}}</div>
          </div>
        </div>
      `;
    }}).join('')}}
  `).join('');
  renderKaTeX(container);
}}

function formatMonth(ym) {{
  const d = new Date(ym + '-01');
  return d.toLocaleDateString('en-US', {{ year: 'numeric', month: 'long' }});
}}

function toggleTimelineEntry(id) {{
  if (state.openEntries.has(id)) {{
    state.openEntries.delete(id);
  }} else {{
    state.openEntries.add(id);
  }}
  renderTimeline();
}}

// ===================== TOGGLE ENTRY =====================
function toggleEntry(id) {{
  if (state.openEntries.has(id)) {{
    state.openEntries.delete(id);
  }} else {{
    state.openEntries.add(id);
  }}
  renderEntries();
}}

// ===================== TAG FILTERING =====================
function switchView(view) {{
  state.view = view;
  document.querySelectorAll('.view-tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  document.getElementById('view-dashboard').style.display = view === 'dashboard' ? 'block' : 'none';
  document.getElementById('view-entries').style.display = view === 'entries' ? 'block' : 'none';
  document.getElementById('view-timeline').style.display = view === 'timeline' ? 'block' : 'none';
  if (view === 'entries') renderEntries();
  if (view === 'timeline') renderTimeline();
}}

function filterTag(tag) {{
  state.activeTag = state.activeTag === tag ? null : tag;
  switchView('entries');
}}

function clearTag() {{
  state.activeTag = null;
  renderEntries();
}}

// ===================== VIEW SWITCHING =====================
document.querySelectorAll('.view-tab').forEach(tab => {{
  tab.addEventListener('click', () => switchView(tab.dataset.view));
}});

// ===================== RERENDER =====================
function rerender() {{
  if (state.view === 'entries') renderEntries();
  if (state.view === 'timeline') renderTimeline();
}}

// ===================== SEARCH =====================
document.getElementById('searchInput').addEventListener('input', e => {{
  state.query = e.target.value;
  rerender();
}});

// ===================== DATE FILTER =====================
document.getElementById('dateFrom').addEventListener('change', e => {{
  state.dateFrom = e.target.value;
  rerender();
}});
document.getElementById('dateTo').addEventListener('change', e => {{
  state.dateTo = e.target.value;
  rerender();
}});

// ===================== SORT =====================
document.getElementById('sortBtn').addEventListener('click', e => {{
  state.sortAsc = !state.sortAsc;
  e.target.innerHTML = state.sortAsc ? '&#x2191; Date' : '&#x2193; Date';
  rerender();
}});

// ===================== TAG CLOUD =====================
document.querySelectorAll('#tagCloud .tag').forEach(el => {{
  el.addEventListener('click', () => filterTag(el.dataset.tag));
}});

// ===================== RECENT ENTRIES =====================
document.querySelectorAll('#recentList .recent-item').forEach(el => {{
  el.addEventListener('click', () => {{
    const id = el.dataset.id;
    state.openEntries.add(id);
    switchView('entries');
    setTimeout(() => {{
      document.querySelector(`[data-id="${{id}}"]`)?.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}, 50);
  }});
}});

// ===================== SCROLL TO TOP =====================
window.addEventListener('scroll', () => {{
  document.getElementById('scrollTop').classList.toggle('visible', window.scrollY > 300);
}});

// ===================== KEYBOARD SHORTCUTS =====================
document.addEventListener('keydown', e => {{
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
    e.preventDefault();
    document.getElementById('searchInput').focus();
  }}
  if (e.key === 'Escape') {{
    document.getElementById('searchInput').blur();
  }}
}});

// ===================== INIT =====================
// Set date filter defaults
if (DATA.length > 0) {{
  document.getElementById('dateFrom').value = DATA[0].date;
  document.getElementById('dateTo').value = DATA[DATA.length - 1].date;
  state.dateFrom = DATA[0].date;
  state.dateTo = DATA[DATA.length - 1].date;
}}
// Render math in initially visible content (dashboard)
setTimeout(() => renderKaTeX(document.getElementById('view-dashboard')), 500);
</script>
</body>
</html>"""


def build():
    entries = build_data()
    if not entries:
        print("No entries found.")
        return
    html_out = build_html(entries)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"Built {OUT} ({len(entries)} entries, {OUT.stat().st_size / 1024:.0f} KB)")
    # Auto-open browser (unless --no-open is set)
    if "--no-open" not in sys.argv:
        try:
            import webbrowser
            webbrowser.open(OUT.resolve().as_uri())
        except Exception:
            pass


def serve(port=8080):
    import http.server
    os.chdir(HERE)
    server = http.server.HTTPServer(("", port), http.server.SimpleHTTPRequestHandler)
    print(f"Serving at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build()
        serve()
    else:
        build()
