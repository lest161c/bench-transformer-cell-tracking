"""HTML report generation for the operator breakdown profiler.

Provides the HTML scaffold, image embedding, and report assembly
functions used by the general and gather profiling modes.
"""

import base64
import io
import os


def file_to_b64(file_path):
    """Read a PNG file and return its base64-encoded string.

    Called by the HTML report builders to embed chart images inline as
    ``data:image/png;base64,...`` so the report is a single self-contained
    file that works offline.
    """
    with open(file_path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode()


def figure_to_b64(figure):
    """Convert a matplotlib Figure to a base64-encoded PNG string.

    Saves the figure to an in-memory buffer and returns the encoded
    bytes — used when figures are generated in memory and don't have a
    file path.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def html_page_head(title, heading, extra_css=""):
    """Build the HTML header scaffold shared by both report builders.

    Returns a list of HTML parts containing the doctype, the dark
    GitHub-style CSS (monospace body, section spacing, image borders),
    the page title, and the top-level heading.  ``extra_css`` is appended
    after the shared rules so a caller can override or extend the style.
    """
    return [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:monospace;max-width:1400px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}",
        "h1{color:#58a6ff} h2{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:4px}",
        "pre{background:#161b22;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px;line-height:1.4}",
        "img{max-width:100%;margin:12px 0;border:1px solid #30363d;border-radius:6px}",
        ".section{margin:32px 0}",
        extra_css,
        "</style></head><body>",
        f"<h1>{heading}</h1>",
    ]


def write_html_report(parts, output_dir, filename):
    """Join HTML parts into a single file inside output_dir.

    Returns the path of the written HTML file.
    """
    html_path = os.path.join(output_dir, filename)
    with open(html_path, "w") as file_handle:
        file_handle.write("\n".join(parts))
    return html_path


def build_general_report(all_results, output_dir):
    """Create the general-mode HTML report with all charts and tables.

    Args:
        all_results: List of profiler result dicts from profile_workload.
        output_dir: Directory to save the HTML report.

    Returns:
        Path to the generated HTML file.
    """
    parts = html_page_head(
        "PyTorch Profiler Report — Attention Benchmark",
        "PyTorch Profiler — Attention Benchmark",
    )

    for i, res in enumerate(all_results):
        parts.append(f"<div class='section'><h2>{i+1}. {res['name']}</h2>")

        charts = res.get("charts", {})

        if "time_bar" in charts:
            parts.append(f"<h3>Time Breakdown</h3>")
            parts.append(f'<img src="data:image/png;base64,{file_to_b64(charts["time_bar"])}" />')

        if "time_selfvtotal" in charts:
            parts.append(f'<img src="data:image/png;base64,{file_to_b64(charts["time_selfvtotal"])}" />')

        if "memory_bar" in charts:
            parts.append(f"<h3>Memory Breakdown</h3>")
            parts.append(f'<img src="data:image/png;base64,{file_to_b64(charts["memory_bar"])}" />')

        if "mem_timeline" in charts:
            parts.append(f"<h3>Memory Timeline</h3>")
            parts.append(f'<img src="data:image/png;base64,{file_to_b64(charts["mem_timeline"])}" />')

        if "calls" in charts:
            parts.append(f"<h3>Call Counts</h3>")
            parts.append(f'<img src="data:image/png;base64,{file_to_b64(charts["calls"])}" />')

        if res.get("time_table"):
            parts.append(f"<h3>Operator Table (Time)</h3><pre>{res['time_table']}</pre>")
        if res.get("mem_table"):
            parts.append(f"<h3>Operator Table (Memory)</h3><pre>{res['mem_table']}</pre>")

        parts.append(f"<p>Chrome trace: <code>{res.get('trace_path', '')}</code></p>")

        parts.append("</div>")

    parts.append("</body></html>")
    return write_html_report(parts, output_dir, "profiler_report.html")


def build_gather_report(all_results, output_dir):
    """Build the gather-mode HTML report with profiler results for all N values.

    Args:
        all_results: List of result dicts from profile_gather_pass.
            Each dict must contain a ``categories`` key mapping category
            names to percentages, pre-computed by
            :func:`benchmark_operator_breakdown.categorize_records`.
        output_dir: Directory to save the HTML report.

    Returns:
        Path to the generated HTML file.
    """
    extra_css = (
        "pre{background:#161b22;padding:12px;border-radius:6px;overflow-x:auto;font-size:11px;line-height:1.3} "
        ".cat{padding:6px;border-radius:4px;margin:4px 0} "
        ".cat-gather{background:#553322} .cat-sdpa{background:#224455} .cat-proj{background:#335533}"
    )
    parts = html_page_head(
        "GatherSparseAttention Profiler Report",
        "GatherSparseAttention Profiler — N=128,256,512 K=16",
        extra_css=extra_css,
    )

    for res in all_results:
        tag = res["tag"]
        parts.append(f"<div class='section'><h2>{tag}</h2>")
        parts.append(f"<p>Total CUDA time: <b>{res['total_cuda_ms']:.4f} ms</b></p>")

        cats = res["categories"]
        parts.append("<h3>Category Breakdown</h3><pre>")
        for cat, pct in cats.items():
            bar = "#" * int(pct / 2)
            parts.append(f"  {cat:<20s} {pct:5.1f}%  {bar}")
        parts.append("</pre>")

        parts.append(f'<img src="data:image/png;base64,{file_to_b64(res["chart_path"])}" />')

        if res.get("table"):
            parts.append(f"<h3>Operator Table</h3><pre>{res['table']}</pre>")

        parts.append(f"<p>Trace: <code>{res.get('trace_path', '')}</code></p>")
        parts.append("</div>")

    parts.append("</body></html>")
    html_path = write_html_report(parts, output_dir, "profile_gather_report.html")
    print(f"\nReport: {html_path}")
    return html_path
