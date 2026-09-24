"""Comprehensive unit tests for LaTeX formatting sanitization and document layout."""

import subprocess
from pathlib import Path
import pytest

from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import _find_pdflatex
from agentic_ml_analog_sim.reporting.latex_sanitizer import (
    convert_markdown_tables_to_latex,
    sanitize_ascii_art,
    sanitize_latex,
    sanitize_latex_document,
    sanitize_latex_text,
)


def test_convert_markdown_tables_to_latex_basic():
    """Verify conversion of standard Markdown table to LaTeX tabular environment with booktabs."""
    md_text = """
Some introductory text.

| Candidate ID | Integration Solver | Latency (ms) | Speedup vs Baseline |
| :--- | :---: | ---: | ---: |
| `ref_rk4_25steps` | Golden RK4 | 3107.55 | 1.00x |
| `cand_euler_10` | First-order Euler | 496.51 | 6.26x |

Some concluding remarks.
"""
    result = convert_markdown_tables_to_latex(md_text)

    assert r"\begin{tabular}{lcrr}" in result
    assert r"\toprule" in result
    assert r"\midrule" in result
    assert r"\bottomrule" in result
    assert r"\end{tabular}" in result
    assert r"\textbf{Candidate ID}" in result
    assert r"\textbf{Integration Solver}" in result
    assert r"\texttt{ref\_rk4\_25steps}" in result
    assert r"\texttt{cand\_euler\_10}" in result
    assert "3107.55 & 1.00x \\\\" in result
    assert "Some introductory text." in result
    assert "Some concluding remarks." in result


def test_convert_markdown_tables_cell_formatting():
    """Verify cell formatting: math symbols, currency, percentages, bold, and escapes."""
    md_text = """
| Phase | Node Role | Calls | Total Tokens | Cost | Token Share |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Phase 0** | `node2_phase0` | 22 | 1,116,633 | $1.84 | 11.41% |
| **Phase 1** | `node3_phase1` | 55 | 7,971,456 | $12.44 | 81.44% |
| **Total** | *All Phases* | **118** | **9,788,256** | **$14.28** | **100.0%** |
"""
    result = convert_markdown_tables_to_latex(md_text)

    assert r"\begin{tabular}{llrrrr}" in result
    assert r"\textbf{Phase 0}" in result
    assert r"\texttt{node2\_phase0}" in result
    assert r"\$1.84" in result
    assert r"\$12.44" in result
    assert r"\$14.28" in result
    assert r"11.41\%" in result
    assert r"81.44\%" in result
    assert r"100.0\%" in result
    assert r"\textit{All Phases}" in result


def test_sanitize_ascii_art_transform():
    """Verify that triple-backtick ASCII diagrams are transformed into LaTeX verbatim."""
    raw_text = """
Here is the diagram:
```
   Latency (ms) [Log Scale]
   3500 |  * ref_rk4 (3107.55 ms)
    500 |            * cand_euler (496.51 ms)
      0 +-------------------------->
```
Next paragraph.
"""
    result = sanitize_ascii_art(raw_text)

    assert "```" not in result
    assert r"\begin{verbatim}" in result
    assert r"\end{verbatim}" in result
    assert "Latency (ms)" in result
    assert "Next paragraph." in result


def test_sanitize_ascii_art_strip():
    """Verify that triple-backtick blocks can be stripped completely."""
    raw_text = """
Before
```
Pipeline Phase Health:
[Solver: 14/14] ====
```
After
"""
    result = sanitize_ascii_art(raw_text, strip=True)

    assert "```" not in result
    assert "Pipeline Phase Health:" not in result
    assert "Before" in result
    assert "After" in result


def test_sanitize_ascii_art_unwraps_embedded_table():
    """Verify that Markdown tables wrapped in triple backticks are unwrapped for table conversion."""
    raw_text = """
```markdown
| Method | Speedup |
| :--- | ---: |
| Euler | 6.26x |
```
"""
    unwrapped = sanitize_ascii_art(raw_text)
    converted = convert_markdown_tables_to_latex(unwrapped)

    assert r"\begin{tabular}{lr}" in converted
    assert r"\textbf{Method}" in converted
    assert "Euler & 6.26x \\\\" in converted


def test_sanitize_latex_full_pipeline():
    """Verify sanitize_latex runs ASCII art, table conversion, and character escaping together."""
    narrative = """
### Optimization Summary
The co-design sweep achieved 16.67% improvement at σ ≤ 0.01 for $14.28 USD.

| Candidate ID | Speedup | Pareto |
| :--- | ---: | :---: |
| `cand_euler_10` | 6.26x | **Yes** |

```
Progress: [==========] 100%
```
"""
    sanitized = sanitize_latex(narrative)

    # Heading converted
    assert r"\subsection{Optimization Summary}" in sanitized
    # Unicode and special characters sanitized
    assert r"$\sigma$" in sanitized
    assert r"$\le$" in sanitized
    assert r"\$14.28 USD" in sanitized
    assert r"16.67\%" in sanitized
    # Table converted
    assert r"\begin{tabular}{lrc}" in sanitized
    assert r"\texttt{cand\_euler\_10}" in sanitized
    assert r"\textbf{Yes}" in sanitized
    # ASCII block converted to verbatim
    assert r"\begin{verbatim}" in sanitized
    assert r"\end{verbatim}" in sanitized
    assert "```" not in sanitized


def test_pdflatex_compilation_of_sanitized_snippet(tmp_path: Path):
    """Verify that pdflatex compiles a document containing converted tables and verbatim blocks with exit code 0."""
    pdflatex_bin = _find_pdflatex()
    if not pdflatex_bin:
        pytest.skip("pdflatex not found on system; skipping compilation verification")

    raw_section = """
### Pareto Assessment
Evaluated with mismatch σ ≤ 0.02 and frequency ω_0 for $12.44 USD:

| Candidate ID | Solver | Latency (ms) | Error | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| `ref_rk4_25steps` | Golden RK4 | 3107.55 | 0.0000 | 1.00x |
| `cand_euler_10` | Euler | 496.51 | 0.0050 | 6.26x |

```
Pipeline Health:
[Rewrites: 14/14 (100%)] ==========
```
"""
    sanitized_body = sanitize_latex(raw_section)

    tex_content = f"""\\documentclass[10pt,conference]{{IEEEtran}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{booktabs}}
\\usepackage{{microtype}}
\\begin{{document}}
\\title{{Test LaTeX Sanitizer}}
\\maketitle
\\section{{Results}}
{sanitized_body}
\\end{{document}}
"""
    out_dir = tmp_path / "test_compile"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_file = out_dir / "test_doc.tex"
    tex_file.write_text(tex_content, encoding="utf-8")

    cmd = [
        pdflatex_bin,
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_file),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    assert res.returncode == 0, f"pdflatex failed with output:\n{res.stdout[-1500:]}"
    assert (out_dir / "test_doc.pdf").exists()
