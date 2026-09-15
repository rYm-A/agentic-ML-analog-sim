"""Unit tests for LaTeX narrative sanitization and compilation."""

import subprocess
from pathlib import Path
import pytest

from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import _find_pdflatex
from agentic_ml_analog_sim.reporting.latex_sanitizer import (
    sanitize_latex_document,
    sanitize_latex_text,
)


def test_sanitize_latex_text_unicode_symbols():
    raw_input = "Mismatch σ ≤ 0.02, frequency ω_0, phase θ ∈ [0, 2π], steps N = 10×, drift ±0.01, sum ∑_j K_ij."
    sanitized = sanitize_latex_text(raw_input)

    assert r"$\sigma$" in sanitized
    assert r"$\le$" in sanitized
    assert r"$\omega$" in sanitized
    assert r"$\theta$" in sanitized
    assert r"$\in$" in sanitized
    assert r"$\pi$" in sanitized
    assert r"$\times$" in sanitized
    assert r"$\pm$" in sanitized
    assert r"$\sum$" in sanitized
    # No raw unicode characters remaining
    for raw_char in ["σ", "≤", "ω", "θ", "∈", "π", "×", "±", "∑"]:
        assert raw_char not in sanitized


def test_sanitize_latex_text_special_characters_and_currency():
    raw_input = "Speedup of 16.67% at max_degree = 64 across Solver & Timing for $0.503 USD and O(N^2)."
    sanitized = sanitize_latex_text(raw_input)

    assert r"16.67\%" in sanitized
    assert r"max\_degree" in sanitized
    assert r"Solver \& Timing" in sanitized
    assert r"\$0.503 USD" in sanitized
    assert r"\textsuperscript{2}" in sanitized or r"$^2$" in sanitized


def test_sanitize_latex_text_preserves_math_and_labels():
    raw_input = r"See Table~\ref{tab:token_cost} for candidate $K_{ij}$ and cand_euler_10."
    sanitized = sanitize_latex_text(raw_input)

    # Reference label key should remain unescaped
    assert r"\ref{tab:token_cost}" in sanitized
    # Subscript inside math mode should remain unescaped
    assert r"$K_{ij}$" in sanitized
    # Underscores outside math mode should be escaped
    assert r"cand\_euler\_10" in sanitized


def test_sanitize_and_compile_original_final_report_tex(tmp_path: Path):
    """Verify that test fixture final_report.tex containing raw Unicode and unescaped characters
    is fully sanitized and compiles cleanly to PDF via pdflatex without errors.
    """
    raw_report_path = Path(__file__).parent / "data" / "final_report.tex"
    if not raw_report_path.exists():
        pytest.skip(f"Reference test file {raw_report_path} does not exist")

    raw_content = raw_report_path.read_text(encoding="utf-8")
    assert "σ" in raw_content or "≤" in raw_content or "16.67%" in raw_content

    # Sanitize document
    sanitized_content = sanitize_latex_document(raw_content)

    # Verify problematic raw characters are resolved
    assert "σ" not in sanitized_content
    assert "≤" not in sanitized_content
    assert "16.67%" not in sanitized_content
    assert r"16.67\%" in sanitized_content
    assert r"Code Rewrite \& Simulation" in sanitized_content

    # Write to temporary output directory and compile with pdflatex
    pdflatex_bin = _find_pdflatex()
    if not pdflatex_bin:
        pytest.skip("pdflatex not found on system; skipping compilation verification")

    out_dir = tmp_path / "sanitized_compile_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_file = out_dir / "sanitized_final_report.tex"
    tex_file.write_text(sanitized_content, encoding="utf-8")

    # Copy figure assets so compilation succeeds
    src_figures = raw_report_path.parent / "figures"
    if src_figures.exists():
        import shutil
        shutil.copytree(src_figures, out_dir / "figures", dirs_exist_ok=True)

    cmd = [
        pdflatex_bin,
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_file),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    assert res.returncode == 0, f"pdflatex failed with output:\n{res.stdout[-1500:]}"

    pdf_file = out_dir / "sanitized_final_report.pdf"
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1000
