"""Sanitization utilities for LaTeX document generation.

Maps raw Unicode mathematical symbols to LaTeX equivalents and escapes
special characters outside math mode to ensure reliable pdflatex compilation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# Mapping of mathematical, Greek, and typographic Unicode characters to LaTeX commands
UNICODE_LATEX_MAP: Dict[str, str] = {
    # Greek lowercase
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ϵ": r"$\epsilon$",
    "ε": r"$\varepsilon$",
    "ζ": r"$\zeta$",
    "η": r"$\eta$",
    "θ": r"$\theta$",
    "ϑ": r"$\vartheta$",
    "ι": r"$\iota$",
    "κ": r"$\kappa$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "ν": r"$\nu$",
    "ξ": r"$\xi$",
    "π": r"$\pi$",
    "ϖ": r"$\varpi$",
    "ρ": r"$\rho$",
    "ϱ": r"$\varrho$",
    "σ": r"$\sigma$",
    "ς": r"$\varsigma$",
    "τ": r"$\tau$",
    "υ": r"$\upsilon$",
    "ϕ": r"$\phi$",
    "φ": r"$\varphi$",
    "χ": r"$\chi$",
    "ψ": r"$\psi$",
    "ω": r"$\omega$",
    # Greek uppercase
    "Γ": r"$\Gamma$",
    "Δ": r"$\Delta$",
    "Θ": r"$\Theta$",
    "Λ": r"$\Lambda$",
    "Ξ": r"$\Xi$",
    "Π": r"$\Pi$",
    "Σ": r"$\Sigma$",
    "Υ": r"$\Upsilon$",
    "Φ": r"$\Phi$",
    "Ψ": r"$\Psi$",
    "Ω": r"$\Omega$",
    # Mathematical relations & operators
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "≠": r"$\neq$",
    "≈": r"$\approx$",
    "±": r"$\pm$",
    "∓": r"$\mp$",
    "×": r"$\times$",
    "÷": r"$\div$",
    "∈": r"$\in$",
    "∉": r"$\notin$",
    "⊂": r"$\subset$",
    "⊆": r"$\subseteq$",
    "∪": r"$\cup$",
    "∩": r"$\cap$",
    "∑": r"$\sum$",
    "∏": r"$\prod$",
    "∫": r"$\int$",
    "∂": r"$\partial$",
    "∇": r"$\nabla$",
    "√": r"$\sqrt{}$",
    "∞": r"$\infty$",
    "∝": r"$\propto$",
    "·": r"$\cdot$",
    "°": r"$^\circ$",
    "→": r"$\rightarrow$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$",
    "⇐": r"$\Leftarrow$",
    "⇔": r"$\Leftrightarrow$",
    "…": r"\dots",
    # Typographic punctuation
    "–": "--",   # en-dash
    "—": "---",  # em-dash
    "−": "$-$",  # minus sign
    "“": "''",
    "”": "''",
    "‘": "'",
    "’": "'",
    "•": r"\textbullet{}",
}


def sanitize_latex_text(text: Any) -> str:
    """Sanitize free-form text or LLM narrative for safe embedding in LaTeX documents.

    - Escapes currency dollar signs ($0.503 -> \\$0.503).
    - Escapes percent symbols outside LaTeX commands (16.67% -> 16.67\\%).
    - Maps mathematical Unicode symbols (σ -> $\\sigma$, ≤ -> $\\le$, etc.).
    - Escapes carets (O(N^2) -> O(N\\textsuperscript{2})).
    - Escapes special characters (&, _, #) outside of math mode ($...$)
      while preserving LaTeX command macros like \\ref{...}, \\label{...}, \\cite{...}.

    Args:
        text: Input string or object to sanitize.

    Returns:
        Sanitized LaTeX string.
    """
    if text is None:
        return ""
    s = str(text)

    # 1. Currency dollars: replace isolated currency $ followed by numbers BEFORE introducing math $...$
    s = re.sub(r'(?<!\\)\$(?=\s*\d)', r'\\$', s)

    # 2. Percentages: 16.67% -> 16.67\%
    s = re.sub(r'(?<!\\)%', r'\%', s)

    # 3. Map mathematical Unicode characters to LaTeX commands
    for char, replacement in UNICODE_LATEX_MAP.items():
        s = s.replace(char, replacement)

    # 4. Carets outside math mode
    s = re.sub(r'(?<!\\)\^(\d+)', r'\\textsuperscript{\1}', s)
    s = re.sub(r'(?<!\\)\^', r'\\textasciicircum{}', s)

    # 5. Split by math mode ($...$) and LaTeX command references (\ref{...}, etc.)
    # to protect inner math and label keys from underscore/ampersand escaping
    tokens = re.split(r'(\$(?:\\\$|[^$])+\$|\\(?:ref|label|cite|includegraphics|begin|end)\{[^}]+\})', s)
    out_tokens: List[str] = []

    for token in tokens:
        if not token:
            continue
        # Preserve math environments intact
        if token.startswith("$") and token.endswith("$") and len(token) >= 2:
            out_tokens.append(token)
        # Preserve reference label keys intact
        elif re.match(r'^\\(?:ref|label|cite|includegraphics|begin|end)\{', token):
            out_tokens.append(token)
        else:
            # Text mode segment: escape LaTeX specials
            p = token
            p = re.sub(r'(?<!\\)&', r'\&', p)
            p = re.sub(r'(?<!\\)_', r'\_', p)
            p = re.sub(r'(?<!\\)#', r'\#', p)
            out_tokens.append(p)

    return "".join(out_tokens)


def sanitize_latex_document(doc_str: str) -> str:
    """Sanitize an entire LaTeX document or existing .tex file.

    Preserves document structure (preamble, tables, figures, environments)
    while sanitizing narrative paragraphs, itemized points, and unescaped table texts.

    Args:
        doc_str: Raw LaTeX document content.

    Returns:
        Fully sanitized LaTeX document string that compiles cleanly with pdflatex.
    """
    lines = doc_str.splitlines()
    sanitized_lines: List[str] = []
    in_document = False
    in_tabular = False

    cmd_prefixes = (
        r"\documentclass", r"\usepackage", r"\hypersetup", r"\title", r"\author",
        r"\IEEE", r"\begin", r"\end", r"\maketitle", r"\caption", r"\label",
        r"\includegraphics", r"\toprule", r"\midrule", r"\bottomrule", r"\centering",
    )

    for line in lines:
        stripped = line.strip()

        if r"\begin{document}" in line:
            in_document = True
            sanitized_lines.append(line)
            continue

        if not in_document:
            sanitized_lines.append(line)
            continue

        # Handle tabular environment boundaries
        if r"\begin{tabular}" in line:
            in_tabular = True
            sanitized_lines.append(line)
            continue
        if r"\end{tabular}" in line:
            in_tabular = False
            sanitized_lines.append(line)
            continue

        # Preserve structural LaTeX commands without escaping backslashes or command names
        if any(stripped.startswith(p) for p in cmd_prefixes):
            sanitized_lines.append(line)
            continue

        # Handle itemized list lines (\item ...)
        if stripped.startswith(r"\item"):
            idx = line.find(r"\item") + 5
            prefix = line[:idx]
            rest = line[idx:]
            sanitized_lines.append(prefix + " " + sanitize_latex_text(rest))
            continue

        # Table rows: preserve column & delimiters while sanitizing unescaped labels
        if in_tabular:
            # Known unescaped ampersands in phase labels
            row = line.replace("Code Rewrite & Simulation", r"Code Rewrite \& Simulation")
            row = row.replace("Auditing & Final Reporting", r"Auditing \& Final Reporting")
            # Map Unicode characters in table rows
            for char, repl in UNICODE_LATEX_MAP.items():
                row = row.replace(char, repl)
            sanitized_lines.append(row)
            continue

        # Regular narrative text line
        sanitized_lines.append(sanitize_latex_text(line))

    return "\n".join(sanitized_lines)
