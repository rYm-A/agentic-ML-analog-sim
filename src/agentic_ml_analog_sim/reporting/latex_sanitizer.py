"""Sanitization utilities for LaTeX document generation.

Maps raw Unicode mathematical symbols to LaTeX equivalents, converts Markdown tables
and ASCII diagrams to native LaTeX environments, and escapes special characters outside
math mode to ensure reliable pdflatex compilation.
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


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return bool(s and "|" in s and not s.startswith(r"\documentclass") and not s.startswith(r"\begin"))


def _is_separator_row(line: str) -> bool:
    s = line.strip()
    if not _is_table_row(s):
        return False
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = s.split("|")
    if not cells:
        return False
    return all(re.match(r"^\s*:?-{2,}:?\s*$", c) for c in cells)


def _parse_alignments(sep_line: str) -> List[str]:
    s = sep_line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = s.split("|")
    aligns: List[str] = []
    for c in cells:
        c = c.strip()
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            aligns.append("c")
        elif right:
            aligns.append("r")
        else:
            aligns.append("l")
    return aligns


def _split_row_cells(row_line: str) -> List[str]:
    s = row_line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    s = s.replace(r"\|", "\x00")
    cells = [c.strip().replace("\x00", "|") for c in s.split("|")]
    return cells


def _sanitize_table_cell(cell: str, is_header: bool = False) -> str:
    c = cell.strip()
    if not c:
        return ""

    # Normalize double backslashes in math like (\\sigma=0) -> (\sigma=0)
    c = re.sub(r"\\\\([a-zA-Z]+)", r"\\\1", c)

    # Convert markdown bold: **foo** -> \textbf{foo}
    c = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", c)
    # Convert markdown italic: *foo* -> \textit{foo}
    c = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", c)

    # Convert inline code: `foo` -> \texttt{foo}
    def _code_repl(m: re.Match) -> str:
        code_text = m.group(1)
        code_text = re.sub(r"(?<!\\)_", r"\_", code_text)
        return r"\texttt{" + code_text + "}"

    c = re.sub(r"`([^`]+)`", _code_repl, c)

    # Map Unicode characters
    for char, replacement in UNICODE_LATEX_MAP.items():
        c = c.replace(char, replacement)

    # Currency dollar sign
    c = re.sub(r"(?<!\\)\$(?=\s*\d)", r"\$", c)

    # Percentages
    c = re.sub(r"(?<!\\)%", r"\%", c)

    # Protect math mode segments ($...$) and commands (\textbf{...}, \texttt{...}, etc.)
    tokens = re.split(
        r"(\$(?:\\\$|[^$])+\$|\\(?:textbf|textit|texttt|ref|label|cite)\{[^}]*\})",
        c,
    )
    out: List[str] = []
    for t in tokens:
        if not t:
            continue
        if (t.startswith("$") and t.endswith("$")) or t.startswith("\\"):
            out.append(t)
        else:
            p = t
            p = re.sub(r"(?<!\\)&", r"\&", p)
            p = re.sub(r"(?<!\\)_", r"\_", p)
            p = re.sub(r"(?<!\\)#", r"\#", p)
            out.append(p)
    res = "".join(out)

    if is_header and not res.startswith(r"\textbf{") and res:
        res = r"\textbf{" + res + "}"
    return res


def convert_markdown_tables_to_latex(text: str) -> str:
    """Automatically detect Markdown tables (| ... |) and convert them into native LaTeX tabular environments with booktabs.

    Handles column alignment (:---, :---:, ---:), header formatting (\\textbf),
    inline code (\\texttt), mathematical expressions, bold/italic, and character escaping.

    Args:
        text: Input string containing Markdown tables.

    Returns:
        String with Markdown tables converted to native LaTeX tabular environments.
    """
    if not text or "|" not in text:
        return text

    lines = text.split("\n")
    out_lines: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        if i + 1 < n and _is_table_row(lines[i]) and _is_separator_row(lines[i + 1]):
            header_line = lines[i]
            sep_line = lines[i + 1]
            aligns = _parse_alignments(sep_line)
            header_cells = _split_row_cells(header_line)
            num_cols = max(len(aligns), len(header_cells))
            while len(aligns) < num_cols:
                aligns.append("l")
            col_spec = "".join(aligns[:num_cols])

            processed_headers = []
            for col_idx in range(num_cols):
                raw_cell = header_cells[col_idx] if col_idx < len(header_cells) else ""
                processed_headers.append(_sanitize_table_cell(raw_cell, is_header=True))
            header_row_str = " & ".join(processed_headers) + r" \\"

            body_rows: List[str] = []
            j = i + 2
            while j < n and _is_table_row(lines[j]) and not _is_separator_row(lines[j]):
                row_cells = _split_row_cells(lines[j])
                processed_cells = []
                for col_idx in range(num_cols):
                    raw_cell = row_cells[col_idx] if col_idx < len(row_cells) else ""
                    processed_cells.append(_sanitize_table_cell(raw_cell, is_header=False))
                body_rows.append(" & ".join(processed_cells) + r" \\")
                j += 1

            tabular_lines = [
                f"\\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
                header_row_str,
                r"\midrule",
            ]
            tabular_lines.extend(body_rows)
            tabular_lines.append(r"\bottomrule")
            tabular_lines.append(r"\end{tabular}")
            out_lines.extend(tabular_lines)
            i = j
        else:
            out_lines.append(lines[i])
            i += 1

    return "\n".join(out_lines)


def sanitize_ascii_art(text: str, strip: bool = False) -> str:
    """Sanitize or strip triple-backtick ASCII diagrams, progress bars, and code blocks.

    If strip is True, triple-backtick blocks are removed.
    If strip is False, triple-backtick blocks are converted into clean LaTeX verbatim environments.
    If a triple-backtick block contains a Markdown table, it is unwrapped so it can be
    converted by convert_markdown_tables_to_latex.

    Args:
        text: Input text containing triple backtick blocks.
        strip: Whether to completely strip ASCII art instead of transforming to verbatim.

    Returns:
        Sanitized text with clean LaTeX environments.
    """
    if not text or "```" not in text:
        return text

    pattern = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)```", re.DOTALL)

    def _repl(m: re.Match) -> str:
        content = m.group(1).rstrip("\n")
        if not content.strip():
            return ""

        # Check if the block is actually a markdown table
        block_lines = content.split("\n")
        if any(_is_separator_row(l) for l in block_lines):
            return "\n" + content.strip() + "\n"

        if strip:
            return ""

        # Clean previously escaped LaTeX specials for clean verbatim presentation
        clean_content = (
            content.replace(r"\_", "_")
            .replace(r"\%", "%")
            .replace(r"\$", "$")
            .replace(r"\&", "&")
            .replace(r"\#", "#")
        )
        return f"\\begin{{verbatim}}\n{clean_content}\n\\end{{verbatim}}"

    return pattern.sub(_repl, text)


def sanitize_latex(text: Any, strip_ascii: bool = False) -> str:
    """Sanitize narrative text or LLM output for safe LaTeX embedding.

    Automatically runs:
    1. sanitize_ascii_art: transforms or strips triple-backtick blocks into clean LaTeX verbatim.
    2. convert_markdown_tables_to_latex: converts Markdown tables to LaTeX booktabs tabulars.
    3. Converts Markdown headings (###, ##, ####) to LaTeX subsections/sections.
    4. Converts Markdown formatting (**bold**, *italic*, `code`).
    5. Maps mathematical Unicode characters to LaTeX commands.
    6. Escapes currency dollar signs and percentages.
    7. Escapes special characters (&, _, #) outside math mode and LaTeX environments.

    Args:
        text: Input string or object to sanitize.
        strip_ascii: If True, strips ASCII art blocks instead of transforming to verbatim.

    Returns:
        Sanitized LaTeX string that compiles cleanly.
    """
    if text is None:
        return ""
    s = str(text)

    # 1. Transform or strip triple-backtick ASCII art / code blocks
    s = sanitize_ascii_art(s, strip=strip_ascii)

    # 2. Convert Markdown tables to LaTeX tabular environments
    s = convert_markdown_tables_to_latex(s)

    # 3. Convert Markdown headings to LaTeX sectioning commands
    s = re.sub(r"^(?:\\#|#){4}\s+(.*)$", r"\\subsubsection{\1}", s, flags=re.MULTILINE)
    s = re.sub(r"^(?:\\#|#){3}\s+(.*)$", r"\\subsection{\1}", s, flags=re.MULTILINE)
    s = re.sub(r"^(?:\\#|#){2}\s+(.*)$", r"\\section{\1}", s, flags=re.MULTILINE)

    # 4. Inline code: `code_identifier` -> \texttt{code\_identifier}
    def _code_repl(m: re.Match) -> str:
        code_text = m.group(1)
        code_text = re.sub(r"(?<!\\)_", r"\_", code_text)
        return r"\texttt{" + code_text + "}"

    s = re.sub(r"`([^`\n]+)`", _code_repl, s)

    # 5. Bold & Italic
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", s)

    # 6. Currency dollars: replace isolated currency $ followed by numbers BEFORE introducing math $...$
    s = re.sub(r"(?<!\\)\$(?=\s*\d)", r"\$", s)

    # 7. Percentages: 16.67% -> 16.67\%
    s = re.sub(r"(?<!\\)%", r"\%", s)

    # 8. Map mathematical Unicode characters to LaTeX commands
    for char, replacement in UNICODE_LATEX_MAP.items():
        s = s.replace(char, replacement)

    # 9. Carets outside math mode
    s = re.sub(r"(?<!\\)\^(\d+)", r"\\textsuperscript{\1}", s)
    s = re.sub(r"(?<!\\)\^", r"\\textasciicircum{}", s)

    # 10. Protect multi-line environments (tabular, verbatim, equation, etc.)
    env_pattern = re.compile(
        r"(\\begin\{(?:tabular|table|table\*|verbatim|quote|equation|align|center|figure|figure\*)\*?\}.*?\\end\{(?:tabular|table|table\*|verbatim|quote|equation|align|center|figure|figure\*)\*?\})",
        re.DOTALL,
    )
    protected_blocks: List[str] = []

    def _prot_env(m: re.Match) -> str:
        idx = len(protected_blocks)
        protected_blocks.append(m.group(1))
        return f"LATEXPROTBLOCK{idx}XYZ"

    s = env_pattern.sub(_prot_env, s)

    # 11. Split by math mode ($...$) and LaTeX commands to protect inner math and label keys
    cmd_re = (
        r"("
        r"\$(?:\\\$|[^$])+\$"
        r"|\\(?:ref|label|cite|includegraphics|section|subsection|subsubsection|textbf|textit|texttt|caption|item)\{[^}]*\}"
        r"|LATEXPROTBLOCK\d+XYZ"
        r")"
    )
    tokens = re.split(cmd_re, s)
    out_tokens: List[str] = []

    for token in tokens:
        if not token:
            continue
        if (
            (token.startswith("$") and token.endswith("$") and len(token) >= 2)
            or token.startswith("\\")
            or token.startswith("LATEXPROTBLOCK")
        ):
            out_tokens.append(token)
        else:
            # Text mode segment: escape LaTeX specials
            p = token
            p = re.sub(r"(?<!\\)&", r"\&", p)
            p = re.sub(r"(?<!\\)_", r"\_", p)
            p = re.sub(r"(?<!\\)#", r"\#", p)
            out_tokens.append(p)

    s = "".join(out_tokens)

    # 12. Restore protected environments
    for idx, block in enumerate(protected_blocks):
        s = s.replace(f"LATEXPROTBLOCK{idx}XYZ", block)

    return s


def sanitize_latex_text(text: Any) -> str:
    """Sanitize free-form text or LLM narrative for safe embedding in LaTeX documents.

    Backward-compatible wrapper delegating to sanitize_latex.
    """
    return sanitize_latex(text)


def sanitize_latex_document(doc_str: str) -> str:
    """Sanitize an entire LaTeX document or existing .tex file.

    Preserves document structure (preamble, tables, figures, environments)
    while sanitizing narrative paragraphs, itemized points, and unescaped table texts.
    Converts embedded Markdown tables and ASCII art blocks.

    Args:
        doc_str: Raw LaTeX document content.

    Returns:
        Fully sanitized LaTeX document string that compiles cleanly with pdflatex.
    """
    # Pre-process document to convert Markdown tables and ASCII blocks
    doc_str = sanitize_ascii_art(doc_str)
    doc_str = convert_markdown_tables_to_latex(doc_str)

    lines = doc_str.splitlines()
    sanitized_lines: List[str] = []
    in_document = False
    in_tabular = False
    in_verbatim = False

    cmd_prefixes = (
        r"\documentclass", r"\usepackage", r"\hypersetup", r"\title", r"\author",
        r"\IEEE", r"\begin", r"\end", r"\maketitle", r"\caption", r"\label",
        r"\includegraphics", r"\toprule", r"\midrule", r"\bottomrule", r"\centering",
        r"\section", r"\subsection", r"\subsubsection",
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

        # Handle verbatim environment boundaries
        if r"\begin{verbatim}" in line:
            in_verbatim = True
            sanitized_lines.append(line)
            continue
        if r"\end{verbatim}" in line:
            in_verbatim = False
            sanitized_lines.append(line)
            continue
        if in_verbatim:
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

        # Preserve structural LaTeX commands without escaping backslashes or command names
        if any(stripped.startswith(p) for p in cmd_prefixes):
            sanitized_lines.append(line)
            continue

        # Handle itemized list lines (\item ...)
        if stripped.startswith(r"\item"):
            idx = line.find(r"\item") + 5
            prefix = line[:idx]
            rest = line[idx:]
            sanitized_lines.append(prefix + " " + sanitize_latex(rest))
            continue

        # Regular narrative text line
        sanitized_lines.append(sanitize_latex(line))

    return "\n".join(sanitized_lines)
