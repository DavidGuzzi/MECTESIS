"""
Genera entrega/tesis/output/anexo_simulaciones_{univariadas,multivariadas,covariadas}.tex
envolviendo cada pagina de los PDFs combinados en un \\begin{figure} + \\includegraphics[page=N].

Esto permite que main.tex haga \\input de estos .tex (mismo patron que las tablas
*_full_summary_combined_table.tex) en lugar de usar \\includepdf, dando control total
a LaTeX sobre el tamano de hoja, captions y page breaks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "entrega" / "tesis" / "output"

BLOCKS = [
    {
        "slug": "univariadas",
        "pretty": "univariado",
        "caption": "Visualizaciones del bloque univariado",
        "label_prefix": "fig:anexo_uni",
    },
    {
        "slug": "multivariadas",
        "pretty": "multivariado",
        "caption": "Visualizaciones del bloque multivariado",
        "label_prefix": "fig:anexo_multi",
    },
    {
        "slug": "covariadas",
        "pretty": "con covariables",
        "caption": "Visualizaciones del bloque con covariables",
        "label_prefix": "fig:anexo_cov",
    },
]


def count_pages(pdf: Path) -> int:
    raw = subprocess.check_output(["pdfinfo", str(pdf)])
    out = raw.decode("latin-1", errors="replace")
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"No 'Pages:' en pdfinfo de {pdf}")


def emit_tex(block: dict) -> Path:
    pdf_name = f"anexo_simulaciones_{block['slug']}.pdf"
    pdf_path = OUTPUT_DIR / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    n_pages = count_pages(pdf_path)

    tex_path = OUTPUT_DIR / f"anexo_simulaciones_{block['slug']}.tex"
    lines = [
        f"% Visualizaciones del bloque {block['pretty']}.",
        "% Generado automaticamente por scripts/build_anexo_visual_wrappers.py.",
        f"% Envuelve cada pagina de {pdf_name} en un environment figure.",
        "",
    ]
    for p in range(1, n_pages + 1):
        suffix = "" if n_pages == 1 else f" (pagina {p} de {n_pages})"
        label = f"{block['label_prefix']}_p{p}" if n_pages > 1 else block["label_prefix"]
        lines.extend([
            r"\begin{figure}[H]",
            r"\centering",
            rf"\adjustbox{{max width=\textwidth, max totalheight=0.88\textheight}}{{\includegraphics[page={p}]{{output/{pdf_name}}}}}",
            rf"\caption{{{block['caption']}{suffix}.}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
            "",
        ])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def main() -> int:
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] no existe {OUTPUT_DIR}", file=sys.stderr)
        return 1
    for block in BLOCKS:
        try:
            path = emit_tex(block)
            print(f"[ok] {path.name}")
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
