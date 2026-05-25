"""Genera el cuadro resumen completo de los experimentos univariados v5.

Lee los CSV producidos por el Monte Carlo univariado v5 (97 experimentos
A.1..G.4) en `notebooks/results/univariate_v5_vertexai/` y emite tres
fragmentos LaTeX (uno por tamano muestral T) mas un wrapper standalone que
compila a un PDF en formato vertical (portrait), una hoja por T.

Cada fragmento es un `tabular` con una fila por experimento y, por cada
bloque de horizonte (Corto/Medio/Largo) valido para ese T, cuatro columnas
con los cocientes Chronos/Clasico de las metricas Bias, Var, RMSE y CRPS.
El coloreado y los triangulos siguen la convencion ya validada en la tesis:
fondo purpura + triangulo si gana Chronos (cociente < 1), fondo azul +
triangulo si gana el clasico (cociente > 1), neutro si la diferencia con 1
es menor al 0.5%.

La primera columna identifica cada experimento por la descripcion del proceso
generador (DGP), extraida de la notebook (ej. "AR(1) rho=0.30"), convertida a
LaTeX legible. Para que cada cuadro entre en una sola hoja A4 vertical se usa
`adjustbox` (max width / max totalheight), que reduce la tabla manteniendo la
proporcion.

Para la metrica `bias` el cociente se computa sobre valores absolutos
(|bias_C|/|bias_cl|), para que `<1` siga representando "Chronos mejor".

Uso:
    python scripts/build_full_univariate_summary.py
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "notebooks" / "results" / "univariate_v5_vertexai"
OUTPUT_DIR = ROOT / "entrega" / "tesis" / "output"
NOTEBOOK = ROOT / "notebooks" / "experimentos_univariados_v5_cloud.ipynb"

T_LIST = [50, 100, 200]
R = 500

BLOCKS = [
    ("Corto", 1, 6),
    ("Medio", 7, 18),
    ("Largo", 19, 24),
]
H_BY_T = {50: 6, 100: 18, 200: 24}

METRICS = ["bias", "variance", "rmse", "crps"]
METRIC_LABELS = {
    "bias":     r"Bias",
    "variance": r"Var",
    "rmse":     r"RMSE",
    "crps":     r"CRPS",
}

EXP_RE = re.compile(r"^exp_([A-G])_(\d+)_T50_R" + str(R) + r"\.csv$")

# Resumen en lenguaje llano de cada bloque, para el renglon de cabecera que
# antecede a sus experimentos (le dice al lector que va a encontrar). Los bloques
# A/B originales (procesos ARMA con y sin tendencia) se fusionan por familia del
# proceso (AR/MA/ARMA); el resto conserva su agrupamiento. Las claves de este
# dict son las claves de grupo que devuelve group_key(), no las letras del CSV.
BLOCK_SUMMARY = {
    "AR":   r"Bloque Procesos AR --- $p$ (orden autorregresivo) $\in \{1,2,3,4\}$, "
            r"$\rho$ (persistencia) $\in \{0.30,\, 0.90\}$, "
            r"$\delta$ (tendencia) $\in \{0,\, 0.02,\, 0.10\}$",
    "MA":   r"Bloque Procesos MA --- $q$ (orden de media m\'ovil) $\in \{1,2,3,4\}$, "
            r"$\theta$ (persistencia) $\in \{0.30,\, 0.90\}$, "
            r"$\delta$ (tendencia) $\in \{0,\, 0.02,\, 0.10\}$",
    "ARMA": r"Bloque Procesos ARMA --- \'ordenes $(p,q) \in \{(1,1),(2,2),(1,4),(4,1)\}$, "
            r"$\rho$ (persistencia) $\in \{0.30,\, 0.90\}$, "
            r"$\delta$ (tendencia) $\in \{0,\, 0.02,\, 0.10\}$",
    "C": r"Bloque Caminata aleatoria --- ra\'iz unitaria con deriva "
         r"$\delta \in \{0,\, 0.05,\, 0.20\}$",
    "D": r"Bloque Volatilidad condicional --- AR(1) con errores ARCH(1) "
         r"($\alpha \in \{0.10,\, 0.50\}$) y GARCH(1,1) "
         r"($\alpha+\beta \in \{0.50,\, 0.95\}$)",
    "E": r"Bloque Suavizado exponencial --- ETS (nivel, tendencia y "
         r"estacionalidad) y m\'etodo Theta",
    "F": r"Bloque Estacionales SARIMA --- $s \in \{4,\, 12\}$, estacionarios e "
         r"integrados",
    "G": r"Bloque No lineales de umbral --- SETAR, LSTAR y ESTAR",
}

# Familia del proceso para los bloques A/B; clasifica por la descripcion (que
# empieza por AR(.. / MA(.. / ARMA(..). ARMA antes que AR en la alternancia.
_FAMILY_RE = re.compile(r"^(ARMA|AR|MA)\b")
FAMILY_ORDER = {"AR": 0, "MA": 1, "ARMA": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def group_key(block_letter: str, idx: int, descriptions: dict[str, str]) -> str:
    """Clave de bloque para el renglon de cabecera. Los bloques A/B (procesos
    ARMA con y sin tendencia) se colapsan en su familia AR/MA/ARMA segun la
    descripcion del experimento; el resto conserva su letra original."""
    if block_letter in ("A", "B"):
        m = _FAMILY_RE.match(descriptions.get(f"{block_letter}.{idx}", ""))
        if m:
            return m.group(1)
        # Respaldo por indice: 8 AR, 8 MA, 8 ARMA por nivel de tendencia.
        return ("AR", "MA", "ARMA")[((idx - 1) % 24) // 8]
    return block_letter


def _block_header_row(block_key: str, total_cols: int) -> str:
    """Renglon de cabecera (gris, negrita) que resume un bloque y abarca toda
    la fila."""
    desc = BLOCK_SUMMARY.get(block_key, f"Bloque {block_key}")
    return (rf"\multicolumn{{{total_cols}}}{{l}}{{\cellcolor{{gray!15}}"
            rf"\textbf{{{desc}}}}} \\")


# ---------------------------------------------------------------------------
# Descripciones del DGP (etiquetas de fila) extraidas de la notebook.
# ---------------------------------------------------------------------------

_GREEK = {
    "rho": r"\rho", "theta": r"\theta", "delta": r"\delta", "alpha": r"\alpha",
    "phi": r"\phi", "beta": r"\beta", "sigma": r"\sigma", "gamma": r"\gamma",
}


def latexify_desc(s: str) -> str:
    """Convierte la descripcion cruda del DGP a LaTeX legible.

    rho/theta/... = valor  ->  $\\rho = valor$
    L^n -> $L^{n}$ ; (1-L)(1-L^n) -> math ; s=n -> $s=n$ ; _n -> \\textsubscript{n}
    """
    s = re.sub(r"\(1-L\)\(1-L\^(\d+)\)",
               lambda m: r"$(1-L)(1-L^{%s})$" % m.group(1), s)
    s = re.sub(r"L\^(\d+)", lambda m: r"$L^{%s}$" % m.group(1), s)
    s = re.sub(r"\b(rho|theta|delta|alpha|phi|beta|sigma|gamma)\s*=\s*([0-9.]+)",
               lambda m: r"$%s=%s$" % (_GREEK[m.group(1)], m.group(2)), s)
    s = re.sub(r"\bs=(\d+)", lambda m: r"$s=%s$" % m.group(1), s)
    s = re.sub(r"_(\d+)", lambda m: r"\textsubscript{%s}" % m.group(1), s)
    return s


def load_descriptions() -> dict[str, str]:
    """Parsea la notebook y devuelve {exp_id: descripcion_latex}.

    Cada celda de experimento define `verify_dgp("A.1 -- AR(1) rho=0.30", ...)`
    y/o un comentario `# A.1 -- ...`. Se toma el texto despues de '--'.
    """
    if not NOTEBOOK.exists():
        print(f"[warn] notebook no encontrada ({NOTEBOOK}); se usaran IDs como etiqueta")
        return {}
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    pat_desc = re.compile(r'verify_dgp\(\s*"([A-G]\.\d+)\s*--\s*([^"]+)"')
    pat_comment = re.compile(r'#\s*([A-G]\.\d+)\s*--\s*(.+)')
    out: dict[str, str] = {}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        for m in pat_desc.finditer(src):
            out[m.group(1)] = latexify_desc(m.group(2).strip())
        for m in pat_comment.finditer(src):
            out.setdefault(m.group(1), latexify_desc(m.group(2).strip()))
    return out


# ---------------------------------------------------------------------------
# Carga y agregacion de los CSV.
# ---------------------------------------------------------------------------

def discover_experiments() -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for p in RESULTS_DIR.glob("exp_*_T50_R*.csv"):
        m = EXP_RE.match(p.name)
        if m:
            found.append((m.group(1), int(m.group(2))))
    found.sort()
    return found


def load_csv(block: str, idx: int, T: int) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"exp_{block}_{idx}_T{T}_R{R}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["horizon"].astype(str).str.match(r"^\d+$")].copy()
    df["horizon"] = df["horizon"].astype(int)
    return df


def chronos_row_name(models: list[str]) -> str:
    for m in models:
        if "chronos" in m.lower():
            return m
    raise KeyError(f"No se encontro fila Chronos entre: {models}")


def classical_row_name(models: list[str]) -> str:
    for m in models:
        if "chronos" not in m.lower():
            return m
    raise KeyError(f"No se encontro fila clasica entre: {models}")


def aggregate_block(df: pd.DataFrame, h_lo: int, h_hi: int) -> dict[str, dict[str, float]]:
    sub = df[(df["horizon"] >= h_lo) & (df["horizon"] <= h_hi)]
    out: dict[str, dict[str, float]] = {}
    for model, grp in sub.groupby("model"):
        out[model] = {m: grp[m].mean() for m in METRICS if m in grp.columns}
    return out


# Degradado en el FONDO de toda la celda: blanco (cerca de 1) -> color, pero
# acotado a un rango claro. El maximo (cocientes extremos) queda apenas por
# encima del tono claro "normal" de la tesis (~t=0.22); cerca de 1 se aclara
# casi a blanco. Nunca se oscurece hacia el color pleno.
CHRONOS_RGB = (0x96, 0x72, 0xB6)   # violeta -> Chronos mejor (cociente < 1)
CLASSIC_RGB = (0x4C, 0x72, 0xB0)   # azul    -> clasico mejor (cociente > 1)
SAT_CAP = 2.5      # fold-change al que el fondo llega a su tono mas intenso
BG_T_MIN = 0.08    # mezcla blanco->color cerca de 1 (muy claro)
BG_T_MAX = 0.30    # mezcla en el extremo (apenas mas que el claro normal ~0.22)


def _blend_white(rgb: tuple[int, int, int], t: float) -> str:
    """Mezcla blanco -> rgb con factor t in [0,1]; devuelve hex 'RRGGBB'."""
    r, g, b = (round(255 + (c - 255) * t) for c in rgb)
    return f"{r:02X}{g:02X}{b:02X}"


def ratio_cell(ratio: float) -> str:
    """Celda con fondo en degradado claro segun la magnitud del cociente.

    El fondo va de casi blanco (cerca de 1) a un tono apenas mas intenso que el
    claro normal de la tesis en el extremo: violeta si Chronos gana (cociente
    < 1), azul si gana el clasico (> 1). La intensidad usa el fold-change
    |log(ratio)| (escala log, simetrica). Zona neutra a +-0.5% de 1. El
    triangulo (color pleno) marca el ganador.
    """
    if pd.isna(ratio):
        return r"---\,\phantom{\winC}"
    val = f"{ratio:.2f}"
    if abs(ratio - 1.0) < 5e-3:
        return rf"{val}\,\phantom{{\winC}}"
    fold = ratio if ratio > 1.0 else 1.0 / ratio
    u = min(math.log(fold) / math.log(SAT_CAP), 1.0)
    blend = BG_T_MIN + (BG_T_MAX - BG_T_MIN) * u
    if ratio < 1.0:
        return rf"\cellcolor[HTML]{{{_blend_white(CHRONOS_RGB, blend)}}}{val}\,\winC"
    return rf"\cellcolor[HTML]{{{_blend_white(CLASSIC_RGB, blend)}}}{val}\,\winT"


def compute_ratio(metric: str, v_chronos: float, v_classical: float) -> float:
    if pd.isna(v_chronos) or pd.isna(v_classical) or v_classical == 0:
        return float("nan")
    if metric == "bias":
        denom = abs(v_classical)
        if denom == 0:
            return float("nan")
        return abs(v_chronos) / denom
    return v_chronos / v_classical


def valid_blocks_for_T(T: int) -> list[tuple[str, int, int]]:
    h_max = H_BY_T[T]
    out: list[tuple[str, int, int]] = []
    for name, lo, hi in BLOCKS:
        if lo > h_max:
            continue
        out.append((name, lo, min(hi, h_max)))
    return out


# ---------------------------------------------------------------------------
# Render del cuadro (un tabular por T, ajustado a una hoja vertical).
# ---------------------------------------------------------------------------

def build_full_summary_for_T(
    T: int,
    experiments: list[tuple[str, int]],
    descriptions: dict[str, str],
) -> str:
    valid = valid_blocks_for_T(T)
    n_blocks = len(valid)
    n_metrics = len(METRICS)
    col_spec = "l " + " ".join(["cccc"] * n_blocks)

    lines: list[str] = []
    lines.append(r"\begin{center}")
    lines.append(rf"\captionof{{table}}{{}}\label{{tab:univ_full_T{T}}}")
    lines.append(r"\par\vspace{3pt}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.9\textheight}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header1_parts: list[str] = [""]
    for name, lo, hi in valid:
        header1_parts.append(
            rf"\multicolumn{{{n_metrics}}}{{c}}{{{name} $h \in [{lo},{hi}]$}}"
        )
    lines.append(" & ".join(header1_parts) + r" \\")

    cmid_parts: list[str] = []
    col_idx = 2
    for _ in valid:
        cmid_parts.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + n_metrics - 1}}}")
        col_idx += n_metrics
    lines.append("".join(cmid_parts))

    metric_header = " & ".join(METRIC_LABELS[m] for m in METRICS)
    header2_parts: list[str] = [r"Experimento"]
    for _ in valid:
        header2_parts.append(metric_header)
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    total_cols = 1 + n_blocks * n_metrics
    prev_block: str | None = None
    for block_letter, idx in experiments:
        g = group_key(block_letter, idx, descriptions)
        if g != prev_block:
            if prev_block is not None:
                lines.append(r"\midrule")
            lines.append(_block_header_row(g, total_cols))
            prev_block = g

        exp_id = f"{block_letter}.{idx}"
        df = load_csv(block_letter, idx, T)
        label = descriptions.get(exp_id, exp_id)
        if df is None or df.empty:
            print(f"[WARN] T={T} {exp_id}: CSV ausente o vacio; fila omitida")
            continue

        cells: list[str] = []
        for _, lo, hi in valid:
            agg = aggregate_block(df, lo, hi)
            try:
                ch_name = chronos_row_name(list(agg.keys()))
                cl_name = classical_row_name(list(agg.keys()))
            except KeyError as exc:
                print(f"[WARN] T={T} {exp_id} bloque [{lo},{hi}]: {exc}")
                cells.extend([ratio_cell(float("nan"))] * n_metrics)
                continue
            ch_vals = agg[ch_name]
            cl_vals = agg[cl_name]
            for m in METRICS:
                vc = ch_vals.get(m, float("nan"))
                vt = cl_vals.get(m, float("nan"))
                cells.append(ratio_cell(compute_ratio(m, vc, vt)))

        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{center}")
    return "\n".join(lines) + "\n"


def build_combined_summary(
    experiments: list[tuple[str, int]],
    descriptions: dict[str, str],
) -> str:
    """Un unico cuadro con todos los T lado a lado.

    Encabezado en tres niveles: T (T=50/100/200) -> bloque H (Corto/Medio/Largo)
    -> metricas (Bias/Var/RMSE/CRPS). Una fila por experimento. Mismo coloreado
    y cocientes que los cuadros por T.
    """
    layout = [(T, valid_blocks_for_T(T)) for T in T_LIST]
    n_metrics = len(METRICS)

    # Especificacion de columnas: 'l' + cccc por cada bloque valido de cada T.
    col_spec_parts = ["l"]
    for _, blocks in layout:
        for _ in blocks:
            col_spec_parts.append("cccc")
    col_spec = " ".join(col_spec_parts)

    # Nivel 1: agrupamiento por T.
    row1_parts = [""]
    cmid1 = []
    idx = 2
    for T, blocks in layout:
        span = len(blocks) * n_metrics
        row1_parts.append(rf"\multicolumn{{{span}}}{{c}}{{$T = {T}$}}")
        cmid1.append(rf"\cmidrule(lr){{{idx}-{idx + span - 1}}}")
        idx += span

    # Nivel 2: bloque de horizonte dentro de cada T.
    row2_parts = [""]
    cmid2 = []
    idx = 2
    for T, blocks in layout:
        for bname, lo, hi in blocks:
            row2_parts.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{{bname}}}")
            cmid2.append(rf"\cmidrule(lr){{{idx}-{idx + n_metrics - 1}}}")
            idx += n_metrics

    # Nivel 3: metricas.
    metric_header = " & ".join(METRIC_LABELS[m] for m in METRICS)
    row3_parts = [r"Experimento"]
    for _, blocks in layout:
        for _ in blocks:
            row3_parts.append(metric_header)

    lines: list[str] = []
    lines.append(r"\begin{center}")
    lines.append(r"\captionof{table}{}\label{tab:univ_full_combinado}")
    lines.append(r"\par\vspace{3pt}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.9\textheight}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(row1_parts) + r" \\")
    lines.append("".join(cmid1))
    lines.append(" & ".join(row2_parts) + r" \\")
    lines.append("".join(cmid2))
    lines.append(" & ".join(row3_parts) + r" \\")
    lines.append(r"\midrule")

    total_cols = 1 + n_metrics * sum(len(b) for _, b in layout)
    prev_block: str | None = None
    for block_letter, idx_exp in experiments:
        g = group_key(block_letter, idx_exp, descriptions)
        if g != prev_block:
            if prev_block is not None:
                lines.append(r"\midrule")
            lines.append(_block_header_row(g, total_cols))
            prev_block = g

        exp_id = f"{block_letter}.{idx_exp}"
        label = descriptions.get(exp_id, exp_id)
        cells: list[str] = []
        any_data = False
        for T, blocks in layout:
            df = load_csv(block_letter, idx_exp, T)
            for _, lo, hi in blocks:
                if df is None or df.empty:
                    cells.extend([ratio_cell(float("nan"))] * n_metrics)
                    continue
                agg = aggregate_block(df, lo, hi)
                try:
                    ch_name = chronos_row_name(list(agg.keys()))
                    cl_name = classical_row_name(list(agg.keys()))
                except KeyError:
                    cells.extend([ratio_cell(float("nan"))] * n_metrics)
                    continue
                any_data = True
                ch_vals = agg[ch_name]
                cl_vals = agg[cl_name]
                for m in METRICS:
                    vc = ch_vals.get(m, float("nan"))
                    vt = cl_vals.get(m, float("nan"))
                    cells.append(ratio_cell(compute_ratio(m, vc, vt)))
        if not any_data:
            print(f"[WARN] {exp_id}: sin datos en ningun T; fila omitida")
            continue
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{center}")
    return "\n".join(lines) + "\n"


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.2cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{colortbl}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{adjustbox}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small,skip=4pt,justification=justified}
\renewcommand{\tablename}{Tabla}

% Macros copiadas de entrega/tesis/main.tex (mismas definiciones de la tesis)
\definecolor{chronoscolor}{HTML}{9672B6}
\definecolor{classiccolor}{HTML}{4C72B0}
\definecolor{chronoslight}{HTML}{E8DDF1}
\definecolor{classiclight}{HTML}{DCE5F0}
\newcommand{\winC}{\textcolor{chronoscolor}{$\blacktriangle$}}
\newcommand{\winT}{\textcolor{classiccolor}{$\blacktriangle$}}
\newcommand{\cellC}[1]{\cellcolor{chronoslight}#1}
\newcommand{\cellT}[1]{\cellcolor{classiclight}#1}

\setlength{\parindent}{0pt}
"""


def make_document(body: str) -> str:
    return PREAMBLE + "\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n"


WRAPPER_BODY = (
    r"\input{univ_full_summary_T50.tex}" "\n\n"
    r"\clearpage" "\n"
    r"\input{univ_full_summary_T100.tex}" "\n\n"
    r"\clearpage" "\n"
    r"\input{univ_full_summary_T200.tex}" "\n"
)

WRAPPER_BODY_COMBINED = r"\input{univ_full_summary_combined_table.tex}" "\n"


def write_wrapper(path: Path, body: str) -> None:
    path.write_text(make_document(body), encoding="utf-8")


def try_compile(wrapper_path: Path) -> Path | None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("[info] pdflatex no encontrado en PATH. Para compilar manualmente:")
        print(f"       cd {wrapper_path.parent}")
        print(f"       pdflatex -interaction=nonstopmode {wrapper_path.name}")
        return None
    # --enable-installer: MiKTeX baja automaticamente paquetes faltantes
    # (p. ej. adjustbox y sus dependencias) sin prompt interactivo.
    cmd = [pdflatex, "--enable-installer",
           "-interaction=nonstopmode", "-halt-on-error", wrapper_path.name]
    pdf_path = wrapper_path.with_suffix(".pdf")
    # Dos pasadas: la segunda resuelve los \label de \captionof.
    for npass in (1, 2):
        # En Windows, Defender puede tener el PDF recien escrito con un lock
        # momentaneo entre pasadas ("I can't write on file"). Reintentamos.
        for attempt in range(1, 4):
            print(f"[run]  pdflatex pasada {npass}"
                  + (f" (reintento {attempt})" if attempt > 1 else "")
                  + f"  (cwd={wrapper_path.parent})")
            result = subprocess.run(cmd, cwd=wrapper_path.parent, check=False,
                                    capture_output=True, text=True)
            if result.returncode == 0:
                break
            locked = "can't write on file" in result.stdout
            if locked and attempt < 3:
                print("[warn] PDF bloqueado (Defender/visor); reintentando en 2 s...")
                time.sleep(2)
                continue
            print("[err]  pdflatex fallo. Stdout (ultimas 40 lineas):")
            for line in result.stdout.splitlines()[-40:]:
                print("       " + line)
            if locked:
                print(f"[hint] Cerra cualquier visor que tenga abierto {pdf_path.name} y reintenta.")
            return None
    if not pdf_path.exists():
        return None
    return pdf_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments()
    if not experiments:
        print(f"[err] No se encontraron experimentos en {RESULTS_DIR}")
        return
    descriptions = load_descriptions()
    # Reordenar fusionando A/B por familia (AR, MA, ARMA) y manteniendo C..G.
    experiments.sort(
        key=lambda bi: (FAMILY_ORDER[group_key(bi[0], bi[1], descriptions)], bi[0], bi[1])
    )
    missing = [f"{b}.{i}" for b, i in experiments if f"{b}.{i}" not in descriptions]
    print(f"[ok]  {len(experiments)} experimentos; {len(descriptions)} descripciones cargadas"
          + (f"; sin descripcion: {missing}" if missing else ""))

    # --- Version A: tres cuadros, uno por T (una hoja cada uno) ---
    for T in T_LIST:
        body = build_full_summary_for_T(T, experiments, descriptions)
        out_path = OUTPUT_DIR / f"univ_full_summary_T{T}.tex"
        header = (
            f"% Resumen completo univariado v5, T={T}.\n"
            "% Generado automaticamente por scripts/build_full_univariate_summary.py\n"
        )
        out_path.write_text(header + body, encoding="utf-8")
        print(f"[ok]   {out_path.relative_to(ROOT)}")

    wrapper_path = OUTPUT_DIR / "univ_full_summary.tex"
    write_wrapper(wrapper_path, WRAPPER_BODY)
    print(f"[ok]   {wrapper_path.relative_to(ROOT)} (wrapper 3 cuadros)")

    # --- Version B: un unico cuadro combinado con todos los T ---
    combined_body = build_combined_summary(experiments, descriptions)
    combined_frag = OUTPUT_DIR / "univ_full_summary_combined_table.tex"
    combined_frag.write_text(
        "% Resumen univariado v5 -- vista combinada (todos los T).\n"
        "% Generado automaticamente por scripts/build_full_univariate_summary.py\n"
        + combined_body,
        encoding="utf-8",
    )
    print(f"[ok]   {combined_frag.relative_to(ROOT)}")

    combined_wrapper = OUTPUT_DIR / "univ_full_summary_combined.tex"
    write_wrapper(combined_wrapper, WRAPPER_BODY_COMBINED)
    print(f"[ok]   {combined_wrapper.relative_to(ROOT)} (wrapper combinado)")

    pdf = try_compile(wrapper_path)
    if pdf is not None:
        print(f"[ok]   PDF generado: {pdf}")
    pdf_c = try_compile(combined_wrapper)
    if pdf_c is not None:
        print(f"[ok]   PDF generado: {pdf_c}")


if __name__ == "__main__":
    main()
