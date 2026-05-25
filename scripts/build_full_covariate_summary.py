"""Genera el cuadro resumen completo de los experimentos covariados v6.

Fork de `build_full_univariate_summary.py`. Los experimentos covariados son de
un solo objetivo (pronostico con covariable exogena), asi que se reportan las
mismas cuatro metricas que el resumen univariado -- Bias, Var, RMSE, CRPS --
como cociente Chronos/Clasico. Lee los CSV del Monte Carlo covariado v6
(51 experimentos C-A.1..C-J.3) en `notebooks/results/covariate_v6_vertexai/`
y emite un unico cuadro combinado (todos los T lado a lado en una sola hoja)
mas, por paridad, tres fragmentos por T.

Particularidades del panel covariado:

  * C-F (cointegracion ADL-ECM) tiene TRES modelos por experimento; se compara
    Chronos contra ARDL-ECM (el modelo correctamente especificado), ignorando
    el SARIMAX(1,1,0) que tambien aparece.
  * C-E (VARX bivariado) tiene metricas marginales por variable (columna `var`
    con valores 0/1, sin fila conjunta); se promedian las dos variables para
    dejar una fila por experimento.

El coloreado y los triangulos siguen la convencion validada en la tesis:
fondo purpura + triangulo si gana Chronos (cociente < 1), fondo azul +
triangulo si gana el clasico (cociente > 1), neutro si la diferencia con 1
es menor al 0.5%. Para `bias` el cociente se computa sobre valores absolutos.

Uso:
    python scripts/build_full_covariate_summary.py
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
RESULTS_DIR = ROOT / "notebooks" / "results" / "covariate_v6_vertexai"
OUTPUT_DIR = ROOT / "entrega" / "tesis" / "output"
NOTEBOOK = ROOT / "notebooks" / "experimentos_covariables_v6_cloud.ipynb"

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

# Orden de los bloques C-A .. C-J.
BLOCK_ORDER = ["C-A", "C-B", "C-C", "C-D", "C-E", "C-F", "C-G", "C-H", "C-I", "C-J"]
EXP_RE = re.compile(r"^exp_(C-[A-J])_(\d+)_T(\d+)_R" + str(R) + r"\.csv$")

# Resumen en lenguaje llano de cada bloque, para el renglon de cabecera que
# antecede a sus experimentos. Tomado de los encabezados "## Bloque C-X --".
BLOCK_SUMMARY = {
    "C-A": r"Bloque ARIMAX(1) univariado --- dise\~no factorial "
           r"$\beta$ (fuerza de la covariable) $\times\ \rho_x$ (persistencia de $X$)",
    "C-B": r"Bloque ARIMAX con din\'amica AR m\'as rica --- "
           r"$\phi \in \{0.9,\, -0.6,\, 0.99,\, 0.3\}$",
    "C-C": r"Bloque ARIMAX con m\'ultiples covariables --- asim\'etricas, "
           r"balanceadas, filtrado, signos opuestos, d\'ebiles",
    "C-D": r"Bloque ARIMAX con volatilidad condicional (GARCH) --- canales de "
           r"media y/o varianza",
    "C-E": r"Bloque VARX bivariado --- promedio de las dos variables "
           r"end\'ogenas $Y_1, Y_2$",
    "C-F": r"Bloque Cointegraci\'on ADL-ECM --- Chronos comparado contra el "
           r"modelo correcto ARDL-ECM",
    "C-G": r"Bloque ARIMAX con tendencia determin\'istica lineal --- "
           r"$\delta \in \{0.05,\, 0.10,\, 0.20\}$",
    "C-H": r"Bloque SARIMAX estacional con covariable --- $s \in \{4,\, 12\}$",
    "C-I": r"Bloque Relaci\'on se\~nal/ruido --- variaci\'on de $\sigma_y$ y $\sigma_x$",
    "C-J": r"Bloque Interacci\'on din\'amica AR $\times$ persistencia de $X$",
}


def _block_header_row(block_letter: str, total_cols: int) -> str:
    """Renglon de cabecera (gris, negrita) que resume un bloque y abarca toda
    la fila."""
    desc = BLOCK_SUMMARY.get(block_letter, f"Bloque {block_letter}")
    return (rf"\multicolumn{{{total_cols}}}{{l}}{{\cellcolor{{gray!15}}"
            rf"\textbf{{{desc}}}}} \\")


# ---------------------------------------------------------------------------
# Descripciones del DGP (etiquetas de fila) extraidas de la notebook.
# ---------------------------------------------------------------------------

_GREEK = {
    "rho": r"\rho", "theta": r"\theta", "delta": r"\delta", "alpha": r"\alpha",
    "phi": r"\phi", "beta": r"\beta", "sigma": r"\sigma", "gamma": r"\gamma",
}
_GREEK_ALT = "|".join(_GREEK)


def latexify_desc(s: str) -> str:
    """Convierte la descripcion cruda del DGP covariado a LaTeX legible.

    Maneja flechas de causalidad, simbolos griegos (con o sin subindice y con
    signo), Phi mayuscula (AR estacional), s=n/k=n y subindices residuales.
    """
    # Flechas de causalidad: '->' -> \rightarrow
    s = s.replace("->", r"$\rightarrow$")
    # Griega con subindice y valor (con signo): rho_x=0.95, sigma_y=2.0, alpha_ecm=-0.3
    s = re.sub(rf"\b({_GREEK_ALT})_([A-Za-z]+)\s*=\s*(-?[0-9.]+)",
               lambda m: r"$%s_{\mathrm{%s}}=%s$" % (_GREEK[m.group(1)], m.group(2), m.group(3)), s)
    # Griega con subindice sin valor: rho_x, sigma_y
    s = re.sub(rf"\b({_GREEK_ALT})_([A-Za-z]+)\b",
               lambda m: r"$%s_{\mathrm{%s}}$" % (_GREEK[m.group(1)], m.group(2)), s)
    # Griega minuscula con valor (con signo): phi=-0.6, beta=0.8
    s = re.sub(rf"\b({_GREEK_ALT})\s*=\s*(-?[0-9.]+)",
               lambda m: r"$%s=%s$" % (_GREEK[m.group(1)], m.group(2)), s)
    # Phi mayuscula (AR estacional), con o sin valor. El lookbehind evita
    # re-procesar el \Phi que la primera sustitucion ya dejo escapado.
    s = re.sub(r"(?<!\\)\bPhi\s*=\s*(-?[0-9.]+)", lambda m: r"$\Phi=%s$" % m.group(1), s)
    s = re.sub(r"(?<!\\)\bPhi\b", r"$\\Phi$", s)
    # Operador de rezago y exponentes.
    s = re.sub(r"L\^(\d+)", lambda m: r"$L^{%s}$" % m.group(1), s)
    # s=n, k=n.
    s = re.sub(r"\b([ks])\s*=\s*(\d+)", lambda m: r"$%s=%s$" % (m.group(1), m.group(2)), s)
    # Subindices residuales _n (numericos).
    s = re.sub(r"_(\d+)", lambda m: r"\textsubscript{%s}" % m.group(1), s)
    return s


def load_descriptions() -> dict[str, str]:
    """Parsea la notebook v6 y devuelve {exp_id: descripcion_latex}.

    Cada experimento llama `verify_dgp_cov("C-A.1 -- descripcion", ...)` (o
    `verify_dgp_mv_cov(...)` para los VARX de C-E). Se toma el texto tras '--'.
    """
    if not NOTEBOOK.exists():
        print(f"[warn] notebook no encontrada ({NOTEBOOK}); se usaran IDs como etiqueta")
        return {}
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    pat_desc = re.compile(r'verify_dgp(?:_mv)?_cov\(\s*"(C-[A-J]\.\d+)\s*--\s*([^"]+)"')
    pat_md = re.compile(r'#{2,3}\s*(C-[A-J]\.\d+)\s*[-—]+\s*(.+)')
    out: dict[str, str] = {}
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code":
            for m in pat_desc.finditer(src):
                out[m.group(1)] = latexify_desc(m.group(2).strip())
        elif c.get("cell_type") == "markdown":
            for m in pat_md.finditer(src):
                out.setdefault(m.group(1), latexify_desc(m.group(2).strip()))
    return out


# ---------------------------------------------------------------------------
# Carga y agregacion de los CSV.
# ---------------------------------------------------------------------------

def discover_experiments() -> list[tuple[str, int]]:
    found: set[tuple[str, int]] = set()
    for p in RESULTS_DIR.glob("exp_*_R*.csv"):
        m = EXP_RE.match(p.name)
        if m:
            found.add((m.group(1), int(m.group(2))))
    return sorted(found, key=lambda bi: (BLOCK_ORDER.index(bi[0])
                                         if bi[0] in BLOCK_ORDER else 99, bi[1]))


def load_csv(block: str, idx: int, T: int) -> pd.DataFrame | None:
    """Carga un experimento, horizontes enteros. Si trae columna `var` (C-E,
    bivariado) promedia las variables marginales -> una fila por (modelo,h)."""
    path = RESULTS_DIR / f"exp_{block}_{idx}_T{T}_R{R}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["horizon"].astype(str).str.match(r"^\d+$")].copy()
    if df.empty:
        return None
    df["horizon"] = df["horizon"].astype(int)
    if "var" in df.columns:
        df = df[df["var"] >= 0]
        df = df.groupby(["model", "horizon"], as_index=False).mean(numeric_only=True)
    return df


def chronos_row_name(models: list[str]) -> str:
    for m in models:
        if "chronos" in m.lower():
            return m
    raise KeyError(f"No se encontro fila Chronos entre: {models}")


def classical_row_name(models: list[str]) -> str:
    """Clasico de referencia. En C-F hay dos no-Chronos (ARDL-ECM y SARIMAX);
    se prefiere ARDL-ECM (modelo correcto de cointegracion)."""
    non_chronos = [m for m in models if "chronos" not in m.lower()]
    if not non_chronos:
        raise KeyError(f"No se encontro fila clasica entre: {models}")
    for m in non_chronos:
        if "ardl" in m.lower():
            return m
    return non_chronos[0]


def aggregate_block(df: pd.DataFrame, h_lo: int, h_hi: int) -> dict[str, dict[str, float]]:
    sub = df[(df["horizon"] >= h_lo) & (df["horizon"] <= h_hi)]
    out: dict[str, dict[str, float]] = {}
    for model, grp in sub.groupby("model"):
        out[model] = {m: grp[m].mean() for m in METRICS if m in grp.columns}
    return out


# Degradado en el FONDO de toda la celda (misma convencion que el univariado).
CHRONOS_RGB = (0x96, 0x72, 0xB6)   # violeta -> Chronos mejor (cociente < 1)
CLASSIC_RGB = (0x4C, 0x72, 0xB0)   # azul    -> clasico mejor (cociente > 1)
SAT_CAP = 2.5
BG_T_MIN = 0.08
BG_T_MAX = 0.30


def _blend_white(rgb: tuple[int, int, int], t: float) -> str:
    """Mezcla blanco -> rgb con factor t in [0,1]; devuelve hex 'RRGGBB'."""
    r, g, b = (round(255 + (c - 255) * t) for c in rgb)
    return f"{r:02X}{g:02X}{b:02X}"


def _fmt_ratio(ratio: float) -> str:
    """2 decimales en el rango normal; notacion compacta (p. ej. '1.1e3') en los
    extremos, para no inflar el ancho de columna en casos degenerados."""
    a = abs(ratio)
    if a != 0 and (a >= 100 or a < 1e-2):
        mant, exp = f"{ratio:.1e}".split("e")
        return f"{mant}e{int(exp)}"
    return f"{ratio:.2f}"


def ratio_cell(ratio: float) -> str:
    """Celda con fondo en degradado claro segun la magnitud del cociente.

    Violeta si Chronos gana (cociente < 1), azul si gana el clasico (> 1).
    Zona neutra a +-0.5% de 1. El triangulo (color pleno) marca el ganador.
    """
    if pd.isna(ratio):
        return r"---\,\phantom{\winC}"
    val = _fmt_ratio(ratio)
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
# Render del cuadro por T.
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
    lines.append(rf"\captionof{{table}}{{}}\label{{tab:cov_full_T{T}}}")
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
        if block_letter != prev_block:
            if prev_block is not None:
                lines.append(r"\midrule")
            lines.append(_block_header_row(block_letter, total_cols))
            prev_block = block_letter

        exp_id = f"{block_letter}.{idx}"
        df = load_csv(block_letter, idx, T)
        label = descriptions.get(exp_id, exp_id)
        if df is None or df.empty:
            cells = [ratio_cell(float("nan"))] * (n_blocks * n_metrics)
            lines.append(label + " & " + " & ".join(cells) + r" \\")
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
    """Un unico cuadro con todos los T lado a lado (una sola hoja).

    Encabezado en tres niveles: T -> bloque H (Corto/Medio/Largo) -> metricas
    (Bias/Var/RMSE/CRPS). Una fila por experimento.
    """
    layout = [(T, valid_blocks_for_T(T)) for T in T_LIST]
    n_metrics = len(METRICS)

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
    lines.append(r"\captionof{table}{}\label{tab:cov_full_combinado}")
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
        if block_letter != prev_block:
            if prev_block is not None:
                lines.append(r"\midrule")
            lines.append(_block_header_row(block_letter, total_cols))
            prev_block = block_letter

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
    r"\input{cov_full_summary_T50.tex}" "\n\n"
    r"\clearpage" "\n"
    r"\input{cov_full_summary_T100.tex}" "\n\n"
    r"\clearpage" "\n"
    r"\input{cov_full_summary_T200.tex}" "\n"
)

WRAPPER_BODY_COMBINED = r"\input{cov_full_summary_combined_table.tex}" "\n"


def write_wrapper(path: Path, body: str) -> None:
    path.write_text(make_document(body), encoding="utf-8")


def try_compile(wrapper_path: Path) -> Path | None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("[info] pdflatex no encontrado en PATH. Para compilar manualmente:")
        print(f"       cd {wrapper_path.parent}")
        print(f"       pdflatex -interaction=nonstopmode {wrapper_path.name}")
        return None
    cmd = [pdflatex, "--enable-installer",
           "-interaction=nonstopmode", "-halt-on-error", wrapper_path.name]
    pdf_path = wrapper_path.with_suffix(".pdf")
    # Dos pasadas: la segunda resuelve los \label de \captionof.
    for npass in (1, 2):
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
    missing = [f"{b}.{i}" for b, i in experiments if f"{b}.{i}" not in descriptions]
    print(f"[ok]  {len(experiments)} experimentos; {len(descriptions)} descripciones cargadas"
          + (f"; sin descripcion: {missing}" if missing else ""))

    # --- Version A: tres cuadros, uno por T (una hoja cada uno) ---
    for T in T_LIST:
        body = build_full_summary_for_T(T, experiments, descriptions)
        out_path = OUTPUT_DIR / f"cov_full_summary_T{T}.tex"
        header = (
            f"% Resumen completo covariado v6, T={T}.\n"
            "% Generado automaticamente por scripts/build_full_covariate_summary.py\n"
        )
        out_path.write_text(header + body, encoding="utf-8")
        print(f"[ok]   {out_path.relative_to(ROOT)}")

    wrapper_path = OUTPUT_DIR / "cov_full_summary.tex"
    write_wrapper(wrapper_path, WRAPPER_BODY)
    print(f"[ok]   {wrapper_path.relative_to(ROOT)} (wrapper 3 cuadros)")

    # --- Version B: un unico cuadro combinado con todos los T (entregable) ---
    combined_body = build_combined_summary(experiments, descriptions)
    combined_frag = OUTPUT_DIR / "cov_full_summary_combined_table.tex"
    combined_frag.write_text(
        "% Resumen covariado v6 -- vista combinada (todos los T).\n"
        "% Generado automaticamente por scripts/build_full_covariate_summary.py\n"
        + combined_body,
        encoding="utf-8",
    )
    print(f"[ok]   {combined_frag.relative_to(ROOT)}")

    combined_wrapper = OUTPUT_DIR / "cov_full_summary_combined.tex"
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
