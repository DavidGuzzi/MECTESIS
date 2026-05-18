"""Genera las tablas de resultados de los tres bloques de la tesis.

Lee los CSV producidos por el Monte Carlo (univariado, multivariado y con
covariables) y emite fragmentos LaTeX con el formato cromatico de la tesis:
filas Chronos-2 y Clasico apareadas por T, columnas en bloques de horizonte
(Corto / Medio / Largo) con las metricas (Bias, Var, RMSE, CRPS) repetidas
dentro de cada bloque. Color de fondo en ambas celdas del par segun el
ganador de la metrica + triangulo sobre el valor ganador para que la lectura
sobreviva a impresion B&N y daltonismo.

Para los CSV multivariados (que incluyen columna `var` con el indice de
variable y filas con `var=-1` que contienen agregados globales), se filtran
las filas `var=-1` y se promedian las metricas sobre las variables antes de
agregar por bloque de horizonte.

Uso:
    python scripts/build_thesis_tables.py

Si falta algun CSV se imprime un warning y se preserva el placeholder existente.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "notebooks" / "results" / "univariate_vertexai"
RESULTS_DIR_MULTI = ROOT / "notebooks" / "results" / "multivariate_vertexai"
RESULTS_DIR_COV = ROOT / "notebooks" / "results" / "covariate_vertexai"
TABLES_DIR = ROOT / "entrega" / "tesis" / "tables"

T_LIST = [25, 50, 100, 200]
R = 500

BLOCKS = [
    ("Corto",  1,  6),
    ("Medio",  7, 18),
    ("Largo", 19, 24),
]

H_BY_T = {25: 6, 50: 18, 100: 24, 200: 24}

METRICS = ["bias", "variance", "rmse", "crps"]
METRIC_LABELS = {
    "bias":     r"Bias",
    "variance": r"Var",
    "rmse":     r"RMSE",
    "crps":     r"CRPS",
}


@dataclass
class ExpConfig:
    exp_id: str           # ID descriptivo (e.g. "A.4")
    csv_prefix: str       # prefijo del CSV (e.g. "A_4")
    classical_label: str  # como mostrarlo en la tabla LaTeX
    short_label: str      # etiqueta breve para la tabla de sintesis


EXPERIMENTS = {
    "4_1": ExpConfig("A.4",  "A_4",  r"ARIMA(2,0,0)",                          r"AR(2)"),
    "4_2": ExpConfig("B.26", "B_26", r"ARIMA(1,0,0)$+$trend",                  r"AR(1) $+$ tendencia"),
    "4_3": ExpConfig("C.2",  "C_2",  r"RW $+$ drift",                          r"RW $+$ drift"),
    "4_4": ExpConfig("D.4",  "D_4",  r"AR(1)$+$GARCH(1,1)",                    r"AR(1) $+$ GARCH"),
    "4_5": ExpConfig("E.6",  "E_6",  r"ETS(A,A,A) $s{=}12$",                   r"ETS(A,A,A)"),
    "4_6": ExpConfig("F.4",  "F_4",  r"SARIMA $(1,0,0)\times(1,0,0)_{12}$",    r"SARIMA $s{=}12$"),
    "4_7": ExpConfig("G.3",  "G_3",  r"AR(1) lineal",                          r"LSTAR(1)"),
}

EXPERIMENTS_MULTI = {
    "5_1": ExpConfig("M-A.2", "M-A_2", r"VAR(1)",                                r"VAR(1) bivariado"),
    "5_2": ExpConfig("M-C.5", "M-C_5", r"VAR(1) $k{=}6$",                        r"VAR(1) $k{=}6$"),
    "5_3": ExpConfig("M-E.1", "M-E_1", r"VECM($r{=}1$)",                         r"VECM"),
}

EXPERIMENTS_COV = {
    "6_1": ExpConfig("C-A.1", "C-A_1", r"SARIMAX(1,0,0)$+X$",                    r"SARIMAX $+X$ fuerte"),
    "6_2": ExpConfig("C-D.1", "C-D_1", r"SARIMAX(1,0,0)$+X$",                    r"SARIMAX $+X$ con GARCH"),
    "6_3": ExpConfig("C-H.4", "C-H_4", r"SARIMAX$(1,0,0)(1,0,0)_{12}+X$",        r"SARIMAX estac. $+X$"),
}


def load_csv(csv_prefix: str, T: int, data_dir: Path = RESULTS_DIR) -> pd.DataFrame | None:
    path = data_dir / f"exp_{csv_prefix}_T{T}_R{R}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["horizon"].astype(str).str.match(r"^\d+$")].copy()
    df["horizon"] = df["horizon"].astype(int)
    # En CSV multivariados existe una columna `var` con un indice de variable
    # (0..k-1) y filas con var=-1 que contienen metricas agregadas (trace_msfe,
    # avg_crps). Descartamos las filas var=-1 para que el promedio sobre
    # variables se haga solo con las metricas estandar.
    if "var" in df.columns:
        df = df[df["var"].astype(str).str.match(r"^\d+$|^-?\d+$")].copy()
        df["var"] = df["var"].astype(int)
        df = df[df["var"] >= 0].copy()
    return df


def aggregate_block(df: pd.DataFrame, h_lo: int, h_hi: int) -> dict[str, dict[str, float]]:
    """Devuelve {model: {metric: valor_promedio_en_bloque}}.

    Si el DataFrame tiene columna `var` (caso multivariado), las metricas se
    promedian primero sobre las variables del sistema y luego sobre los
    horizontes del bloque.
    """
    sub = df[(df["horizon"] >= h_lo) & (df["horizon"] <= h_hi)]
    out: dict[str, dict[str, float]] = {}
    for model, grp in sub.groupby("model"):
        out[model] = {m: grp[m].mean() for m in METRICS if m in grp.columns}
    return out


def winner(metric: str, val_chronos: float, val_class: float) -> str:
    """'C' si Chronos gana, 'T' si el clasico, '=' si empata.

    Para bias gana el modelo con menor |bias| (mas cercano a cero);
    para variance, rmse y crps gana el menor valor.
    """
    if pd.isna(val_chronos) or pd.isna(val_class):
        return "="
    if metric == "bias":
        d_c, d_t = abs(val_chronos), abs(val_class)
    else:
        d_c, d_t = val_chronos, val_class
    if abs(d_c - d_t) < 1e-12:
        return "="
    return "C" if d_c < d_t else "T"


def fmt(v: float) -> str:
    if pd.isna(v):
        return "---"
    return f"{v:.3f}"


def cell(value: str, win: str, role: str) -> str:
    """Construye una celda con color de fondo + marcador segun el ganador.

    Las celdas sin marcador reservan el mismo ancho con \\phantom{\\winC} para
    que las columnas queden alineadas vertical y horizontalmente.
    """
    if win == role:
        marker = r"\,\winC" if role == "C" else r"\,\winT"
    else:
        marker = r"\,\phantom{\winC}"
    if win == "C":
        return rf"\cellC{{{value}{marker}}}"
    if win == "T":
        return rf"\cellT{{{value}{marker}}}"
    return f"{value}{marker}"


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


def empty_cells() -> tuple[list[str], list[str]]:
    """Devuelve dos listas de 4 celdas vacias (---) para Chronos y Clasico."""
    blanks = [r"---"] * len(METRICS)
    return blanks, blanks


def build_table(cfg: ExpConfig, data_dir: Path = RESULTS_DIR) -> str:
    # Estructura: para cada T cargamos el CSV (si existe) y producimos dos filas
    # con 12 celdas cada una (4 metricas x 3 bloques H). Si no hay CSV para
    # ese T, igualmente se emite la fila con todas las celdas en "---" para
    # mantener la uniformidad estructural de las tablas.
    per_T_rows: list[tuple[int, list[str], list[str]]] = []  # (T, fila_chronos, fila_clasico)
    any_data = False
    for T in T_LIST:
        df = load_csv(cfg.csv_prefix, T, data_dir)
        if df is None:
            blanks = [r"---"] * (len(METRICS) * len(BLOCKS))
            per_T_rows.append((T, list(blanks), list(blanks)))
            continue
        any_data = True
        h_max = H_BY_T[T]
        ch_cells: list[str] = []
        cl_cells: list[str] = []
        for _, lo, hi in BLOCKS:
            if lo > h_max:
                ch_cells.extend([r"---"] * len(METRICS))
                cl_cells.extend([r"---"] * len(METRICS))
                continue
            hi_eff = min(hi, h_max)
            agg = aggregate_block(df, lo, hi_eff)
            try:
                ch_name = chronos_row_name(list(agg.keys()))
                cl_name = classical_row_name(list(agg.keys()))
            except KeyError as exc:
                print(f"[WARN] {cfg.csv_prefix} T={T}: {exc}")
                ch_cells.extend([r"---"] * len(METRICS))
                cl_cells.extend([r"---"] * len(METRICS))
                continue
            ch_vals = agg[ch_name]
            cl_vals = agg[cl_name]
            for m in METRICS:
                vc = ch_vals.get(m, float("nan"))
                vt = cl_vals.get(m, float("nan"))
                win = winner(m, vc, vt)
                ch_cells.append(cell(fmt(vc), win, "C"))
                cl_cells.append(cell(fmt(vt), win, "T"))
        per_T_rows.append((T, ch_cells, cl_cells))

    if not any_data:
        return ""

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        rf"\caption*{{Experimento {cfg.exp_id} --- bias, varianza, RMSE y CRPS por "
        r"tama\~no muestral y bloque de horizonte. El color de fondo y "
        r"$\blacktriangle$ marcan al ganador de cada m\'etrica para ese par "
        r"$(T, \text{bloque}\!-\!h)$. $R = 500$ r\'eplicas.}"
    )
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll cccc cccc cccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{4}{c}{Corto $h \in [1,6]$}"
        r"  & \multicolumn{4}{c}{Medio $h \in [7,18]$}"
        r"  & \multicolumn{4}{c}{Largo $h \in [19,24]$} \\"
    )
    lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}")
    metric_header = " & ".join([METRIC_LABELS[m] for m in METRICS])
    lines.append(rf"$T$ & Modelo & {metric_header} & {metric_header} & {metric_header} \\")
    lines.append(r"\midrule")

    for idx, (T, ch_cells, cl_cells) in enumerate(per_T_rows):
        lines.append(
            rf"\multirow{{2}}{{*}}{{{T}}} & Chronos-2 & " + " & ".join(ch_cells) + r" \\"
        )
        lines.append(
            rf"                       & {cfg.classical_label} & " + " & ".join(cl_cells) + r" \\"
        )
        if idx < len(per_T_rows) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def ratio_cell(ratio: float) -> str:
    """Construye una celda con el cociente metrica_Chronos/metrica_classical,
    coloreada segun el ganador (purpura si Chronos, azul si el clasico) y con
    marcador. Si la diferencia con 1 es menor que 0.5 %, no se colorea ni se
    marca: se rellena con \\phantom para mantener alineacion vertical.
    """
    if pd.isna(ratio):
        return r"---\,\phantom{\winC}"
    val = f"{ratio:.2f}"
    if abs(ratio - 1.0) < 5e-3:
        return rf"{val}\,\phantom{{\winC}}"
    if ratio < 1.0:
        return rf"\cellC{{{val}\,\winC}}"
    return rf"\cellT{{{val}\,\winT}}"


# Para la sintesis: que bloques H aplican a cada T.
SUMMARY_LAYOUT: list[tuple[int, list[str]]] = [
    (25,  ["Corto"]),
    (50,  ["Corto", "Medio"]),
    (100, ["Corto", "Medio", "Largo"]),
    (200, ["Corto", "Medio", "Largo"]),
]
BLOCK_RANGES = {name: (lo, hi) for name, lo, hi in BLOCKS}


def build_summary_table(
    metric: str,
    metric_label_math: str,
    experiments: dict[str, ExpConfig] = None,
    data_dir: Path = RESULTS_DIR,
    block_name: str = "univariado",
) -> str:
    """Tabla de sintesis de un bloque: cociente metric_Chronos / metric_classical
    por experimento, T y bloque-H. Filas = experimentos, columnas = (T, bloque-H).
    """
    if experiments is None:
        experiments = EXPERIMENTS
    rows_data: list[tuple[str, list[str]]] = []
    for tag, cfg in experiments.items():
        cells: list[str] = []
        for T, block_names in SUMMARY_LAYOUT:
            df = load_csv(cfg.csv_prefix, T, data_dir)
            for bname in block_names:
                lo, hi = BLOCK_RANGES[bname]
                if df is None:
                    cells.append(ratio_cell(float("nan")))
                    continue
                h_max = H_BY_T[T]
                if lo > h_max:
                    cells.append(ratio_cell(float("nan")))
                    continue
                hi_eff = min(hi, h_max)
                agg = aggregate_block(df, lo, hi_eff)
                try:
                    ch_name = chronos_row_name(list(agg.keys()))
                    cl_name = classical_row_name(list(agg.keys()))
                except KeyError:
                    cells.append(ratio_cell(float("nan")))
                    continue
                v_c = agg[ch_name].get(metric, float("nan"))
                v_t = agg[cl_name].get(metric, float("nan"))
                if pd.isna(v_c) or pd.isna(v_t) or v_t == 0:
                    cells.append(ratio_cell(float("nan")))
                    continue
                cells.append(ratio_cell(v_c / v_t))
        label = rf"{tag.replace('_', '.')} {cfg.short_label}"
        rows_data.append((label, cells))

    col_spec = "l " + " ".join(["c" * len(b) for _, b in SUMMARY_LAYOUT])

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(
        rf"\caption*{{S\'intesis del bloque {block_name} --- cociente "
        rf"${metric_label_math}_{{\mathrm{{Chronos}}}}/{metric_label_math}_{{\mathrm{{cl}}}}$ "
        r"por experimento, tama\~no muestral $T$ y bloque de horizonte. "
        r"Valores $< 1$ (fondo p\'urpura, $\blacktriangle$) favorecen a Chronos-2; "
        r"valores $> 1$ (fondo azul, $\blacktriangle$) favorecen al cl\'asico. "
        r"$R = 500$ r\'eplicas.}"
    )
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header1_parts = [r""]
    for T, blocks in SUMMARY_LAYOUT:
        header1_parts.append(rf"\multicolumn{{{len(blocks)}}}{{c}}{{$T = {T}$}}")
    lines.append(" & ".join(header1_parts) + r" \\")

    cmid_parts = []
    col_idx = 2
    for _, blocks in SUMMARY_LAYOUT:
        cmid_parts.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + len(blocks) - 1}}}")
        col_idx += len(blocks)
    lines.append("".join(cmid_parts))

    abbr = {"Corto": "C", "Medio": "M", "Largo": "L"}
    header2_parts = [r"Experimento"]
    for _, blocks in SUMMARY_LAYOUT:
        header2_parts.extend(abbr[b] for b in blocks)
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    for label, cells in rows_data:
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


BLOCK_SPECS = [
    ("univariado", EXPERIMENTS,       RESULTS_DIR,       "univ"),
    ("multivariado", EXPERIMENTS_MULTI, RESULTS_DIR_MULTI, "multi"),
    ("con covariables", EXPERIMENTS_COV,   RESULTS_DIR_COV,   "cov"),
]


# ---------------------------------------------------------------------------
# Empirical validation tables (Section 7 of the thesis).
# Independent of the Monte Carlo CSVs; consume the wide CSV produced by
# notebooks/tesis_visuales.ipynb.
# ---------------------------------------------------------------------------

EMP_OUTPUT_DIR = ROOT / "entrega" / "tesis" / "output"
EMP_CSV = EMP_OUTPUT_DIR / "tabla_resumen_pi.csv"

EMP_SECTIONS = [
    ("univariada",         "Univariado",
     ["AutoARIMA", "AutoETS", "AutoTheta", "Chronos-2"],
     {"Chronos-2"}),
    ("multivariada",       "Multivariado",
     ["VECM(r=1)", "ChronosMultivariate"],
     {"ChronosMultivariate"}),
    ("covariadas_lagged",  "Con covariables",
     ["AutoSARIMAX", "ChronosCov"],
     {"ChronosCov"}),
]

EMP_METRICS = [
    ("rmse", r"\mathrm{RMSE}"),
    ("mae",  r"\mathrm{MAE}"),
    ("crps", r"\mathrm{CRPS}"),
    ("mase", r"\mathrm{MASE}"),
]
EMP_HORIZONS = [1, 3, 6, 12]


def build_series_table() -> str:
    """Metadata table for the 6 empirical series used in Section 7."""
    rows = [
        (r"IPC (inflaci\'on)",            r"$\pi_t$",        r"INDEC",          r"\% mensual (TEM)"),
        (r"Tasa de pol\'itica monetaria", r"$\mathrm{TPM}_t$", r"BCRA",        r"\% mensual (TNA)"),
        (r"Tasa BADLAR",                  r"$\mathrm{BADLAR}_t$", r"BCRA",     r"\% mensual (TNA)"),
        (r"Tipo de cambio mayorista",     r"$\mathrm{TCM}_t$", r"BCRA",        r"ARS/USD"),
        (r"Agregado monetario M2 privado", r"$\mathrm{M2}_t$",  r"BCRA",       r"prom. m\'ovil 30 d\'ias, var. \% i.a."),
        (r"Expectativas de inflaci\'on (REM, pr\'ox. 12m)",  r"$\mathrm{REM}_t$", r"BCRA (REM)",  r"mediana, var. \% i.a."),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Series macroecon\'omicas argentinas utilizadas en la validaci\'on emp\'irica."
        r" Per\'iodo com\'un: diciembre 2016 -- abril 2026 (113 observaciones mensuales).}",
        r"\label{tab:emp_series}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Serie & S\'imbolo & Fuente & Unidad \\",
        r"\midrule",
    ]
    for nombre, simb, fuente, unidad in rows:
        lines.append(f"{nombre} & {simb} & {fuente} & {unidad} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_empirical_summary_table() -> str:
    """Consolidated table for Section 7 — 8 models x (4 metrics x 4 horizons).

    Reads `entrega/tesis/output/tabla_resumen_pi.csv` (wide format, MultiIndex
    columns (metric, horizon)). Colors the winning cell per (section, metric,
    horizon) — purple if a Chronos model wins, blue if a classical wins.
    """
    if not EMP_CSV.exists():
        return ""

    df = pd.read_csv(EMP_CSV, header=[0, 1], index_col=[0, 1])
    # Normalize horizon column header to int (CSV may parse them as strings).
    df.columns = pd.MultiIndex.from_tuples(
        [(m, int(h)) for m, h in df.columns], names=["metric", "horizon"]
    )

    def fmt_val(v: float) -> str:
        if v >= 100:   return f"{v:.0f}"
        if v >= 10:    return f"{v:.1f}"
        return f"{v:.2f}"

    body_rows: list[str] = []
    multirow_pending = True

    for sec_key, sec_label, models, chronos_models in EMP_SECTIONS:
        # Find winner per (metric, horizon) for this section.
        winners: dict[tuple[str, int], str] = {}
        for metric, _ in EMP_METRICS:
            for h in EMP_HORIZONS:
                col = (metric, h)
                vals = {m: df.loc[(sec_key, m), col] for m in models if (sec_key, m) in df.index}
                if not vals:
                    continue
                winners[(metric, h)] = min(vals, key=vals.get)

        for i, model in enumerate(models):
            if (sec_key, model) not in df.index:
                continue
            row_cells: list[str] = []
            # First col: section label (only on first model row, via \multirow)
            if i == 0:
                row_cells.append(rf"\multirow{{{len(models)}}}{{*}}{{{sec_label}}}")
            else:
                row_cells.append("")
            # Second col: model name (escape underscores)
            row_cells.append(model.replace("_", r"\_"))

            # Numeric cells: one per (metric, h). Background coloured by model
            # family (Chronos -> purple, classical -> blue). Winner of each
            # (section, metric, horizon) additionally gets the matching
            # triangle marker (\winC / \winT); losers carry an invisible
            # \phantom{\winC} for alignment.
            row_is_chronos = model in chronos_models
            macro = r"\cellC" if row_is_chronos else r"\cellT"
            for metric, _ in EMP_METRICS:
                for h in EMP_HORIZONS:
                    v = df.loc[(sec_key, model), (metric, h)]
                    txt = fmt_val(v)
                    winner_model = winners.get((metric, h))
                    if winner_model == model:
                        win_sym = r"\winC" if row_is_chronos else r"\winT"
                        row_cells.append(f"{macro}{{{txt}\\,{win_sym}}}")
                    else:
                        row_cells.append(f"{macro}{{{txt}\\,\\phantom{{\\winC}}}}")
            body_rows.append(" & ".join(row_cells) + r" \\")
        body_rows.append(r"\midrule")

    if body_rows and body_rows[-1] == r"\midrule":
        body_rows[-1] = r"\bottomrule"

    # Build header rows.
    # Top header: \multicolumn{4}{c}{RMSE} ...
    top_header = " & ".join(
        ["", ""] + [rf"\multicolumn{{4}}{{c}}{{${lbl}$}}" for _, lbl in EMP_METRICS]
    ) + r" \\"

    # cmidrules: cols 3-6, 7-10, 11-14, 15-18 (2 etiquetas + 16 numericas)
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3 + 4*i}-{6 + 4*i}}}" for i in range(len(EMP_METRICS))
    )

    sub_header = " & ".join(
        [r"Secci\'on", r"Modelo"]
        + [str(h) for _ in EMP_METRICS for h in EMP_HORIZONS]
    ) + r" \\"

    col_spec = "ll" + " ".join(["cccc"] * len(EMP_METRICS))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption*{Resumen de m\'etricas sobre $\pi_t$ (IPC mensual). "
        r"Para cada secci\'on, m\'etrica y horizonte se colorea la celda ganadora: "
        r"fondo p\'urpura ($\blacktriangle$) si gana un modelo Chronos, fondo azul si "
        r"gana un cl\'asico. Horizontes $h \in \{1, 3, 6, 12\}$.}",
        r"\label{tab:emp_resumen}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        top_header,
        cmidrules,
        sub_header,
        r"\midrule",
        *body_rows,
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Empirical (Section 7) tables — independent of the Monte Carlo CSVs.
    series_path = TABLES_DIR / "tabla_series.tex"
    series_body = build_series_table()
    series_header = (
        "% Tabla de metadatos de las series utilizadas en la validacion empirica.\n"
        "% Generada automaticamente por scripts/build_thesis_tables.py\n"
    )
    series_path.write_text(series_header + series_body, encoding="utf-8")
    print(f"[ok]   tabla_series.tex -> {series_path.relative_to(ROOT)}")

    emp_path = TABLES_DIR / "tabla_empirica_resumen.tex"
    emp_body = build_empirical_summary_table()
    if emp_body:
        emp_header = (
            "% Tabla resumen de la validacion empirica (Seccion 7).\n"
            "% Generada automaticamente por scripts/build_thesis_tables.py desde\n"
            "% entrega/tesis/output/tabla_resumen_pi.csv\n"
        )
        emp_path.write_text(emp_header + emp_body, encoding="utf-8")
        print(f"[ok]   tabla_empirica_resumen.tex -> {emp_path.relative_to(ROOT)}")
    else:
        print(f"[skip] tabla_empirica_resumen.tex: {EMP_CSV} no existe")

    for block_name, experiments, data_dir, summary_prefix in BLOCK_SPECS:
        # Tablas individuales por experimento.
        for tag, cfg in experiments.items():
            out_path = TABLES_DIR / f"exp_{tag}.tex"
            body = build_table(cfg, data_dir)
            if not body:
                print(f"[skip] {tag} ({cfg.exp_id}): no se encontraron CSV; placeholder preservado")
                continue
            header = (
                f"% Tabla del Experimento {tag.replace('_', '.')} ({cfg.exp_id}).\n"
                f"% Generada automaticamente por scripts/build_thesis_tables.py\n"
            )
            out_path.write_text(header + body, encoding="utf-8")
            print(f"[ok]   {tag} ({cfg.exp_id}) -> {out_path.relative_to(ROOT)}")

        # Tablas de sintesis: una por metrica resumen.
        summaries = [
            ("rmse", r"\mathrm{RMSE}", f"{summary_prefix}_summary.tex"),
            ("crps", r"\mathrm{CRPS}", f"{summary_prefix}_summary_crps.tex"),
        ]
        for metric, label_math, filename in summaries:
            summary_path = TABLES_DIR / filename
            summary_body = build_summary_table(
                metric, label_math, experiments=experiments,
                data_dir=data_dir, block_name=block_name,
            )
            summary_header = (
                f"% Tabla de sintesis del bloque {block_name} (cociente {metric.upper()}).\n"
                "% Generada automaticamente por scripts/build_thesis_tables.py\n"
            )
            summary_path.write_text(summary_header + summary_body, encoding="utf-8")
            print(f"[ok]   {filename} -> {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
