"""Genera las tablas de resultados del bloque univariado (Sec 4 de la tesis).

Lee los CSV producidos por el Monte Carlo y emite fragmentos LaTeX (uno por
experimento destacado) con el formato cromatico de la tesis: filas Chronos-2
y Clasico apareadas por T, columnas en bloques de horizonte (Corto / Medio /
Largo) con las metricas (Bias, Var, RMSE, CRPS) repetidas dentro de cada bloque.
Color de fondo en ambas celdas del par segun el ganador de la metrica + triangulo
sobre el valor ganador para que la lectura sobreviva a impresion B&N y daltonismo.

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


EXPERIMENTS = {
    "4_1": ExpConfig("A.4",  "A_4",  r"ARIMA(2,0,0)"),
    "4_2": ExpConfig("B.26", "B_26", r"ARIMA(1,0,0)$+$trend"),
    "4_3": ExpConfig("C.2",  "C_2",  r"RW $+$ drift"),
    "4_4": ExpConfig("D.4",  "D_4",  r"AR(1)$+$GARCH(1,1)"),
    "4_5": ExpConfig("E.6",  "E_6",  r"ETS(A,A,A) $s{=}12$"),
    "4_6": ExpConfig("F.4",  "F_4",  r"SARIMA $(1,0,0)\times(1,0,0)_{12}$"),
    "4_7": ExpConfig("G.3",  "G_3",  r"AR(1) lineal"),
}


def load_csv(csv_prefix: str, T: int) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"exp_{csv_prefix}_T{T}_R{R}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["horizon"].astype(str).str.match(r"^\d+$")].copy()
    df["horizon"] = df["horizon"].astype(int)
    return df


def aggregate_block(df: pd.DataFrame, h_lo: int, h_hi: int) -> dict[str, dict[str, float]]:
    """Devuelve {model: {metric: valor_promedio_en_bloque}}."""
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
    """Construye una celda con color de fondo + marcador segun el ganador."""
    marker = ""
    if win == role:
        marker = r"\,\winC" if role == "C" else r"\,\winT"
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


def build_table(cfg: ExpConfig) -> str:
    # Estructura: para cada T cargamos el CSV (si existe) y producimos dos filas
    # con 12 celdas cada una (4 metricas x 3 bloques H).
    per_T_rows: list[tuple[int, list[str], list[str]]] = []  # (T, fila_chronos, fila_clasico)
    any_data = False
    for T in T_LIST:
        df = load_csv(cfg.csv_prefix, T)
        if df is None:
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
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        rf"\caption*{{Experimento {cfg.exp_id} --- bias, varianza, RMSE y CRPS por "
        r"tama\~no muestral y bloque de horizonte. El color de fondo y "
        r"$\blacktriangle$ marcan al ganador de cada m\'etrica para ese par "
        r"$(T, \text{bloque}\!-\!h)$. $R = 500$ r\'eplicas.}"
    )
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
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for tag, cfg in EXPERIMENTS.items():
        out_path = TABLES_DIR / f"exp_{tag}.tex"
        body = build_table(cfg)
        if not body:
            print(f"[skip] {tag} ({cfg.exp_id}): no se encontraron CSV; placeholder preservado")
            continue
        header = (
            f"% Tabla del Experimento {tag.replace('_', '.')} ({cfg.exp_id}).\n"
            f"% Generada automaticamente por scripts/build_thesis_tables.py\n"
        )
        out_path.write_text(header + body, encoding="utf-8")
        print(f"[ok]   {tag} ({cfg.exp_id}) -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
