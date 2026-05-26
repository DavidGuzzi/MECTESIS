"""Build notebooks/tesis_visuales.ipynb from a structured cell list.

Generates 4 vector PDFs + 1 summary CSV to entrega/tesis/output/ for use in
entrega/tesis/main.tex via the \\figplaceholder macro.

Outputs:
  - panel_series.pdf            : 6 series stacked, IPC in blue, rest in black
  - forecast_univariado.pdf     : 2x2 grid (AutoARIMA/AutoETS/AutoTheta/Chronos-2)
  - forecast_multivariado.pdf   : 1x2 (ChronosMultivariate + VECM(r=1))
  - forecast_covariables.pdf    : 1x2 (AutoSARIMAX + ChronosCov) with X_lagged
  - tabla_resumen_pi.csv        : consolidated metrics for inflation (pi)
"""

import nbformat as nbf
from pathlib import Path


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


CELLS = [
    md("""# Visuales para tesis — `entrega/tesis/main.tex`

Genera figuras (PDF vector) y tabla resumen (CSV) a `entrega/tesis/output/` para ser consumidas por `main.tex` vía el macro `\\figplaceholder{...}` y `\\input{tables/...}`.

**Outputs**:
- `panel_series.pdf` — 6 series macro stacked (IPC azul, resto negro, sin título).
- `forecast_univariado.pdf` — grid 2×2: AutoARIMA, AutoETS, AutoTheta, Chronos-2.
- `forecast_multivariado.pdf` — 1×2: ChronosMultivariate + VECM(r=1) sobre π.
- `forecast_covariables.pdf` — 1×2: AutoSARIMAX + ChronosCov con `X_lagged`.
- `tabla_resumen_pi.csv` — tabla consolidada de las 3 secciones (filtrada a variable = π).

Reusa `predict_all_at_last_origin` (no corre el backtest completo). Tiempo estimado: ~30-60 min en CPU."""),
    md("## 0 · Setup"),
    code("""import sys, warnings, pathlib

sys.path.insert(0, str(pathlib.Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mectesis.empirical.loaders import build_panel
from mectesis.empirical.transforms import log_diff, select_optimal_lags, apply_lags
from mectesis.empirical import describe as desc
from mectesis.empirical.backtest import predict_all_at_last_origin
from mectesis.empirical.autoselect import (
    AutoARIMAModel, AutoETSModel, AutoThetaModel, AutoSARIMAXModel,
)
from mectesis.models.var_model import VARModel, VECMModel
from mectesis.models.chronos import ChronosModel
from mectesis.models.chronos_multivariate import ChronosMultivariateModel
from mectesis.models.chronos_covariate import ChronosCovariateModel

START, END = "2016-12-01", "2026-05-01"
INITIAL_WINDOW = 72
HORIZONS = [1, 3, 6, 12]

ROOT = pathlib.Path.cwd().parent
OUTPUT_DIR = ROOT / "entrega" / "tesis" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

# rcParams para calidad tesis
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # Tipografía: serif tipo Computer Modern para coincidir con main.tex (pdflatex default).
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,    # TrueType embebido (compatible con LaTeX)
    "ps.fonttype": 42,
})

# Paleta consistente: Chronos (foundation) en violeta, clásicos en azul.
CHRONOS_COLOR  = "#9672B6"
CLASSIC_COLOR  = "#4C72B0"
MODEL_COLORS = {
    "AutoARIMA":   CLASSIC_COLOR,
    "AutoETS":     CLASSIC_COLOR,
    "AutoTheta":   CLASSIC_COLOR,
    "VECM(r=1)":   CLASSIC_COLOR,
    "AutoSARIMAX": CLASSIC_COLOR,
    "Chronos-2":   CHRONOS_COLOR,
}

# Pipeline Chronos compartido entre las 3 secciones (uni, multi, cov)
chronos_pipeline = ChronosModel(device="cpu")"""),
    code("""panel = build_panel(start=START, end=END)
print(f"Panel: {panel.shape}, {panel.index.min().date()} -> {panel.index.max().date()}")
print(panel.columns.tolist())
panel.head()"""),
    md("""## 1 · Panel de series (`panel_series.pdf`)

6 series macro stacked: inflación (`ipc`) en **azul**, resto en **negro**. Sin título (va en `\\caption` del .tex)."""),
    code("""# Custom plot: no usamos desc.plot_panel porque queremos colores específicos.
# Grilla 3x2 (3 filas, 2 columnas) — 6 paneles, sin celdas vacías.
COL_ORDER = ["ipc", "tpm", "badlar", "tcm", "m2", "rem"]
LABEL_MAP = {
    "ipc":    r"$\\pi_t$ (% mensual)",
    "tpm":    "TPM (%)",
    "badlar": "BADLAR (%)",
    "tcm":    "TCM (AR$/USD)",
    "m2":     "M2 (var. % mensual)",
    "rem":    "REM (% mensual)",
}

fig, axes = plt.subplots(
    nrows=3, ncols=2, sharex=True,
    figsize=(10, 6.5), constrained_layout=False,
)
axes_flat = axes.flatten()
for ax, col in zip(axes_flat, COL_ORDER):
    color = "#1f4e8c" if col == "ipc" else "k"  # azul saturado para inflación
    lw = 1.2 if col == "ipc" else 0.9
    ax.plot(panel.index, panel[col], color=color, linewidth=lw)
    ax.set_ylabel(LABEL_MAP.get(col, col), fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.tick_params(axis="y", labelsize=7)
    ax.margins(x=0.01)
# Eje x sólo visible en la fila inferior (sharex=True ya oculta arriba)
for ax in axes_flat[-2:]:
    ax.tick_params(axis="x", labelsize=8)
# Alinear y-labels por columna
fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])
fig.tight_layout(h_pad=0.4, w_pad=1.0)

out = OUTPUT_DIR / "panel_series.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()"""),
    md("## 2 · Forecasts univariados (`forecast_univariado.pdf`)\n\nGrid 2×2: AutoARIMA, AutoETS, AutoTheta, Chronos-2 forecasting π en el último origen."),
    code("""pi = panel["ipc"].rename("pi")

uni_factories = {
    "AutoARIMA": lambda: AutoARIMAModel(season_length=12),
    "AutoETS":   lambda: AutoETSModel(season_length=12),
    "AutoTheta": lambda: AutoThetaModel(season_length=12),
    "Chronos-2": lambda: ChronosModel(device="cpu"),
}
uni_forecasts = predict_all_at_last_origin(
    factories=uni_factories,
    y=pi, horizons=HORIZONS, initial_window=INITIAL_WINDOW,
)
origin = uni_forecasts["AutoARIMA"]["origin_idx"]
print(f"Origen último: t={pi.index[origin].date()}, h_max={max(HORIZONS)}")

fig = desc.plot_forecast_grid(
    pi, uni_forecasts,
    origin_idx=origin, horizon=max(HORIZONS),
    ncols=2, figsize_per=(5.5, 3.0), title="",
    colors=MODEL_COLORS,
)
out = OUTPUT_DIR / "forecast_univariado.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()"""),
    md("""## 3 · Forecasts multivariados sobre π (`forecast_multivariado.pdf`)

1×2: **ChronosMultivariate + VECM(r=1)**. VAR(1) excluido por mala performance (RMSE h=12 = 20.22 vs ~6-8 de los otros)."""),
    code("""# Sistema endógeno (mismo que v2)
Y = pd.concat({
    "pi":       panel["ipc"],
    "dlog_tcm": log_diff(panel["tcm"], 100),
    "m2":       panel["m2"],
    "d_badlar": panel["badlar"].diff(),
}, axis=1).dropna()
print(f"Y: {Y.shape}, columnas: {Y.columns.tolist()}")

multi_factories_visual = {
    "VECM(r=1)": lambda: VECMModel(coint_rank=1, k_ar_diff=1),
    "Chronos-2": lambda: ChronosMultivariateModel(chronos_pipeline),
}
multi_forecasts = predict_all_at_last_origin(
    factories=multi_factories_visual,
    y=Y, horizons=HORIZONS, initial_window=INITIAL_WINDOW,
)
origin_m = multi_forecasts["VECM(r=1)"]["origin_idx"]
print(f"Origen multi: t={Y.index[origin_m].date()}")

# variable_idx=0 -> 'pi' (primera columna en Y)
fig = desc.plot_forecast_grid(
    Y["pi"], multi_forecasts,
    origin_idx=origin_m, horizon=max(HORIZONS),
    ncols=2, figsize_per=(5.5, 3.0), title="",
    variable_idx=0,
    colors=MODEL_COLORS,
)
out = OUTPUT_DIR / "forecast_multivariado.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()"""),
    md("""## 4 · Forecasts con covariables exógenas (`forecast_covariables.pdf`)

1×2: **AutoSARIMAX + ChronosCov** sobre π con `X_lagged` (lags óptimos data-driven, mismo procedimiento que v2 Sec. 3)."""),
    code("""pi_cov = panel["ipc"].rename("pi")
X_raw = pd.concat({
    "dlog_tcm": log_diff(panel["tcm"], 100),
    "rem":      panel["rem"],
    "badlar":   panel["badlar"],
}, axis=1).reindex(pi_cov.index).dropna()
pi_cov = pi_cov.loc[X_raw.index]
gpanel = pd.concat([pi_cov.rename("pi"), X_raw], axis=1).dropna()

lag_map = select_optimal_lags(
    y=pi_cov.rename("pi"), X=X_raw,
    max_lag=12, min_lag=1,
    granger_panel=gpanel, alpha=0.10, granger_maxlag=6,
)
print(f"lag_map: {lag_map}")

X_lagged = apply_lags(X_raw, lag_map).dropna()
pi_lag = pi_cov.loc[X_lagged.index]
print(f"X_lagged: {X_lagged.shape}, pi_lag: {pi_lag.shape}")

cov_factories = {
    "AutoSARIMAX": lambda: AutoSARIMAXModel(season_length=12),
    "Chronos-2":   lambda: ChronosCovariateModel(
        chronos_pipeline,
        n_covariates=X_lagged.shape[1],
        cov_names=list(X_lagged.columns),
    ),
}
cov_forecasts = predict_all_at_last_origin(
    factories=cov_factories,
    y=pi_lag, X=X_lagged, horizons=HORIZONS, initial_window=INITIAL_WINDOW,
)
origin_c = cov_forecasts["AutoSARIMAX"]["origin_idx"]
print(f"Origen cov: t={pi_lag.index[origin_c].date()}")

fig = desc.plot_forecast_grid(
    pi_lag, cov_forecasts,
    origin_idx=origin_c, horizon=max(HORIZONS),
    ncols=2, figsize_per=(5.5, 3.0), title="",
    colors=MODEL_COLORS,
)
out = OUTPUT_DIR / "forecast_covariables.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()"""),
    md("""## 5 · Tabla resumen sobre π (`tabla_resumen_pi.csv`)

Consolidación de las 3 secciones, filtrada a `variable = π`:
- Univariado: AutoARIMA, AutoETS, AutoTheta, Chronos-2 (4 filas).
- Multivariado: ChronosMultivariate, VECM(r=1) (2 filas; VAR(1) excluido).
- Covariables: AutoSARIMAX, ChronosCov (2 filas).

Total: 8 filas × (4 métricas × 4 horizontes) = 8 × 16. El formato es wide; el ajuste a booktabs/colortbl en `.tex` se hace en `scripts/build_thesis_tables.py` u otro script de empaquetado."""),
    code("""# Cargar resultados del run v2 (debe haber corrido antes).
RESULTS_DIR = ROOT / "results" / "empirical"
tabla_v2 = pd.read_csv(RESULTS_DIR / "tabla_v2_long.csv")
print(f"tabla_v2_long: {tabla_v2.shape}, columnas: {tabla_v2.columns.tolist()}")
tabla_v2.head()"""),
    code("""# Filtrar: univariado/cov tienen 'variable' NaN; multi tiene 'pi' explícito.
mask_uni_cov = tabla_v2["seccion"].isin(["univariada", "covariadas_lagged"])
mask_multi_pi = (tabla_v2["seccion"] == "multivariada") & (tabla_v2["variable"] == "pi")
mask_multi_keep = tabla_v2["model"].isin(["ChronosMultivariate", "VECM(r=1)"])

sub = pd.concat([
    tabla_v2[mask_uni_cov],
    tabla_v2[mask_multi_pi & mask_multi_keep],
], ignore_index=True)
print(f"Filas seleccionadas: {sub.shape[0]}  modelos: {sub['model'].unique().tolist()}")
print(f"Secciones: {sub['seccion'].unique().tolist()}")

# Pivot a wide: rows=(seccion, model), cols=(metric, horizon)
metrics = ["rmse", "mae", "crps", "mase"]
sub_long = sub.melt(
    id_vars=["seccion", "model", "horizon"],
    value_vars=metrics,
    var_name="metric", value_name="value",
)
wide = sub_long.pivot_table(
    index=["seccion", "model"],
    columns=["metric", "horizon"],
    values="value",
)
# Orden de columnas: métricas en orden definido, horizontes ascendentes
wide = wide.reindex(columns=pd.MultiIndex.from_product([metrics, HORIZONS], names=["metric", "horizon"]))

# Orden de filas: secciones uni -> multi -> cov, modelos en orden lógico
section_order = ["univariada", "multivariada", "covariadas_lagged"]
model_order = {
    "univariada":        ["AutoARIMA", "AutoETS", "AutoTheta", "Chronos-2"],
    "multivariada":      ["VECM(r=1)", "ChronosMultivariate"],
    "covariadas_lagged": ["AutoSARIMAX", "ChronosCov"],
}
ordered_idx = []
for sec in section_order:
    for m in model_order[sec]:
        if (sec, m) in wide.index:
            ordered_idx.append((sec, m))
wide = wide.loc[ordered_idx]

print(f"\\nTabla resumen: {wide.shape}")
out = OUTPUT_DIR / "tabla_resumen_pi.csv"
wide.to_csv(out)
print(f"Saved: {out}")
wide.round(3)"""),
    md("""## 6 · Verificación

Resumen de archivos generados en `entrega/tesis/output/`."""),
    code("""for fname in ["panel_series.pdf", "forecast_univariado.pdf",
              "forecast_multivariado.pdf", "forecast_covariables.pdf",
              "tabla_resumen_pi.csv"]:
    p = OUTPUT_DIR / fname
    status = "OK" if p.exists() else "MISSING"
    size_kb = p.stat().st_size / 1024 if p.exists() else 0
    print(f"  [{status}] {fname:30s} {size_kb:8.1f} KB")"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "tesis_visuales.ipynb"
    nbf.write(nb, out)
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
