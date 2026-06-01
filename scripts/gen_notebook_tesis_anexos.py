"""Build notebooks/tesis_anexos_simulaciones.ipynb from a structured cell list.

Genera 3 PDFs en entrega/tesis/output/ para los anexos de la tesis,
mostrando, para una seleccion curada de DGPs Monte Carlo, los fan-plots
overlay de Chronos-2 vs. el mejor metodo clasico (estilo plot_forecast_fan).

Outputs:
  - anexo_simulaciones_univariadas.pdf   : 8 DGPs (4x2) cubriendo bloques A-G
  - anexo_simulaciones_multivariadas.pdf : 6 DGPs (3x2), cada uno con 2 vars apiladas
  - anexo_simulaciones_covariadas.pdf    : 6 DGPs (3x2), cada uno con target + exog apilados
"""

import nbformat as nbf
from pathlib import Path


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


CELLS = [
    md("""# Anexos de simulaciones --- `entrega/tesis/main.tex`

Genera 3 figuras PDF para el anexo de la tesis. Cada figura compara, sobre una
seleccion curada de DGPs Monte Carlo, el pronostico de **Chronos-2** contra el
**mejor metodo clasico** del bloque. Estilo: `plot_forecast_fan` (historia +
realizado + media de cada modelo + bandas 80/95 %).

**Outputs en `entrega/tesis/output/`**:
- `anexo_simulaciones_univariadas.pdf` --- 8 DGPs (grid 4x2): 1 por bloque A-G + extra A.4
- `anexo_simulaciones_multivariadas.pdf` --- 6 DGPs bivariados (3x2), cada uno con 2 sub-paneles (una variable por panel)
- `anexo_simulaciones_covariadas.pdf` --- 6 DGPs (3x2), cada uno con target + exogena apilados

Requisitos: el modelo `amazon/chronos-2` debe estar cacheado en `~/.cache/huggingface`.
Tiempo estimado: ~15-30 min en CPU local."""),
    md("## 0 · Setup"),
    code("""import os
# Modo offline para HuggingFace: usa solo el modelo Chronos-2 cacheado localmente
# (~/.cache/huggingface). Necesario porque la validacion HEAD vs huggingface.co
# falla con SSLCertVerificationError en este entorno.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys, warnings, pathlib

sys.path.insert(0, str(pathlib.Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from mectesis.empirical.describe import plot_forecast_fan
from mectesis.dgp import (
    ARpDGP, MAqDGP, ARMApqDGP, ARMApqWithTrendDGP,
    RandomWalk, AR1GARCH, DampedTrendDGP, SeasonalDGP,
    LocalTrendDGP, LocalLevelSeasonalDGP,
    SETARDGp, LSTARDGp,
    VARDGP, VECMBivariateDGP, VARGARCHDiagonalDGP,
    ARIMAX_DGP, ARIMAX2Cov_DGP, ARIMAX_GARCH_DGP, ARIMAX_TREND_DGP,
    SARIMAX_SEASONAL_DGP, ADL_ECM_DGP,
)
from mectesis.models import (
    ChronosModel, ChronosMultivariateModel, ChronosCovariateModel,
    ARIMAModel, ETSModel, SARIMAModel, ARGARCHModel, ARIMAXGARCHModel,
    SARIMAXModel, ARDLModel,
    VARModel, VECMModel, VARGARCHDiagonalModel,
)
from mectesis.models.arima_ext import ARIMAWithTrendModel
from mectesis.models.garch_model import ARIMAXGARCHModel

SEED = 3649 + 99991    # = 103640. Mismo seed efectivo que plot_simulation_v3
                       # del v5_cloud (SEED del proyecto = 3649, + offset 99991
                       # que el cloud usa para la visualizacion).
TOTAL = 100            # longitud total de la serie simulada
HORIZON = 18           # h del pronostico (H_BY_T[100]=18 en v5_cloud)
T_TRAIN = TOTAL - HORIZON   # = 82 observaciones de entrenamiento (split del v5_cloud)
HISTORY_TAIL = T_TRAIN # muestra TODA la serie de entrenamiento

ROOT = pathlib.Path.cwd().parent
OUTPUT_DIR = ROOT / "entrega" / "tesis" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

# rcParams tipograficos (serif tipo Computer Modern, coincide con main.tex)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Paleta: Chronos (foundation) violeta vs cualquier clasico azul.
CHRONOS_COLOR = "#9672B6"
CLASSIC_COLOR = "#4C72B0"


def colors_for(forecasts):
    \"\"\"Mapea cada nombre de modelo a CHRONOS_COLOR si contiene 'Chronos', si no CLASSIC_COLOR.\"\"\"
    return {
        name: (CHRONOS_COLOR if \"Chronos\" in name else CLASSIC_COLOR)
        for name in forecasts
    }

print("Cargando Chronos-2 desde cache local...")
chronos = ChronosModel(device="cpu")
print("Listo.")"""),
    md("## 1 · Helpers"),
    code("""def to_series(arr, name=\"y\"):
    return pd.Series(arr, index=pd.RangeIndex(len(arr)), name=name)


def to_frame(arr, prefix=\"y\"):
    return pd.DataFrame(
        arr,
        index=pd.RangeIndex(arr.shape[0]),
        columns=[f\"{prefix}{j+1}\" for j in range(arr.shape[1])],
    )


def _try_interval_80(model, horizon, **kwargs):
    \"\"\"Devuelve (lo80, hi80); (None, None) si el modelo no soporta intervalos.\"\"\"
    if not getattr(model, \"supports_intervals\", False):
        return None, None
    try:
        return model.forecast_intervals(horizon, level=0.80, **kwargs)
    except NotImplementedError:
        return None, None


def _to_arr(x):
    return None if x is None else np.asarray(x, dtype=float)


def fan_dict_uni(model, y_train, horizon):
    \"\"\"Fit + forecast + banda 80% para un modelo univariado.\"\"\"
    model.fit(np.asarray(y_train, dtype=float))
    mean = np.asarray(model.forecast(horizon), dtype=float)
    lo80, hi80 = _try_interval_80(model, horizon)
    return {\"mean\": mean, \"lo80\": _to_arr(lo80), \"hi80\": _to_arr(hi80)}


def fan_dict_multi(model, Y_train, horizon):
    \"\"\"Fit + forecast + banda 80% para un modelo multivariado. Cada entrada es (horizon, k).\"\"\"
    model.fit(np.asarray(Y_train, dtype=float))
    mean = np.asarray(model.forecast(horizon), dtype=float)
    lo80, hi80 = _try_interval_80(model, horizon)
    return {\"mean\": mean, \"lo80\": _to_arr(lo80), \"hi80\": _to_arr(hi80)}


def fan_dict_cov(model, y_train, X_train, X_future, horizon):
    \"\"\"Fit + forecast + banda 80% con covariables. Tolera modelos sin intervalos (e.g. ARDLModel).\"\"\"
    model.fit(np.asarray(y_train, dtype=float), X_train=np.asarray(X_train, dtype=float))
    mean = np.asarray(model.forecast(horizon, X_future=X_future), dtype=float)
    lo80, hi80 = _try_interval_80(model, horizon, X_future=X_future)
    return {\"mean\": mean, \"lo80\": _to_arr(lo80), \"hi80\": _to_arr(hi80)}


def clip_to_history(ax, y_history, k_sigma=4.0):
    \"\"\"Limita el eje y a mean(historia) +- k*std(historia) si los pronosticos divergen.\"\"\"
    mu = float(np.mean(y_history))
    sigma = float(np.std(y_history))
    span = max(k_sigma * sigma, abs(mu) * 0.1, 1.0)
    cur_lo, cur_hi = ax.get_ylim()
    new_lo = max(cur_lo, mu - span)
    new_hi = min(cur_hi, mu + span)
    if new_lo < new_hi:
        ax.set_ylim(new_lo, new_hi)


def plot_history_only(ax, y, color=\"#7f7f7f\", lw=0.9, label=None):
    \"\"\"Dibuja solo la trayectoria observada (para variables exogenas, sin pronostico).\"\"\"
    ax.plot(y.index, y.values, color=color, linewidth=lw, label=label)
    ax.axvline(T_TRAIN, color=\"grey\", linestyle=\":\", linewidth=0.7)
    ax.grid(alpha=0.3)
    ax.margins(x=0.01)


def grid_pos(i, ncols=2):
    \"\"\"Devuelve (row, col) llenando la grilla derecha-luego-izquierda, top-down.\"\"\"
    return i // ncols, (ncols - 1) - (i % ncols)


from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


def add_top_legend(fig, with_cov=False):
    # Una unica leyenda a nivel figura, encima de los paneles (no se repite por panel).
    # La 'Banda 80%' se dibuja con un swatch de dos colores (Chronos y clasico).
    band_chronos = Patch(facecolor=CHRONOS_COLOR, alpha=0.30, edgecolor=\"none\")
    band_classic = Patch(facecolor=CLASSIC_COLOR, alpha=0.30, edgecolor=\"none\")
    handles = [
        Line2D([0], [0], color=\"#1f4068\", lw=1.6),
        Line2D([0], [0], color=\"#1f4068\", lw=1.6, ls=\"--\", alpha=0.6),
        Line2D([0], [0], color=CHRONOS_COLOR, lw=2.2),
        Line2D([0], [0], color=CLASSIC_COLOR, lw=2.2),
        (band_chronos, band_classic),
    ]
    labels = [\"Observado\", \"Realizado\", \"Chronos-2\", \"Modelo clásico\", \"Banda 80%\"]
    if with_cov:
        handles.append(Line2D([0], [0], color=\"#3a3a3a\", lw=1.2))
        labels.append(\"Covariable(s)\")
    fig.legend(handles, labels, loc=\"lower center\", ncol=len(labels),
               fontsize=7, frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.1, handletextpad=0.4,
               handler_map={tuple: HandlerTuple(ndivide=None)})"""),
    md("""## 2 · Bloque univariado --- `anexo_simulaciones_univariadas.pdf`

20 DGPs del v5_cloud distribuidos en 2 paginas A4 (5 filas x 2 columnas por pagina).
Orden de llenado por pagina: derecha-luego-izquierda, top-down.
Cada panel: Chronos-2 vs el clasico oracle correctamente especificado del v5_cloud,
banda 80 percent, mismo seed (SEED+99991=103640) y mismo T_vis/H_vis (100/18) que
`plot_simulation_v3` del cloud."""),
    code("""# (id, label, dgp_factory, classical_factory) — IDs y params exactos del v5_cloud
uni_dgps_pages = [
    [  # Pagina 1: bloque A completo (10 DGPs)
        (\"A.1\",  \"AR(1) rho=0.30\",
            lambda: ARpDGP(phis=[0.3], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(1, 0, 0))),
        (\"A.2\",  \"AR(1) rho=0.90\",
            lambda: ARpDGP(phis=[0.9], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(1, 0, 0))),
        (\"A.7\",  \"AR(4) rho=0.30\",
            lambda: ARpDGP(phis=[0.3, 0.1, 0.05, 0.02], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(4, 0, 0))),
        (\"A.8\",  \"AR(4) rho=0.90\",
            lambda: ARpDGP(phis=[0.9, -0.2, 0.1, -0.05], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(4, 0, 0))),
        (\"A.9\",  r\"MA(1) $\\theta=0.30$\",
            lambda: MAqDGP(thetas=[0.3], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(0, 0, 1))),
        (\"A.10\", r\"MA(1) $\\theta=0.90$\",
            lambda: MAqDGP(thetas=[0.9], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(0, 0, 1))),
        (\"A.15\", r\"MA(4) $\\theta=0.30$\",
            lambda: MAqDGP(thetas=[0.3, 0.1, -0.05, 0.02], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(0, 0, 4))),
        (\"A.16\", r\"MA(4) $\\theta=0.90$\",
            lambda: MAqDGP(thetas=[0.9, 0.1, -0.05, 0.02], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(0, 0, 4))),
        (\"A.19\", \"ARMA(2,2) rho=0.30\",
            lambda: ARMApqDGP(phis=[0.3, 0.1], thetas=[0.1, 0.05], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(2, 0, 2))),
        (\"A.20\", \"ARMA(2,2) rho=0.90\",
            lambda: ARMApqDGP(phis=[0.9, -0.2], thetas=[0.3, -0.1], sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(2, 0, 2))),
    ],
    [  # Pagina 2: B/D/E/F/G (10 DGPs)
        (\"B.43\", r\"ARMA(2,2) rho=0.30 ($\\delta=0.1$)\",
            lambda: ARMApqWithTrendDGP(phis=[0.3, 0.1], thetas=[0.1, 0.05], delta=0.1, sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAWithTrendModel(order=(2, 0, 2), trend=\"ct\")),
        (\"B.44\", r\"ARMA(2,2) rho=0.90 ($\\delta=0.1$)\",
            lambda: ARMApqWithTrendDGP(phis=[0.9, -0.2], thetas=[0.3, -0.1], delta=0.1, sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAWithTrendModel(order=(2, 0, 2), trend=\"ct\")),
        (\"D.3\",  \"AR(1)-GARCH(1,1) baja persist.\",
            lambda: AR1GARCH(seed=SEED).simulate(TOTAL, phi=0.5, omega=0.5, alpha=0.10, beta=0.40),
            lambda: ARGARCHModel(ar_lags=1, p=1, q=1)),
        (\"D.4\",  \"AR(1)-GARCH(1,1) alta persist.\",
            lambda: AR1GARCH(seed=SEED).simulate(TOTAL, phi=0.5, omega=0.1, alpha=0.10, beta=0.85),
            lambda: ARGARCHModel(ar_lags=1, p=1, q=1)),
        (\"E.3\",  \"LLT fuerte ETS(A,A,N)\",
            lambda: LocalTrendDGP(seed=SEED).simulate(TOTAL, sigma_eps=1.0, sigma_eta=0.2, sigma_zeta=0.20, b0=0.5),
            lambda: ETSModel(trend=\"add\")),
        (\"E.5\",  \"Seasonal aditiva s=12\",
            lambda: LocalLevelSeasonalDGP(seed=SEED).simulate(TOTAL, s=12, sigma_eps=0.5, sigma_eta=0.1, sigma_zeta=0.0, sigma_omega=0.05, b0=0.0),
            lambda: ETSModel(seasonal=\"add\", seasonal_periods=12)),
        (\"F.4\",  r\"SAR(1)(1)$_{12}$ alta persist.\",
            lambda: SeasonalDGP(seed=SEED).simulate(TOTAL, phi=0.9, Phi=0.6, s=12, sigma=1.0, integrated=False),
            lambda: SARIMAModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))),
        (\"F.5\",  r\"$(1-L)(1-L^4)$ integrado\",
            lambda: SeasonalDGP(seed=SEED).simulate(TOTAL, s=4, sigma=1.0, integrated=True),
            lambda: SARIMAModel(order=(0, 1, 0), seasonal_order=(0, 1, 0, 4))),
        (\"G.2\",  \"SETAR(2;1) alta persist.\",
            lambda: SETARDGp(phi1=0.90, phi2=-0.50, threshold=0.0, delay=1, sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(1, 0, 0))),
        (\"G.3\",  \"LSTAR(1) asimetrico\",
            lambda: LSTARDGp(phi1=0.30, phi2=0.90, gamma=2.0, c=0.0, delay=1, sigma=1.0, seed=SEED).simulate(TOTAL),
            lambda: ARIMAModel(order=(1, 0, 0))),
    ],
]

out = OUTPUT_DIR / \"anexo_simulaciones_univariadas.pdf\"
with PdfPages(out) as pdf:
    for page_idx, page_dgps in enumerate(uni_dgps_pages, start=1):
        fig, axes = plt.subplots(5, 2, figsize=(6.30, 9.00))
        for i, (exp_id, label, dgp_fn, classic_factory) in enumerate(page_dgps):
            r, c = grid_pos(i, ncols=2)
            ax = axes[r, c]
            arr = np.asarray(dgp_fn(), dtype=float)
            y_full = to_series(arr)
            y_train = y_full.iloc[:T_TRAIN].values

            forecasts = {}
            try:
                forecasts[\"Chronos-2\"] = fan_dict_uni(chronos, y_train, HORIZON)
            except Exception as e:
                print(f\"[{exp_id}] Chronos fallo: {e}\")

            classic = classic_factory()
            try:
                forecasts[classic.name] = fan_dict_uni(classic, y_train, HORIZON)
            except Exception as e:
                print(f\"[{exp_id}] {classic.name} fallo: {e}\")

            plot_forecast_fan(
                y_full, forecasts,
                origin_idx=T_TRAIN, horizon=HORIZON,
                history_tail=HISTORY_TAIL,
                title=label,
                ax=ax, colors=colors_for(forecasts),
                show_legend=False,
            )
            ax.tick_params(labelsize=6)
        fig.tight_layout(h_pad=1.0, w_pad=1.5)
        add_top_legend(fig)
        pdf.savefig(fig, bbox_inches=\"tight\")
        plt.show()
        plt.close(fig)
        print(f\"Pagina {page_idx} de uni: OK\")

print(f\"Saved: {out}\")"""),
    md("""## 3 · Bloque multivariado --- `anexo_simulaciones_multivariadas.pdf`

6 DGPs bivariados (k=2) cubriendo los bloques M-A a M-F del experimento
`multivariate_v6_vertexai`. Cada celda muestra **solo Y1** (`variable_idx=0`)
con Chronos-2 (joint mode) y el VAR/VECM clasico correspondiente. Layout: 5x2
en una sola pagina A4, orden derecha-luego-izquierda."""),
    code("""# (id, label, dgp_factory, classical_factory) — params exactos del v6_cloud multi
multi_dgps = [
    (\"M-A.1\", \"VAR(1) baja interdependencia\",
        lambda: VARDGP(seed=SEED,
                       A_list=[np.array([[0.5, 0.1], [0.1, 0.5]])],
                       Sigma=np.array([[1.0, 0.3], [0.3, 1.0]])).simulate(TOTAL),
        lambda: VARModel(lags=1)),
    (\"M-A.2\", \"VAR(1) alta interdependencia\",
        lambda: VARDGP(seed=SEED,
                       A_list=[np.array([[0.4, 0.4], [0.4, 0.4]])],
                       Sigma=np.array([[1.0, 0.3], [0.3, 1.0]])).simulate(TOTAL),
        lambda: VARModel(lags=1)),
    (\"M-B.1\", \"VAR(2) baseline\",
        lambda: VARDGP(seed=SEED,
                       A_list=[np.array([[0.5, 0.2], [0.1, 0.4]]),
                               np.array([[0.1, 0.0], [0.0, 0.1]])],
                       Sigma=np.array([[1.0, 0.3], [0.3, 1.0]])).simulate(TOTAL),
        lambda: VARModel(lags=2)),
    (\"M-B.5\", \"VAR(1) cerca unit root\",
        lambda: VARDGP(seed=SEED,
                       A_list=[np.array([[0.95, 0.02], [0.02, 0.93]])],
                       Sigma=np.array([[1.0, 0.3], [0.3, 1.0]])).simulate(TOTAL),
        lambda: VARModel(lags=1)),
    (\"M-C.1\", \"VAR(1) k=3 tridiagonal\",
        lambda: VARDGP(seed=SEED,
                       A_list=[np.array([[0.5, 0.1, 0.0], [0.1, 0.5, 0.1], [0.0, 0.1, 0.5]])],
                       Sigma=np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 1.0]])).simulate(TOTAL),
        lambda: VARModel(lags=1)),
    (\"M-C.2\", \"VAR(1) k=4 tridiagonal\",
        lambda: VARDGP(seed=SEED,
                       A_list=[np.array([[0.4, 0.1, 0.0, 0.0], [0.1, 0.4, 0.1, 0.0],
                                         [0.0, 0.1, 0.4, 0.1], [0.0, 0.0, 0.1, 0.4]])],
                       Sigma=np.array([[1.0, 0.2, 0.0, 0.0], [0.2, 1.0, 0.2, 0.0],
                                       [0.0, 0.2, 1.0, 0.2], [0.0, 0.0, 0.2, 1.0]])).simulate(TOTAL),
        lambda: VARModel(lags=1)),
    (\"M-D.1\", \"VAR(1) + GARCH baseline\",
        lambda: VARGARCHDiagonalDGP(seed=SEED,
                                    A1=np.array([[0.5, 0.1], [0.1, 0.5]]),
                                    omegas=[0.1, 0.1], alphas=[0.1, 0.15], betas=[0.8, 0.75]).simulate(TOTAL),
        lambda: VARGARCHDiagonalModel(seed=SEED)),
    (\"M-D.6\", \"VAR+GARCH k=3 tridiagonal\",
        lambda: VARGARCHDiagonalDGP(seed=SEED,
                                    A1=np.array([[0.4, 0.1, 0.0], [0.1, 0.4, 0.1], [0.0, 0.1, 0.4]]),
                                    omegas=[0.1, 0.1, 0.1], alphas=[0.1, 0.1, 0.1], betas=[0.8, 0.8, 0.8]).simulate(TOTAL),
        lambda: VARGARCHDiagonalModel(seed=SEED)),
    (\"M-E.1\", \"VECM baseline ajuste medio\",
        lambda: VECMBivariateDGP(seed=SEED, alpha=[-0.4, 0.2], beta=[1.0, -1.0],
                                 Gamma1=[[0.3, 0.0], [0.0, 0.3]],
                                 Sigma=[[1.0, 0.0], [0.0, 1.0]]).simulate(TOTAL),
        lambda: VECMModel(coint_rank=1, k_ar_diff=1)),
    (\"M-E.5\", r\"VECM dinamica corta + $\\Sigma$ corr\",
        lambda: VECMBivariateDGP(seed=SEED, alpha=[-0.4, 0.2], beta=[1.0, -1.0],
                                 Gamma1=[[0.5, 0.2], [0.2, 0.5]],
                                 Sigma=[[1.0, 0.5], [0.5, 1.0]]).simulate(TOTAL),
        lambda: VECMModel(coint_rank=1, k_ar_diff=1)),
]

# Layout: 5 filas x 2 cols de DGPs, una celda por DGP (solo Y1)
fig, axes = plt.subplots(5, 2, figsize=(6.30, 9.00))

for i, (exp_id, label, dgp_fn, classic_factory) in enumerate(multi_dgps):
    r, c = grid_pos(i, ncols=2)
    ax = axes[r, c]

    Y_arr = np.asarray(dgp_fn(), dtype=float)
    Y_full = to_frame(Y_arr, prefix=\"Y\")
    Y_train = Y_full.iloc[:T_TRAIN].values

    # Chronos joint multivariate
    chronos_mv = ChronosMultivariateModel(chronos)
    try:
        f_chronos = fan_dict_multi(chronos_mv, Y_train, HORIZON)
    except Exception as e:
        print(f\"[{exp_id}] Chronos multi fallo: {e}\")
        f_chronos = None

    classic = classic_factory()
    try:
        f_classic = fan_dict_multi(classic, Y_train, HORIZON)
    except Exception as e:
        print(f\"[{exp_id}] {classic.name} fallo: {e}\")
        f_classic = None

    forecasts = {}
    if f_chronos is not None:
        forecasts[\"Chronos-2\"] = f_chronos
    if f_classic is not None:
        forecasts[classic.name] = f_classic

    plot_forecast_fan(
        Y_full.iloc[:, 0], forecasts,
        origin_idx=T_TRAIN, horizon=HORIZON,
        history_tail=HISTORY_TAIL,
        title=label,
        ax=ax, variable_idx=0, colors=colors_for(forecasts),
        show_legend=False,
    )
    ax.tick_params(labelsize=6)

fig.tight_layout(h_pad=1.0, w_pad=1.5)
add_top_legend(fig)
out = OUTPUT_DIR / \"anexo_simulaciones_multivariadas.pdf\"
fig.savefig(out, bbox_inches=\"tight\")
print(f\"Saved: {out}\")
plt.show()"""),
    md("""## 4 · Bloque covariadas --- `anexo_simulaciones_covariadas.pdf`

6 DGPs con un target univariado y una covariable exogena, cubriendo los
bloques C-A, C-C y C-H del notebook **ajustado** `covariate_v6_vertexai_ajustado`.
Cada celda tiene dos sub-paneles apilados: arriba el target Y con fan-plot
(Chronos-2 con X vs SARIMAX con xreg); abajo la exogena X1 (solo historia,
linea gris). Layout 3x2, orden derecha-luego-izquierda."""),
    code("""# (id, label, dgp_fn que devuelve dict {y, X}, classical_factory) — IDs del ajustado
cov_dgps = [
    (\"C-A.1\", r\"ARIMAX(1) $\\beta=0.8$\",
        lambda: ARIMAX_DGP(seed=SEED).simulate(TOTAL, phi=0.6, beta=0.8, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), name_suffix=\"con X\")),
    (\"C-A.3\", r\"ARIMAX(1) $\\beta=0.2$\",
        lambda: ARIMAX_DGP(seed=SEED).simulate(TOTAL, phi=0.6, beta=0.2, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), name_suffix=\"con X\")),
    (\"C-C.4\", r\"ARIMAX 2-cov signos opuestos\",
        lambda: ARIMAX2Cov_DGP(seed=SEED).simulate(TOTAL, phi=0.6, beta1=0.8, beta2=-0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), name_suffix=\"con X\")),
    (\"C-C.3\", r\"ARIMAX 2-cov filtrado ($\\beta_2=0$)\",
        lambda: ARIMAX2Cov_DGP(seed=SEED).simulate(TOTAL, phi=0.6, beta1=0.8, beta2=0.0, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), name_suffix=\"con X\")),
    (\"C-D.1\", r\"ARIMAX-GARCH solo media\",
        lambda: ARIMAX_GARCH_DGP(seed=SEED).simulate(TOTAL, phi=0.4, beta_mean=0.5, omega=0.1, alpha=0.1, beta_garch=0.75, delta_var=0.0, sigma_x=1.0, rho_x=0.7),
        lambda: ARIMAXGARCHModel(ar_lags=1, p=1, q=1)),
    (\"C-G.2\", r\"ARIMAX + tendencia fuerte ($\\delta=0.10$)\",
        lambda: ARIMAX_TREND_DGP(seed=SEED).simulate(TOTAL, phi=0.6, beta=0.5, alpha=0.0, delta=0.10, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), trend=\"ct\", name_suffix=\"con X+trend\")),
    (\"C-H.1\", r\"Estacional s=4 $\\beta=0.5$\",
        lambda: SARIMAX_SEASONAL_DGP(seed=SEED).simulate(TOTAL, s=4, phi=0.3, Phi=0.7, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), name_suffix=\"con X\")),
    (\"C-H.3\", r\"Estacional s=12 $\\beta=0.5$\",
        lambda: SARIMAX_SEASONAL_DGP(seed=SEED).simulate(TOTAL, s=12, phi=0.3, Phi=0.7, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7),
        lambda: SARIMAXModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), name_suffix=\"con X\")),
]

# Layout: 4 filas x 2 cols de DGPs, cada DGP en sub-grid (Y arriba con ratio 1.7, X1 abajo con ratio 1.0)
fig = plt.figure(figsize=(6.30, 9.40))
outer = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.30,
                 top=0.97, bottom=0.045, left=0.10, right=0.975)

for i, (exp_id, label, dgp_fn, classic_factory) in enumerate(cov_dgps):
    r, c = grid_pos(i, ncols=2)
    inner = outer[r, c].subgridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.25)

    data = dgp_fn()
    y_arr = np.asarray(data[\"y\"], dtype=float)
    X_arr = np.asarray(data[\"X\"], dtype=float)  # (T, p_x)

    y_full = to_series(y_arr, name=\"Y\")
    y_train = y_full.iloc[:T_TRAIN].values
    X_train = X_arr[:T_TRAIN]
    X_future = X_arr[T_TRAIN:T_TRAIN + HORIZON]
    p_x = X_arr.shape[1]

    # Chronos con covariables (univariado target)
    chronos_cov = ChronosCovariateModel(chronos, n_covariates=p_x,
                                        cov_names=[f\"x{j}\" for j in range(p_x)])
    forecasts = {}
    try:
        forecasts[\"Chronos-2\"] = fan_dict_cov(chronos_cov, y_train, X_train, X_future, HORIZON)
    except Exception as e:
        print(f\"[{exp_id}] Chronos cov fallo: {e}\")

    classic = classic_factory()
    try:
        forecasts[classic.name] = fan_dict_cov(classic, y_train, X_train, X_future, HORIZON)
    except Exception as e:
        print(f\"[{exp_id}] {classic.name} fallo: {e}\")

    # Sub-panel 1: target Y con fan-plot
    ax_y = fig.add_subplot(inner[0])
    plot_forecast_fan(
        y_full, forecasts,
        origin_idx=T_TRAIN, horizon=HORIZON,
        history_tail=HISTORY_TAIL,
        title=label,
        ax=ax_y, colors=colors_for(forecasts),
        show_legend=False,
    )
    ax_y.set_ylabel(\"$Y_t$\", fontsize=7)
    ax_y.tick_params(labelsize=5)

    # Sub-panel 2: exogenas (solo historia, una linea por covariable, overlay con distintos grises)
    ax_x = fig.add_subplot(inner[1])
    x_colors = [\"#3a3a3a\", \"#9a9a9a\", \"#6a6a6a\"]  # oscuro, claro, medio
    for j in range(p_x):
        xs = pd.RangeIndex(len(X_arr))
        ax_x.plot(xs, X_arr[:, j], color=x_colors[j % len(x_colors)], linewidth=0.8,
                   label=f\"$X_{{{j+1}}}$\")
    ax_x.axvline(T_TRAIN, color=\"grey\", linestyle=\":\", linewidth=0.7)
    ax_x.grid(alpha=0.3)
    ax_x.margins(x=0.01)
    ax_x.set_ylabel(\"$X_t$\", fontsize=7)
    ax_x.tick_params(labelsize=5)
    ax_x.set_xlim(*ax_y.get_xlim())

add_top_legend(fig, with_cov=True)
out = OUTPUT_DIR / \"anexo_simulaciones_covariadas.pdf\"
fig.savefig(out, bbox_inches=\"tight\")
print(f\"Saved: {out}\")
plt.show()"""),
    md("## 5 · Verificacion"),
    code("""for fname in [
    \"anexo_simulaciones_univariadas.pdf\",
    \"anexo_simulaciones_multivariadas.pdf\",
    \"anexo_simulaciones_covariadas.pdf\",
]:
    p = OUTPUT_DIR / fname
    status = \"OK\" if p.exists() else \"MISSING\"
    size_kb = p.stat().st_size / 1024 if p.exists() else 0
    print(f\"  [{status}] {fname:42s} {size_kb:8.1f} KB\")"""),
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
    out = Path(__file__).resolve().parents[1] / "notebooks" / "tesis_anexos_simulaciones.ipynb"
    nbf.write(nb, out)
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
