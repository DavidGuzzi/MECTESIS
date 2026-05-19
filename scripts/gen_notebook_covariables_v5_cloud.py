"""
Generador del notebook `experimentos_covariables_v5_cloud.ipynb`.

Fork de gen_notebook_covariables_v4_cloud.py con:
  - CRPS de Chronos calculado via estimador propio sobre cuantiles
    (regla trapezoidal del pinball loss, K=39 cuantiles uniformes)
  - Cache nuevo results/covariate_v5_vertexai/ (no toca v4)

Cubre los mismos experimentos que v4_cloud (C-A..C-H).

Uso:
    python scripts/gen_notebook_covariables_v5_cloud.py
Genera:
    notebooks/experimentos_covariables_v5_cloud.ipynb
"""
import json
from pathlib import Path


NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "experimentos_covariables_v5_cloud.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) if text else [],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) if text else [],
    }


# ────────────────────────────────────────────────────────────────────────────
# Cell 0 — title
# ────────────────────────────────────────────────────────────────────────────
TITLE = """# Experimentos con Covariables v5 Cloud (Vertex AI)

**Tesis MEC** — 25 DGPs con covariables x T in {25,50,100,200} x R=500
**Horizonte por T:** T=25->H=6 * T=50->H=18 * T=100,200->H=24
**Metricas (univariado):** Bias, Varianza, RMSE, MAE, CRPS, Cobertura/Amplitud/Winkler 80%-95%
**Metricas (multivariado):** mismas per-variable + Trace MSFE y avgCRPS conjuntos (donde el engine las expone)
**Bloques h:** Corto h=1-6 * Medio h=7-18 * Largo h=19-24
**Logging:** dual stdout + `results/covariate_v5_vertexai/run_YYYYMMDD_HHMMSS.log`
**Resultados:** `results/covariate_v5_vertexai/` — si existen se cargan sin re-simular

---

### Calculo de CRPS para Chronos (cambio respecto a v4_cloud)

Chronos-2 expone unicamente cuantiles (no samples i.i.d.). En v4_cloud se pasaban 5 cuantiles ordenados a `crps_ensemble`, lo cual rompe el supuesto i.i.d. del estimador de Hersbach e introduce un sesgo relativo positivo de hasta ~100% en el benchmark $\\\\mathcal{N}(0,1)$. **v5 calcula el CRPS via la representacion integral propia** $\\\\mathrm{CRPS}(F,y) = 2 \\\\int_0^1 \\\\mathrm{QL}_\\\\tau(F^{-1}(\\\\tau), y)\\\\,d\\\\tau$ aproximada con regla trapezoidal sobre **K=39 cuantiles uniformes** $\\\\{0.025, 0.050, \\\\dots, 0.975\\\\}$. Convergencia $\\\\mathcal{O}(1/K^2)$, scoring rule propia, misma familia que la WQL del paper de Chronos (Ansari et al. 2024). Para los modelos clasicos se sigue usando `crps_gaussian` (exacta) o `crps_ensemble` con paths bootstrap (samples genuinos). **Los CSV en `results/covariate_v5_vertexai/` no son comparables directamente con los de v4_cloud para Chronos.**

**Bloques de experimentos:**
- **C-A** (5) — ARIMAX(1) univariado: variacion de fuerza (beta) y persistencia de X (rho_x)
- **C-B** (2) — ARIMAX con dinamica AR mas rica (alta persistencia / signo negativo)
- **C-C** (3) — ARIMAX con multiples covariables (asimetricas / balanceadas / con ruido)
- **C-D** (3) — ARIMAX-GARCH: efecto en media y/o en varianza
- **C-E** (3) — VARX bivariado: sistema multivariado con covariable exogena comun
- **C-F** (2) — ADL-ECM: cointegracion entre Y_t y X_t (I(1) con relacion de largo plazo)
- **C-G** (3) — ARIMAX con tendencia deterministica lineal (delta*t)
- **C-H** (4) — SARIMAX estacional con covariable (s=4 trimestral y s=12 mensual)

**Notas metodologicas:**
- En todos los experimentos las covariables son *completamente observadas*: se proveen historico (`X_train`) y futuro conocido (`X_future`) a los modelos que las aceptan.
- Chronos-2 recibe las covariables via la API de `past_covariates` / `future_covariates`.
- Los modelos clasicos (SARIMAX, VARMAX, ARDL-ECM) son el contraste correctamente especificado para cada DGP.
- C-H.3/C-H.4 (s=12) restringen T_list a [50, 100, 200]: T=25 da solo ~2 ciclos estacionales, insuficiente para identificar el componente.
"""


# ────────────────────────────────────────────────────────────────────────────
# Cell 1 — imports / setup / Chronos load
# ────────────────────────────────────────────────────────────────────────────
SETUP = """import os
import warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import copy
import logging
import sys
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from IPython.display import display

from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.var_model import VAR as SMVAR

from mectesis.dgp import (
    ARIMAX_DGP, ARIMAX2Cov_DGP, ARIMAX_GARCH_DGP,
    ARIMAX_TREND_DGP, SARIMAX_SEASONAL_DGP,
    VARX_DGP, ADL_ECM_DGP,
)
from mectesis.models import (
    SARIMAXModel, VARMAXModel, ARDLModel,
    ChronosModel, ChronosCovariateModel,
    ChronosMultivariateCovariateModel,
)
from mectesis.simulation import (
    CovariateMonteCarloEngine, CovariateMultivariateEngine,
)

# ── Parametros globales (replican v4_cloud) ──────────────────────────────────
SEED    = 3649
H_BY_T  = {25: 6, 50: 18, 100: 24, 200: 24}
H_MAX   = 24
R_LIST  = [500]
T_LIST  = [25, 50, 100, 200]
RESULTS = Path("results/covariate_v5_vertexai")
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Logging dual: notebook + archivo .log ────────────────────────────────────
log_path = RESULTS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger().info
log(f"Log en: {log_path}")

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)

# ── Chronos-2: cargar una sola vez y envolver para univariado/multivariado ──
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Cargando Chronos-2 en {device} (puede tardar ~30 s la primera vez)...")
_chronos_base   = ChronosModel(device=device)
chronos_cov1    = ChronosCovariateModel(_chronos_base, n_covariates=1)
chronos_cov2    = ChronosCovariateModel(_chronos_base, n_covariates=2,
                       cov_names=["x0", "x1"])
chronos_mv_cov1 = ChronosMultivariateCovariateModel(_chronos_base, n_covariates=1)
log("Chronos-2 listo.")
"""


# ────────────────────────────────────────────────────────────────────────────
# Cell 2 — helper functions
# ────────────────────────────────────────────────────────────────────────────
HELPERS = """# ─── Funciones auxiliares ───────────────────────────────────────────────────

def _cache_path(exp_id: str, T: int, R: int) -> Path:
    return RESULTS / f"exp_{exp_id.replace('.', '_')}_T{T}_R{R}.csv"


# ── Univariate save / load ──────────────────────────────────────────────────

def _save_results(results: dict, path: Path):
    frames = []
    for mname, df in results.items():
        tmp = df.copy()
        tmp.insert(0, "model", mname)
        frames.append(tmp)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _load_results(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        mname: grp.drop(columns="model").reset_index(drop=True)
        for mname, grp in df.groupby("model", sort=False)
    }


def run_exp_cov(dgp, make_models_fn, dgp_params, exp_id,
                T_list=None, R_list=None, H_by_T=None, seed=SEED):
    \"\"\"Univariado con covariables: barre (T, R), cachea CSVs, retorna
    {(T, R): {model_name: DataFrame}}.\"\"\"
    T_list = T_list if T_list is not None else T_LIST
    R_list = R_list if R_list is not None else R_LIST
    H_by_T = H_by_T if H_by_T is not None else H_BY_T
    n_runs = len(T_list) * len(R_list)
    combos = ", ".join(
        f"(T={t}, H={H_by_T.get(t, H_MAX)}, R={r})"
        for t in T_list for r in R_list
    )
    log(f"Exp {exp_id}: {n_runs} ejecucion(es) -> {combos}")

    all_results = {}
    for T in T_list:
        h = H_by_T.get(T, H_MAX)
        for R in R_list:
            cache = _cache_path(exp_id, T, R)
            if cache.exists():
                log(f"  T={T} H={h}, R={R}: cargando {cache.name}")
                all_results[(T, R)] = _load_results(cache)
                continue
            log(f"  T={T} H={h}, R={R}: simulando...")
            dgp.rng = np.random.default_rng(seed)
            models  = make_models_fn(T)
            engine  = CovariateMonteCarloEngine(dgp, models, seed=seed)
            t0 = time.time()
            results = engine.run_monte_carlo(R, T, h, dgp_params, verbose=False)
            log(f"  T={T} H={h}, R={R}: OK ({time.time()-t0:.0f}s)")
            _save_results(results, cache)
            all_results[(T, R)] = results
    return all_results


# ── Multivariate save / load ────────────────────────────────────────────────

def _save_results_mv(results: dict, path: Path):
    frames = []
    for mname, var_dict in results.items():
        for var_idx, df in var_dict.items():
            tmp = df.copy()
            tmp.insert(0, "var", var_idx)
            tmp.insert(0, "model", mname)
            frames.append(tmp)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _load_results_mv(path: Path) -> dict:
    df = pd.read_csv(path)
    results = {}
    for mname, mgrp in df.groupby("model", sort=False):
        results[mname] = {}
        for var_idx, vgrp in mgrp.groupby("var", sort=True):
            results[mname][int(var_idx)] = (
                vgrp.drop(columns=["model", "var"]).reset_index(drop=True)
            )
    return results


def run_exp_mv_cov(dgp, make_models_fn, dgp_params, exp_id,
                    T_list=None, R_list=None, H_by_T=None, seed=SEED):
    \"\"\"Multivariado con covariables: barre (T, R), cachea CSVs, retorna
    {(T, R): {model_name: {var_idx: DataFrame}}}.\"\"\"
    T_list = T_list if T_list is not None else T_LIST
    R_list = R_list if R_list is not None else R_LIST
    H_by_T = H_by_T if H_by_T is not None else H_BY_T
    n_runs = len(T_list) * len(R_list)
    combos = ", ".join(
        f"(T={t}, H={H_by_T.get(t, H_MAX)}, R={r})"
        for t in T_list for r in R_list
    )
    log(f"Exp {exp_id}: {n_runs} ejecucion(es) -> {combos}")

    all_results = {}
    for T in T_list:
        h = H_by_T.get(T, H_MAX)
        for R in R_list:
            cache = _cache_path(exp_id, T, R)
            if cache.exists():
                log(f"  T={T} H={h}, R={R}: cargando {cache.name}")
                all_results[(T, R)] = _load_results_mv(cache)
                continue
            log(f"  T={T} H={h}, R={R}: simulando...")
            dgp.rng = np.random.default_rng(seed)
            models  = make_models_fn(T)
            engine  = CovariateMultivariateEngine(dgp, models, seed=seed)
            t0 = time.time()
            results = engine.run_monte_carlo(R, T, h, dgp_params, verbose=False)
            log(f"  T={T} H={h}, R={R}: OK ({time.time()-t0:.0f}s)")
            _save_results_mv(results, cache)
            all_results[(T, R)] = results
    return all_results


# ─── Bloques v3 (Corto / Medio / Largo) ─────────────────────────────────────

BLOCK_DEFS  = [("C", 1, 6), ("M", 7, 18), ("L", 19, 24)]
METRICS_V3  = ["bias", "variance", "rmse", "crps"]


def compute_blocks_v3(results_TR: dict) -> dict:
    out = {}
    for mname, df in results_TR.items():
        df_h = df[df["horizon"] != "avg_all"].copy()
        df_h["horizon"] = pd.to_numeric(df_h["horizon"], errors="coerce")
        blks = {}
        for blk, h1, h2 in BLOCK_DEFS:
            mask = (df_h["horizon"] >= h1) & (df_h["horizon"] <= h2)
            blks[blk] = df_h[mask].mean(numeric_only=True)
        out[mname] = blks
    return out


def build_grid_table(all_results: dict, classical_name: str,
                     chronos_name: str = "Chronos-2 (con X)"):
    rows = []
    for (T, R), res_TR in sorted(all_results.items()):
        blk_data = compute_blocks_v3(res_TR)
        cl_blks  = blk_data.get(classical_name, {})
        ch_blks  = blk_data.get(chronos_name, {})

        for mname, blks in blk_data.items():
            row = {"T": T, "Modelo": mname}
            for blk, h1, h2 in BLOCK_DEFS:
                s = blks.get(blk, pd.Series(dtype=float))
                for m in METRICS_V3:
                    row[f"{m}_{blk}"] = (
                        round(float(s[m]), 4)
                        if m in s.index and pd.notna(s[m]) else np.nan
                    )
                cl_s = cl_blks.get(blk, pd.Series(dtype=float))
                ch_s = ch_blks.get(blk, pd.Series(dtype=float))
                for m in ["rmse", "crps"]:
                    cv = float(cl_s[m]) if m in cl_s.index and pd.notna(cl_s[m]) else np.nan
                    hv = float(ch_s[m]) if m in ch_s.index and pd.notna(ch_s[m]) else np.nan
                    if np.isnan(cv) or np.isnan(hv):
                        row[f"best_{m}_{blk}"] = np.nan
                    else:
                        row[f"best_{m}_{blk}"] = "C" if cv <= hv else "T"
            rows.append(row)

    df_out = pd.DataFrame(rows).set_index(["T", "Modelo"])
    display(df_out.style.format(precision=4, na_rep="—"))


def compute_blocks_mv(results_TR: dict) -> dict:
    out = {}
    for mname, var_dict in results_TR.items():
        out[mname] = {}
        for var_idx, df in var_dict.items():
            df_h = df[df["horizon"] != "avg_all"].copy()
            df_h["horizon"] = pd.to_numeric(df_h["horizon"], errors="coerce")
            blks = {}
            for blk, h1, h2 in BLOCK_DEFS:
                mask = (df_h["horizon"] >= h1) & (df_h["horizon"] <= h2)
                blks[blk] = df_h[mask].mean(numeric_only=True)
            out[mname][var_idx] = blks
    return out


def build_grid_table_mv(all_results, classical_name: str,
                         chronos_name: str = "Chronos-2 joint (con X)",
                         var_names=None):
    rows = []
    for (T, R), res_TR in sorted(all_results.items()):
        blk_data = compute_blocks_mv(res_TR)
        var_idx_set = set()
        for var_dict in blk_data.values():
            var_idx_set.update(var_dict.keys())
        var_idx_set = {v for v in var_idx_set if v >= 0}   # excluir joint

        for var_idx in sorted(var_idx_set):
            vname = var_names[var_idx] if var_names else f"Y{var_idx+1}"
            cl_blks = blk_data.get(classical_name, {}).get(var_idx, {})
            ch_blks = blk_data.get(chronos_name, {}).get(var_idx, {})

            for mname, var_blks in blk_data.items():
                if var_idx not in var_blks:
                    continue
                blks = var_blks[var_idx]
                row = {"T": T, "Variable": vname, "Modelo": mname}
                for blk, h1, h2 in BLOCK_DEFS:
                    s = blks.get(blk, pd.Series(dtype=float))
                    for m in METRICS_V3:
                        row[f"{m}_{blk}"] = (
                            round(float(s[m]), 4)
                            if m in s.index and pd.notna(s[m]) else np.nan
                        )
                    cl_s = cl_blks.get(blk, pd.Series(dtype=float))
                    ch_s = ch_blks.get(blk, pd.Series(dtype=float))
                    for m in ["rmse", "crps"]:
                        cv = float(cl_s[m]) if m in cl_s.index and pd.notna(cl_s[m]) else np.nan
                        hv = float(ch_s[m]) if m in ch_s.index and pd.notna(ch_s[m]) else np.nan
                        if np.isnan(cv) or np.isnan(hv):
                            row[f"best_{m}_{blk}"] = np.nan
                        else:
                            row[f"best_{m}_{blk}"] = "C" if cv <= hv else "T"
                rows.append(row)

    df_out = pd.DataFrame(rows).set_index(["T", "Variable", "Modelo"])
    display(df_out.style.format(precision=4, na_rep="—"))


def build_grid_table_mv_joint(all_results, classical_name: str,
                               chronos_name: str = "Chronos-2 joint (con X)"):
    \"\"\"Tabla 2 (multivariada conjunta) por (T, Modelo) con Trace MSFE y avgCRPS.
    Lee la fila virtual var=-1 que inyecta el engine (si esta presente).\"\"\"
    rows = []
    for (T, R), res_TR in sorted(all_results.items()):
        blk_data = compute_blocks_mv(res_TR)
        cl_blks = blk_data.get(classical_name, {}).get(-1, {})
        ch_blks = blk_data.get(chronos_name, {}).get(-1, {})

        for mname, var_blks in blk_data.items():
            joint = var_blks.get(-1, None)
            if joint is None:
                continue
            row = {"T": T, "Modelo": mname}
            for blk, h1, h2 in BLOCK_DEFS:
                s = joint.get(blk, pd.Series(dtype=float))
                for m in ["trace_msfe", "avg_crps"]:
                    row[f"{m}_{blk}"] = (
                        round(float(s[m]), 4)
                        if m in s.index and pd.notna(s[m]) else np.nan
                    )
                cl_s = cl_blks.get(blk, pd.Series(dtype=float))
                ch_s = ch_blks.get(blk, pd.Series(dtype=float))
                for m in ["trace_msfe", "avg_crps"]:
                    cv = float(cl_s[m]) if m in cl_s.index and pd.notna(cl_s[m]) else np.nan
                    hv = float(ch_s[m]) if m in ch_s.index and pd.notna(ch_s[m]) else np.nan
                    if np.isnan(cv) or np.isnan(hv):
                        row[f"best_{m}_{blk}"] = np.nan
                    else:
                        row[f"best_{m}_{blk}"] = "C" if cv <= hv else "T"
            rows.append(row)

    if not rows:
        log("  [tabla joint] sin filas — el CSV no contiene metricas joint "
            "(probablemente el engine no inyecta var=-1; ignorar).")
        return

    df_out = pd.DataFrame(rows).set_index(["T", "Modelo"])
    display(df_out.style.format(precision=4, na_rep="—"))


# ─── Visualizacion de simulacion representativa ─────────────────────────────

def plot_simulation_cov(dgp, models, dgp_params, title="", T_vis=100, seed=SEED):
    \"\"\"Visualiza una realizacion univariada: historico + test + forecasts + intervalos
    para cada modelo (con o sin covariables).\"\"\"
    H_vis = H_BY_T.get(T_vis, H_MAX)
    dgp_copy = copy.deepcopy(dgp)
    dgp_copy.rng = np.random.default_rng(seed + 99991)
    data = dgp_copy.simulate(T=T_vis, **dgp_params)
    y, X = data["y"], data["X"]
    y_train, y_test = y[:-H_vis], y[-H_vis:]
    X_train, X_future = X[:-H_vis], X[-H_vis:]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    t_train = np.arange(len(y_train))
    t_test  = np.arange(len(y_train), T_vis)
    ax.plot(t_train, y_train, color="gray", lw=1.5, label="Historico")
    ax.axvline(len(y_train) - 0.5, color="black", ls="--", lw=1, alpha=0.6)
    ax.plot(t_test, y_test, color="black", lw=1.5, marker="o", ms=3, label="Observado")

    palette = ["steelblue", "darkorange", "seagreen", "purple", "teal", "crimson"]
    for i, model in enumerate(models):
        try:
            fkw = {"X_train": X_train} if getattr(model, "supports_covariates", False) else {}
            model.fit(y_train, **fkw)
            pkw = {"X_future": X_future} if getattr(model, "supports_covariates", False) else {}
            fcst = model.forecast(H_vis, **pkw)
            c = palette[i % len(palette)]
            ax.plot(t_test, fcst, color=c, lw=1.5, ls="--", marker="s", ms=3,
                    label=model.name)
            if getattr(model, "supports_intervals", False):
                lo, hi = model.forecast_intervals(H_vis, level=0.80, **pkw)
                ax.fill_between(t_test, lo, hi, color=c, alpha=0.15)
        except Exception as e:
            log(f"  [plot] {model.name} fallo: {e}")

    ax.set(title=title, xlabel="t", ylabel="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_simulation_mv_cov(dgp, models, dgp_params, var_names=None,
                            title="", T_vis=100, seed=SEED):
    \"\"\"Visualizacion multivariada: k subplots verticales con historico + test +
    forecasts + intervalos para cada modelo y cada variable.\"\"\"
    H_vis = H_BY_T.get(T_vis, H_MAX)
    dgp_copy = copy.deepcopy(dgp)
    dgp_copy.rng = np.random.default_rng(seed + 99991)
    data = dgp_copy.simulate(T=T_vis, **dgp_params)
    Y, X = data["y"], data["X"]
    k = Y.shape[1]
    y_train, y_test = Y[:-H_vis], Y[-H_vis:]
    X_train, X_future = X[:-H_vis], X[-H_vis:]

    for m in models:
        try:
            fkw = {"X_train": X_train} if getattr(m, "supports_covariates", False) else {}
            m.fit(y_train, **fkw)
        except Exception as e:
            log(f"  [plot] {m.name} fit fallo: {e}")

    palette = ["steelblue", "darkorange", "seagreen", "purple", "teal", "crimson"]
    fig, axes = plt.subplots(k, 1, figsize=(11, 3.0 * k), squeeze=False)
    x_tr = np.arange(len(y_train))
    x_te = np.arange(len(y_train), T_vis)

    for j, ax in enumerate(axes[:, 0]):
        vname = var_names[j] if var_names else f"Y{j+1}"
        ax.plot(x_tr, y_train[:, j], color="gray", lw=1.4, alpha=0.85, label="Historico")
        ax.plot(x_te, y_test[:, j], "k--", lw=1.5, label="Observado (test)")
        ax.axvline(len(y_train) - 0.5, color="black", ls="--", lw=1, alpha=0.5)

        for i, m in enumerate(models):
            try:
                pkw = {"X_future": X_future} if getattr(m, "supports_covariates", False) else {}
                y_hat = m.forecast(H_vis, **pkw)
                c = palette[i % len(palette)]
                ax.plot(x_te, y_hat[:, j], color=c, lw=1.5, ls="--",
                        marker="s", ms=3, label=m.name)
                if getattr(m, "supports_intervals", False):
                    lo, hi = m.forecast_intervals(H_vis, level=0.80, **pkw)
                    ax.fill_between(x_te, lo[:, j], hi[:, j], color=c, alpha=0.12)
            except Exception as e:
                log(f"  [plot] {m.name} forecast fallo en var {j}: {e}")

        ax.set(title=f"{vname}", xlabel="t", ylabel=vname)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ─── Verificacion DGP con covariables ───────────────────────────────────────

def verify_dgp_cov(label, dgp, dgp_params, checks):
    \"\"\"Replica el patron `verify_dgp` de v4_cloud para DGPs univariados con
    covariables. Cada check recibe (y, X, dgp, dgp_params).\"\"\"
    log(f"{'─'*60}")
    log(f"VERIFICACION DGP: {label}")
    log(f"{'─'*60}")
    dgp_copy = copy.deepcopy(dgp)
    dgp_copy.rng = np.random.default_rng(7777)
    try:
        data = dgp_copy.simulate(T=1000, **dgp_params)
        y, X = data["y"], data["X"]
    except Exception as e:
        log(f"  [FAIL] simulate() lanzo excepcion: {e}")
        return
    n_fail = 0
    for check_name, check_fn in checks:
        try:
            ok, msg = check_fn(y, X, dgp, dgp_params)
        except Exception as e:
            ok, msg = False, f"excepcion inesperada: {e}"
        tag = "PASS" if ok else "FAIL"
        log(f"  [{tag}] {check_name}: {msg}")
        if not ok:
            n_fail += 1
    if n_fail == 0:
        log("  -> TODAS LAS VERIFICACIONES PASARON")
    else:
        log(f"  -> {n_fail} FALLO(S)")


def verify_dgp_mv_cov(label, dgp, dgp_params, checks):
    \"\"\"Version multivariada: y es (T, k), X es (T, p).\"\"\"
    log(f"{'─'*60}")
    log(f"VERIFICACION DGP: {label}")
    log(f"{'─'*60}")
    dgp_copy = copy.deepcopy(dgp)
    dgp_copy.rng = np.random.default_rng(7777)
    try:
        data = dgp_copy.simulate(T=500, **dgp_params)
        Y, X = data["y"], data["X"]
    except Exception as e:
        log(f"  [FAIL] simulate() lanzo excepcion: {e}")
        return
    n_fail = 0
    for check_name, check_fn in checks:
        try:
            ok, msg = check_fn(Y, X, dgp, dgp_params)
        except Exception as e:
            ok, msg = False, f"excepcion inesperada: {e}"
        tag = "PASS" if ok else "FAIL"
        log(f"  [{tag}] {check_name}: {msg}")
        if not ok:
            n_fail += 1
    if n_fail == 0:
        log("  -> TODAS LAS VERIFICACIONES PASARON")
    else:
        log(f"  -> {n_fail} FALLO(S)")


# ── Checks individuales (univariados con covariables) ───────────────────────

def chk_y_stationary(y, X, dgp, params):
    pval = adfuller(y, autolag="AIC")[1]
    return pval < 0.05, f"ADF Y p={pval:.4f} (umbral 0.05)"


def chk_y_nonstationary(y, X, dgp, params):
    pval = adfuller(y, autolag="AIC")[1]
    return pval > 0.10, f"ADF Y p={pval:.4f} (se espera >0.10)"


def chk_y_zero_mean(y, X, dgp, params):
    \"\"\"Media de Y ~ 0 usando standard error robusto a autocorrelacion
    (Newey-West / Bartlett kernel). El SE tipico sigma/sqrt(T) subestima
    la varianza de la media muestral cuando Y es persistente — caso
    tipico de ARIMAX con phi alto y/o rho_x alto. Si el DGP tiene
    componente estacional con periodo s, ampliamos bw a max(bw_default,
    2*s) para capturar las autocovarianzas a lags estacionales — el
    default floor(4*(T/100)^(2/9)) subestima LRV cuando rho[s] es alta.\"\"\"
    mu = y.mean(); T = len(y)
    e = y - mu
    bw_default = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))
    s = int(params.get("s", 0))
    bw = max(bw_default, 2 * s) if s > 0 else bw_default
    gamma0 = float(np.dot(e, e) / T)
    lrv = gamma0
    for k in range(1, bw + 1):
        w = 1.0 - k / (bw + 1.0)
        gamma_k = float(np.dot(e[k:], e[:-k]) / T)
        lrv += 2.0 * w * gamma_k
    se = np.sqrt(max(lrv, 1e-12) / T)
    tol = 3.0 * se
    tag = f", adaptado a s={s}" if s > 0 else ""
    return abs(mu) < tol, f"media={mu:.4f}, tol=+/-{tol:.4f} (HAC bw={bw}{tag})"


def chk_x_stationary(y, X, dgp, params):
    \"\"\"Cada columna de X debe ser estacionaria (excepto cuando rho_x es 1, p.ej. ADL-ECM).\"\"\"
    pvals = [float(adfuller(X[:, i], autolag="AIC")[1]) for i in range(X.shape[1])]
    ok = all(p < 0.05 for p in pvals)
    return ok, f"ADF X p-vals={[round(p, 4) for p in pvals]}"


def chk_x_nonstationary(y, X, dgp, params):
    \"\"\"X debe ser I(1) — ADF en niveles no rechaza, ADF en diferencias rechaza.\"\"\"
    msgs, ok_all = [], True
    for i in range(X.shape[1]):
        pv_lvl = float(adfuller(X[:, i], autolag="AIC")[1])
        pv_dif = float(adfuller(np.diff(X[:, i]), autolag="AIC")[1])
        ok_i = (pv_lvl > 0.10) and (pv_dif < 0.05)
        ok_all = ok_all and ok_i
        msgs.append(f"X{i}: lvl={pv_lvl:.3f}, dif={pv_dif:.3f}")
    return ok_all, " | ".join(msgs)


def chk_acf_rho_x(y, X, dgp, params):
    rho = params.get("rho_x", None)
    if rho is None:
        return True, "no aplica (sin rho_x)"
    empirics = []
    for i in range(X.shape[1]):
        emp = float(acf(X[:, i], nlags=1, fft=True)[1])
        empirics.append(emp)
    diffs = [abs(e - rho) for e in empirics]
    ok = all(d < 0.15 for d in diffs)
    return ok, f"ACF[1] X={[round(e, 3) for e in empirics]}, rho_x_DGP={rho}"


def chk_yx_correlation(y, X, dgp, params):
    \"\"\"Correlacion contemporanea no nula y con signo correcto cuando el efecto
    exogeno en la media no es cero. Si el DGP tiene tendencia deterministica
    (delta != 0), la varianza total de Y queda dominada por delta*t y la
    correlacion contemporanea queda diluida — en ese caso detrend Y primero
    via OLS lineal para aislar la senal estocastica.\"\"\"
    for key in ("beta", "beta_mean", "beta1"):
        if key in params and params[key] != 0:
            val = params[key]
            y_used = y
            tag = ""
            if params.get("delta", 0.0) != 0.0:
                T = len(y)
                t = np.arange(T)
                slope, intercept = np.polyfit(t, y, 1)
                y_used = y - (slope * t + intercept)
                tag = "_detrended"
            corr = float(np.corrcoef(y_used, X[:, 0])[0, 1])
            ok = (abs(corr) > 0.05) and (np.sign(corr) == np.sign(val))
            return ok, f"corr(Y{tag},X[0])={corr:.4f}, {key}={val}"
    return True, "no aplica (efecto exogeno en media = 0)"


def chk_arch_effects_resid(y, X, dgp, params):
    \"\"\"Detecta efectos ARCH en residuos de una regresion OLS de Y sobre X (proxy).\"\"\"
    X_with_const = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
    resid = y - X_with_const @ coef
    lb = acorr_ljungbox(resid ** 2, lags=[10], return_df=True)
    pval = float(lb["lb_pvalue"].iloc[0])
    return pval < 0.05, f"LB(10) sobre resid^2: p={pval:.4f} (se espera <0.05)"


def chk_fit_sarimax(y, X, dgp, params):
    \"\"\"Smoke test: SARIMAX con covariables ajusta y forecastea sin NaN.\"\"\"
    try:
        m = SARIMAXModel((1, 0, 0), name_suffix="check")
        m.fit(y[:800], X_train=X[:800])
        fc = m.forecast(horizon=6, X_future=X[800:806])
        ok = (len(fc) == 6) and (not np.any(np.isnan(fc)))
        return ok, f"SARIMAX fit+forecast OK, primeros: {np.round(fc[:3], 4)}"
    except Exception as e:
        return False, str(e)


# ── Checks individuales (multivariados con covariables, k=2) ────────────────

def chk_varx_stability(Y, X, dgp, params):
    A = np.asarray(getattr(dgp, "A"))
    eig = np.linalg.eigvals(A)
    mod_max = float(np.max(np.abs(eig)))
    return mod_max < 0.999, f"max|lambda(A)|={mod_max:.4f} (umbral 0.999)"


def chk_sigma_psd_mv(Y, X, dgp, params):
    Sigma = np.asarray(getattr(dgp, "Sigma"))
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as e:
        return False, f"Sigma no es PSD: {e}"
    eigs = np.linalg.eigvalsh(Sigma)
    return float(eigs.min()) > 1e-8, f"min eig(Sigma)={float(eigs.min()):.6f}"


def chk_x_stationary_mv(Y, X, dgp, params):
    rho = getattr(dgp, "rho_x", None)
    pval = float(adfuller(X[:, 0], autolag="AIC")[1])
    msg = f"ADF X p={pval:.4f}" + (f", rho_x_DGP={rho}" if rho is not None else "")
    return pval < 0.05, msg


def chk_empirical_finite_mv(Y, X, dgp, params):
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(X))):
        return False, "Y o X contiene NaN/inf"
    sd_y = np.std(Y, axis=0)
    return bool(np.all(sd_y < 1e6)), f"std(Y)={np.round(sd_y, 3).tolist()}"


def chk_varx_gamma_signal(Y, X, dgp, params):
    \"\"\"Si gamma!=0, la correlacion Y_i ~ X[0] debe tener signo correcto.\"\"\"
    gamma = np.asarray(getattr(dgp, "gamma", [0.0, 0.0]))
    msgs, ok_all = [], True
    for i in range(Y.shape[1]):
        if gamma[i] == 0:
            msgs.append(f"Y{i+1}: no aplica (gamma=0)")
            continue
        corr = float(np.corrcoef(Y[:, i], X[:, 0])[0, 1])
        ok_i = (abs(corr) > 0.03) and (np.sign(corr) == np.sign(gamma[i]))
        ok_all = ok_all and ok_i
        msgs.append(f"Y{i+1}: corr={corr:.3f}, gamma={gamma[i]}")
    return ok_all, " | ".join(msgs)


# ── Checks ADL-ECM (cointegracion) ──────────────────────────────────────────

def chk_y_I1(y, X, dgp, params):
    pv_lvl = float(adfuller(y, autolag="AIC")[1])
    pv_dif = float(adfuller(np.diff(y), autolag="AIC")[1])
    ok = (pv_lvl > 0.10) and (pv_dif < 0.05)
    return ok, f"ADF Y lvl={pv_lvl:.3f}, dif={pv_dif:.3f}"


def chk_cointegration_yx(y, X, dgp, params):
    \"\"\"La combinacion lineal Y - X debe ser I(0).\"\"\"
    z = y - X[:, 0]
    pv = float(adfuller(z, autolag="AIC")[1])
    return pv < 0.05, f"ADF(Y-X) p={pv:.4f} (se espera <0.05)"


def chk_fit_ardl(y, X, dgp, params):
    try:
        m = ARDLModel()
        m.fit(y[:800], X_train=X[:800])
        fc = m.forecast(horizon=6, X_future=X[800:806])
        ok = (len(fc) == 6) and (not np.any(np.isnan(fc)))
        return ok, f"ARDL-ECM fit+forecast OK, primeros: {np.round(fc[:3], 4)}"
    except Exception as e:
        return False, str(e)


# ── Grupos de checks por familia de DGP ─────────────────────────────────────

CHECKS_ARIMAX = [
    ("Y estacionaria (ADF)",                  chk_y_stationary),
    ("Y media ~ 0",                           chk_y_zero_mean),
    ("X estacionaria (ADF)",                  chk_x_stationary),
    ("ACF[1] de X ~ rho_x",                   chk_acf_rho_x),
    ("Correlacion Y vs X consistente",        chk_yx_correlation),
    ("SARIMAX(1,0,0) con X fit+forecast",     chk_fit_sarimax),
]

CHECKS_ARIMAX_GARCH = [
    ("Y estacionaria en media (ADF)",         chk_y_stationary),
    ("Y media ~ 0",                           chk_y_zero_mean),
    ("X estacionaria (ADF)",                  chk_x_stationary),
    ("ACF[1] de X ~ rho_x",                   chk_acf_rho_x),
    ("Efectos ARCH detectables en residuos",  chk_arch_effects_resid),
    ("SARIMAX con X fit+forecast",            chk_fit_sarimax),
]

CHECKS_VARX = [
    ("Estabilidad VAR (max|eig(A)|<1)",       chk_varx_stability),
    ("Sigma PSD",                             chk_sigma_psd_mv),
    ("X estacionaria (ADF)",                  chk_x_stationary_mv),
    ("Salida finita y std razonable",         chk_empirical_finite_mv),
    ("Senal de gamma en Y",                   chk_varx_gamma_signal),
]

CHECKS_ADL_ECM = [
    ("Y es I(1) (ADF niveles/diferencias)",   chk_y_I1),
    ("X es I(1) (ADF niveles/diferencias)",   chk_x_nonstationary),
    ("Y - X es I(0) (cointegracion)",         chk_cointegration_yx),
    ("ARDL-ECM fit+forecast",                 chk_fit_ardl),
]


# ── Checks tendencia / estacionalidad ───────────────────────────────────────

def chk_trend_slope(y, X, dgp, params):
    \"\"\"OLS slope de Y vs t cercana a delta. Con T=1000 y delta=0.05, la
    desviacion tipica del estimador es ~sigma_y * sqrt(12)/T^{1.5} ~ 0.0001
    para sigma_y=1 y T=1000, asi que tol=0.01 es comodo y conservador.\"\"\"
    delta = params.get("delta", None)
    if delta is None:
        return True, "no aplica"
    T = len(y)
    slope = float(np.polyfit(np.arange(T), y, 1)[0])
    tol = max(0.01, 0.20 * abs(delta))
    return abs(slope - delta) < tol, f"slope_OLS={slope:.5f}, delta_DGP={delta:.5f}, tol={tol:.4f}"


def chk_seasonal_acf(y, X, dgp, params):
    \"\"\"ACF[s] de los residuos OLS de Y sobre X significativamente positiva.
    Se quita el efecto exogeno aproximadamente regresando Y sobre X para
    aislar la senal estacional.\"\"\"
    s = params.get("s", 4)
    X_with_const = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
    resid = y - X_with_const @ coef
    T = len(resid)
    thr = 2.0 / np.sqrt(T)
    acf_vals = acf(resid, nlags=s + 1, fft=True)
    val_s = abs(acf_vals[s])
    return val_s > thr, f"|ACF[s={s}](resid)|={val_s:.4f}, umbral 2/sqrt(T)={thr:.4f}"


CHECKS_ARIMAX_TREND = [
    ("Y tendencia: slope_OLS ~ delta",        chk_trend_slope),
    ("X estacionaria (ADF)",                  chk_x_stationary),
    ("ACF[1] de X ~ rho_x",                   chk_acf_rho_x),
    ("Correlacion Y vs X consistente",        chk_yx_correlation),
    ("SARIMAX con X+trend fit+forecast",      chk_fit_sarimax),
]

CHECKS_ARIMAX_SEASONAL = [
    ("Y estacionaria (ADF)",                  chk_y_stationary),
    ("Y media ~ 0",                           chk_y_zero_mean),
    ("X estacionaria (ADF)",                  chk_x_stationary),
    ("ACF[s] estacional significativa",       chk_seasonal_acf),
    ("Correlacion Y vs X consistente",        chk_yx_correlation),
    ("SARIMAX con X fit+forecast",            chk_fit_sarimax),
]
"""


# ────────────────────────────────────────────────────────────────────────────
# Helpers to build experiment cells
# ────────────────────────────────────────────────────────────────────────────
def exp_uni_cell(exp_id: str, title: str, dgp_class: str, dgp_args: str,
                  models_lambda: str, dgp_params: str,
                  classical_name: str,
                  checks_var: str = "CHECKS_ARIMAX",
                  chronos_name: str = "Chronos-2 (con X)",
                  T_list_override: str = None) -> str:
    var_id = exp_id.replace(".", "_").replace("-", "_")
    extra_arg = f",\n        T_list={T_list_override}" if T_list_override else ""
    return f"""try:
    # {exp_id} -- {title}
    dgp_{var_id} = {dgp_class}({dgp_args})
    make_models_{var_id} = lambda T: [
        {models_lambda},
    ]
    dgp_params_{var_id} = dict({dgp_params})
    verify_dgp_cov("{exp_id} -- {title}", dgp_{var_id},
                   dgp_params_{var_id}, {checks_var})
    res_{var_id} = run_exp_cov(
        dgp_{var_id}, make_models_{var_id}, dgp_params_{var_id},
        exp_id="{exp_id}"{extra_arg},
    )
    log("\\n" + "="*60 + "\\n{exp_id} -- {title}\\n" + "="*60)
    build_grid_table(res_{var_id}, classical_name="{classical_name}",
                     chronos_name="{chronos_name}")
    plot_simulation_cov(dgp_{var_id}, make_models_{var_id}(200),
                        dgp_params_{var_id},
                        title="{exp_id} -- {title}")
except Exception as _exc:
    log("\\n" + "!"*60)
    log("[CELDA {exp_id} FALLO] " + type(_exc).__name__ + ": " + str(_exc))
    log("!"*60)
    log(traceback.format_exc())
"""


def exp_mv_cell(exp_id: str, title: str, dgp_args: str,
                 models_lambda: str, dgp_params: str,
                 classical_name: str,
                 checks_var: str = "CHECKS_VARX",
                 var_names: str = '["Y1", "Y2"]') -> str:
    var_id = exp_id.replace(".", "_").replace("-", "_")
    return f"""try:
    # {exp_id} -- {title}
    dgp_{var_id} = VARX_DGP({dgp_args})
    make_models_{var_id} = lambda T: [
        {models_lambda},
    ]
    dgp_params_{var_id} = dict({dgp_params})
    verify_dgp_mv_cov("{exp_id} -- {title}", dgp_{var_id},
                      dgp_params_{var_id}, {checks_var})
    res_{var_id} = run_exp_mv_cov(
        dgp_{var_id}, make_models_{var_id}, dgp_params_{var_id},
        exp_id="{exp_id}",
    )
    log("\\n" + "="*60 + "\\n{exp_id} -- {title}\\n" + "="*60)
    build_grid_table_mv(res_{var_id}, classical_name="{classical_name}",
                        var_names={var_names})
    print("\\n--- Tabla 2: metricas multivariadas conjuntas (si disponibles) ---")
    build_grid_table_mv_joint(res_{var_id}, classical_name="{classical_name}")
    plot_simulation_mv_cov(dgp_{var_id}, make_models_{var_id}(200),
                            dgp_params_{var_id}, var_names={var_names},
                            title="{exp_id} -- {title}")
except Exception as _exc:
    log("\\n" + "!"*60)
    log("[CELDA {exp_id} FALLO] " + type(_exc).__name__ + ": " + str(_exc))
    log("!"*60)
    log(traceback.format_exc())
"""


# ────────────────────────────────────────────────────────────────────────────
# Block headers
# ────────────────────────────────────────────────────────────────────────────
BLOCK_CA = """---
## Bloque C-A — ARIMAX(1) univariado: fuerza del efecto y persistencia de X (5 exps)

Aisla el efecto de dos parametros estructurales en el caso mas simple
(ARIMAX(1) con una covariable estacionaria):

- **beta**: fuerza del efecto exogeno. Variantes: 0.2 (debil), 0.5 (medio), 0.8 (fuerte).
- **rho_x**: persistencia de la covariable X. Variantes: 0.0 (ruido), 0.7 (default), 0.95 (casi RW).

**DGP general:** $Y_t = 0.6\\,Y_{t-1} + \\beta\\,X_t + \\varepsilon_t$, con
$X_t = \\rho_x X_{t-1} + \\eta_t$, $\\sigma_y = \\sigma_x = 1$.

**Modelos:** SARIMAX(1,0,0) con X (clasico correctamente especificado) vs Chronos-2 con X.
"""

BLOCK_CB = """---
## Bloque C-B — ARIMAX con dinamica AR mas rica (2 exps)

Explora si Chronos sigue capturando la senal exogena cuando la dinamica
autoregresiva de $Y$ se vuelve mas dificil:

- **C-B.1**: alta persistencia ($\\phi = 0.9$) — el componente AR domina sobre la covariable.
- **C-B.2**: signo negativo ($\\phi = -0.6$) — dinamica oscilatoria, ACF alternante.

Notese que `ARIMAX_DGP` solo expone `phi` escalar (AR(1)); no se simula AR(2)
sin extender el DGP. La cobertura de dinamicas mas complejas queda implicita
en estos dos casos limite.
"""

BLOCK_CC = """---
## Bloque C-C — ARIMAX con multiples covariables (3 exps)

Mide como escala la performance cuando hay mas de un regresor exogeno.

**DGP:** $Y_t = 0.6\\,Y_{t-1} + \\beta_1 X_{1t} + \\beta_2 X_{2t} + \\varepsilon_t$,
con $X_{i,t} = 0.7 X_{i,t-1} + \\eta_{i,t}$ independientes.

Variantes:
- **C-C.1**: efectos asimetricos ($\\beta_1 = 0.8$, $\\beta_2 = 0.4$).
- **C-C.2**: efectos balanceados ($\\beta_1 = \\beta_2 = 0.5$).
- **C-C.3**: capacidad de filtrado — un regresor relevante y otro nulo
  ($\\beta_1 = 0.8$, $\\beta_2 = 0.0$).
"""

BLOCK_CD = """---
## Bloque C-D — ARIMAX con volatilidad condicional (3 exps)

La covariable entra en la media y/o en la varianza condicional del error.

**DGP:** $Y_t = 0.4\\,Y_{t-1} + \\beta_{\\text{mean}}\\,X_t + \\varepsilon_t$,
con $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2
+ \\beta_{\\text{garch}}\\,\\sigma_{t-1}^2 + \\delta_{\\text{var}}\\,X_t^2$
y $X_t = 0.7\\,X_{t-1} + \\eta_t$.

Variantes:
- **C-D.1**: efecto solo en la media ($\\beta_{\\text{mean}} = 0.5$, $\\delta_{\\text{var}} = 0$).
- **C-D.2**: efecto en media y varianza ($\\beta_{\\text{mean}} = 0.5$, $\\delta_{\\text{var}} = 0.1$).
- **C-D.3**: efecto solo en varianza ($\\beta_{\\text{mean}} = 0$, $\\delta_{\\text{var}} = 0.3$).

SARIMAX no modela la varianza condicional explicitamente — pierde calibracion
en intervalos cuando $\\delta_{\\text{var}} > 0$. Aqui se mide si Chronos
absorbe esa estructura implicitamente.
"""

BLOCK_CE = """---
## Bloque C-E — VARX bivariado (3 exps)

Sistema multivariado con covariable exogena comun:

$$\\mathbf{Y}_t = A\\,\\mathbf{Y}_{t-1} + \\boldsymbol{\\gamma}\\,X_t + \\boldsymbol{\\varepsilon}_t$$

con $\\boldsymbol{\\varepsilon}_t \\sim \\mathcal{N}(0, \\Sigma)$, $A \\in \\mathbb{R}^{2 \\times 2}$,
$\\boldsymbol{\\gamma} \\in \\mathbb{R}^2$, y $X_t$ AR(1) estacionaria.

Variantes:
- **C-E.1**: baseline ($A$ de dependencia debil, $\\gamma$ asimetrico).
- **C-E.2**: dependencia cruzada y efecto exogeno fuertes simultaneamente.
- **C-E.3**: efecto exogeno debil ($\\gamma$ bajo) — testea si Chronos joint
  separa la senal AR de la covariable.

**Modelos:** `VARMAX(1) con X` (clasico) vs `Chronos-2 joint (con X)`.
"""

BLOCK_CG = """---
## Bloque C-G — ARIMAX con tendencia deterministica lineal (3 exps)

Captura series con drift estructural ademas de la dinamica AR y la covariable.
Relevante para macroeconomicas en niveles (PIB, IPC, exportaciones).

**DGP:** $Y_t = \\alpha + \\delta\\,t + \\phi\\,Y_{t-1}^{stat} + \\beta\\,X_t + \\varepsilon_t$,
con $X_t = \\rho_x\\,X_{t-1} + \\eta_t$. La tendencia se agrega al output final;
la dinamica interna sigue siendo estacionaria (recuperable por SARIMAX con `trend="ct"`).

**Modelo clasico correcto:** `SARIMAXModel(order=(1,0,0), trend="ct", name_suffix="con X+trend")`.
Comparar con SARIMAX sin tendencia ya mostro mejoras de **+44% a +122% en RMSE** —
la misspecificacion de tendencia es muy costosa a horizontes largos.

Variantes:
- **C-G.1**: tendencia leve ($\\delta = 0.05$), $\\beta = 0.5$.
- **C-G.2**: tendencia fuerte ($\\delta = 0.10$), $\\beta = 0.5$.
- **C-G.3**: tendencia leve + cov fuerte ($\\delta = 0.05$, $\\beta = 0.8$).
"""

BLOCK_CH = """---
## Bloque C-H — SARIMAX estacional con covariable (4 exps)

Series con componente estacional periodico ademas de AR y covariable.
Relevante para PIB trimestral (s=4) e IPC mensual (s=12).

**DGP:** SARIMA(1,0,0)(1,0,0)[s] con covariable, forma multiplicativa exacta:
$$(1 - \\phi L)(1 - \\Phi L^s)\\,Y_t = \\beta\\,X_t + \\varepsilon_t$$

Expansion: $Y_t = \\phi Y_{t-1} + \\Phi Y_{t-s} - \\phi\\Phi Y_{t-s-1} + \\beta X_t + \\varepsilon_t$.

**Modelo clasico correcto:** `SARIMAXModel(order=(1,0,0), seasonal_order=(1,0,0,s))`.
En MC R=80 a T=200 con $\\phi=0.3$, $\\Phi=0.7$, el bien especificado gana **+7% a +15%**
sobre SARIMAX sin componente estacional.

Variantes:
- **C-H.1**: trimestral ($s=4$), baseline $\\beta = 0.5$.
- **C-H.2**: trimestral ($s=4$), cov fuerte $\\beta = 0.8$.
- **C-H.3**: mensual ($s=12$), baseline $\\beta = 0.5$. T_list = [50, 100, 200] (T=25 da solo 2 ciclos).
- **C-H.4**: mensual ($s=12$), cov fuerte $\\beta = 0.8$. T_list = [50, 100, 200].

Parametros estacionales: $\\phi = 0.3$ (AR no estacional debil), $\\Phi = 0.7$
(persistencia estacional fuerte), $\\rho_x = 0.7$.
"""

BLOCK_CF = """---
## Bloque C-F — Cointegracion ADL-ECM (2 exps)

Series $Y_t$, $X_t$ ambas $I(1)$ con relacion de equilibrio de largo plazo
$Y_t - X_t \\sim I(0)$:

$$\\Delta Y_t = \\alpha_{\\text{ecm}}\\,(Y_{t-1} - X_{t-1}) + \\Delta X_t + \\eta_t$$

El parametro $\\alpha_{\\text{ecm}} < 0$ controla la velocidad de correccion
al equilibrio. Variantes:
- **C-F.1**: $\\alpha_{\\text{ecm}} = -0.3$ (correccion moderada).
- **C-F.2**: $\\alpha_{\\text{ecm}} = -0.6$ (correccion rapida).

**Modelos:** `ARDL-ECM` (clasico cointegrado), `SARIMAX(1,1,0) con X`
(benchmark sin estructura cointegrada), `Chronos-2 (con X)`.
"""


# ────────────────────────────────────────────────────────────────────────────
# Per-experiment markdown intros + code cells
# ────────────────────────────────────────────────────────────────────────────
# Each entry is (intro_markdown, code_block)
SARIMAX_NAME_1 = "SARIMAX(1, 0, 0) con X"
SARIMAX_NAME_2 = "SARIMAX(2, 0, 0) con X"
SARIMAX_NAME_1_1_0 = "SARIMAX(1, 1, 0) con X"
SARIMAX_NAME_TREND = "SARIMAX(1, 0, 0) con X+trend"
SARIMAX_NAME_SEAS_4 = "SARIMAX(1, 0, 0)x(1, 0, 0, 4) con X"
SARIMAX_NAME_SEAS_12 = "SARIMAX(1, 0, 0)x(1, 0, 0, 12) con X"
VARMAX_NAME_1 = "VARMAX(1) con X"

EXPERIMENTS = []

# ── C-A ─────────────────────────────────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-A.1 — ARIMAX(1) efecto fuerte (beta = 0.8)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.8\\,X_t + \\varepsilon_t$, $X_t = 0.7 X_{t-1} + \\eta_t$.\n\n"
        "**Hipotesis:** efecto exogeno dominante; ambos modelos deberian capturarlo. A T=200, Chronos suele superar a SARIMAX en RMSE/CRPS (resultado preliminar de Exp 3.1).\n",
        exp_uni_cell(
            "C-A.1", "ARIMAX(1) beta=0.8",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.6, beta=0.8, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-A.2 — ARIMAX(1) efecto medio (beta = 0.5)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$.\n\n"
        "**Hipotesis:** punto intermedio entre C-A.1 y C-A.3 — permite ver la curva de utilidad de la covariable como funcion de $\\beta$.\n",
        exp_uni_cell(
            "C-A.2", "ARIMAX(1) beta=0.5",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.6, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-A.3 — ARIMAX(1) efecto debil (beta = 0.2)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.2\\,X_t + \\varepsilon_t$.\n\n"
        "**Hipotesis:** la covariable aporta poca senal frente al ruido $\\varepsilon_t$. SARIMAX deberia dominar (parsimonia gana cuando la senal es debil).\n",
        exp_uni_cell(
            "C-A.3", "ARIMAX(1) beta=0.2",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.6, beta=0.2, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-A.4 — Covariable casi raiz unitaria (rho_x = 0.95)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$, $X_t = 0.95\\,X_{t-1} + \\eta_t$.\n\n"
        "**Hipotesis:** $X$ es muy persistente — casi $I(1)$ — lo cual hace que su nivel sea informativo de largo plazo. Chronos podria aprovechar mejor esta inercia frente a SARIMAX.\n",
        exp_uni_cell(
            "C-A.4", "rho_x=0.95",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.6, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.95",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-A.5 — Covariable cercana a ruido blanco (rho_x = 0.0)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$, $X_t = \\eta_t$.\n\n"
        "**Hipotesis:** $X$ no tiene estructura predecible — su valor *futuro* sigue siendo conocido (covariable observada), pero no hay senal en su trayectoria pasada. La utilidad del componente exogeno se reduce al efecto contemporaneo del nivel futuro.\n",
        exp_uni_cell(
            "C-A.5", "rho_x=0.0",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.6, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.0",
            SARIMAX_NAME_1,
        ),
    ),
]

# ── C-B ─────────────────────────────────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-B.1 — ARIMAX(1) alta persistencia (phi = 0.9)\n\n"
        "**DGP:** $Y_t = 0.9\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$, $X_t = 0.7\\,X_{t-1} + \\eta_t$.\n\n"
        "**Hipotesis:** el AR es casi raiz unitaria y domina la dinamica; el aporte marginal de la covariable es secundario. Mide si Chronos identifica correctamente la persistencia interna sin sobre-pesar la senal exogena.\n",
        exp_uni_cell(
            "C-B.1", "ARIMAX(1) phi=0.9",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.9, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-B.2 — ARIMAX(1) dinamica oscilatoria (phi = -0.6)\n\n"
        "**DGP:** $Y_t = -0.6\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$, $X_t = 0.7\\,X_{t-1} + \\eta_t$.\n\n"
        "**Hipotesis:** ACF alternante en signo. SARIMAX(1,0,0) captura el AR negativo; Chronos enfrenta un patron mas inusual en su pretraining (donde dominan series con $\\phi > 0$).\n",
        exp_uni_cell(
            "C-B.2", "ARIMAX(1) phi=-0.6",
            "ARIMAX_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=-0.6, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
]

# ── C-C ─────────────────────────────────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-C.1 — Dos covariables asimetricas (beta1=0.8, beta2=0.4)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.8\\,X_{1t} + 0.4\\,X_{2t} + \\varepsilon_t$.\n\n"
        "**Hipotesis:** SARIMAX especifica ambos regresores; Chronos debe distinguir su peso relativo sin senal de cuales son mas importantes. Replica Exp 3.3.\n",
        exp_uni_cell(
            "C-C.1", "ARIMAX 2-cov asimetricas",
            "ARIMAX2Cov_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov2",
            "phi=0.6, beta1=0.8, beta2=0.4, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-C.2 — Dos covariables balanceadas (beta1=beta2=0.5)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.5\\,X_{1t} + 0.5\\,X_{2t} + \\varepsilon_t$.\n\n"
        "**Hipotesis:** efectos iguales — no hay informacion previa sobre cual regresor pesa mas. Ambos modelos deberian explotarlos equitativamente.\n",
        exp_uni_cell(
            "C-C.2", "ARIMAX 2-cov balanceadas",
            "ARIMAX2Cov_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov2",
            "phi=0.6, beta1=0.5, beta2=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
    (
        "### C-C.3 — Una covariable relevante y otra nula (beta1=0.8, beta2=0.0)\n\n"
        "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.8\\,X_{1t} + 0\\cdot X_{2t} + \\varepsilon_t$.\n\n"
        "**Hipotesis:** capacidad de filtrado/regularizacion. SARIMAX estima $\\hat\\beta_2 \\to 0$ con muestra grande; Chronos no estima coeficientes — testea si el ruido de la covariable irrelevante degrada sus pronosticos.\n",
        exp_uni_cell(
            "C-C.3", "ARIMAX 2-cov filtrado",
            "ARIMAX2Cov_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov2",
            "phi=0.6, beta1=0.8, beta2=0.0, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
        ),
    ),
]

# ── C-D ─────────────────────────────────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-D.1 — ARIMAX-GARCH: efecto solo en la media (delta_var = 0)\n\n"
        "**DGP:** $Y_t = 0.4\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$, $\\sigma_t^2 = 0.1 + 0.1\\varepsilon_{t-1}^2 + 0.75\\sigma_{t-1}^2$.\n\n"
        "**Hipotesis:** GARCH puro sin influencia exogena en la varianza. SARIMAX no modela $\\sigma_t$ y tendra intervalos mal calibrados; Chronos quantilico puede capturar la heteroscedasticidad implicitamente.\n",
        exp_uni_cell(
            "C-D.1", "ARIMAX-GARCH solo media",
            "ARIMAX_GARCH_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.4, beta_mean=0.5, omega=0.1, alpha=0.1, beta_garch=0.75, "
            "delta_var=0.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
            checks_var="CHECKS_ARIMAX_GARCH",
        ),
    ),
    (
        "### C-D.2 — ARIMAX-GARCH: media + varianza (delta_var = 0.1)\n\n"
        "**DGP:** misma estructura, con $X_t^2$ inyectado en la varianza ($\\delta_{\\text{var}} = 0.1$).\n\n"
        "**Hipotesis:** caso completo (replica Exp 3.5). La covariable contribuye a ambos momentos; SARIMAX solo captura el primero.\n",
        exp_uni_cell(
            "C-D.2", "ARIMAX-GARCH media+var",
            "ARIMAX_GARCH_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.4, beta_mean=0.5, omega=0.1, alpha=0.1, beta_garch=0.75, "
            "delta_var=0.1, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
            checks_var="CHECKS_ARIMAX_GARCH",
        ),
    ),
    (
        "### C-D.3 — ARIMAX-GARCH: efecto solo en la varianza (beta_mean = 0)\n\n"
        "**DGP:** $Y_t = 0.4\\,Y_{t-1} + 0\\cdot X_t + \\varepsilon_t$, varianza con $\\delta_{\\text{var}} = 0.3$.\n\n"
        "**Hipotesis:** la covariable no informa el nivel pero si la volatilidad. SARIMAX la incluye en la media y el coef estimado deberia ir a cero. Chronos debe identificar el canal de varianza para mejorar intervalos.\n",
        exp_uni_cell(
            "C-D.3", "ARIMAX-GARCH solo varianza",
            "ARIMAX_GARCH_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), name_suffix='con X'),\n        chronos_cov1",
            "phi=0.4, beta_mean=0.0, omega=0.1, alpha=0.1, beta_garch=0.75, "
            "delta_var=0.3, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_1,
            checks_var="CHECKS_ARIMAX_GARCH",
        ),
    ),
]

# ── C-E (VARX bivariado, multivariado) ──────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-E.1 — VARX bivariado baseline\n\n"
        "**DGP:** $A = \\begin{pmatrix} 0.5 & 0.1 \\\\ 0.1 & 0.5 \\end{pmatrix}$, "
        "$\\boldsymbol{\\gamma} = (0.5, 0.3)$, $\\Sigma$ con correlacion 0.3 off-diag.\n\n"
        "**Hipotesis:** dependencia cruzada moderada y efecto exogeno asimetrico. Replica Exp 3.4.\n",
        exp_mv_cell(
            "C-E.1", "VARX baseline",
            "seed=SEED, A=[[0.5, 0.1], [0.1, 0.5]], gamma=[0.5, 0.3], "
            "Sigma=[[1.0, 0.3], [0.3, 1.0]], sigma_x=1.0, rho_x=0.7",
            "VARMAXModel(1),\n        chronos_mv_cov1",
            "",  # dgp_params vacio — VARX lleva params en el constructor
            VARMAX_NAME_1,
        ),
    ),
    (
        "### C-E.2 — VARX dependencia y efecto exogeno fuertes\n\n"
        "**DGP:** $A = \\begin{pmatrix} 0.6 & 0.3 \\\\ 0.3 & 0.6 \\end{pmatrix}$, "
        "$\\boldsymbol{\\gamma} = (0.8, 0.6)$.\n\n"
        "**Hipotesis:** ambas fuentes de senal son fuertes simultaneamente. VARMAX deberia dominar (esta correctamente especificado); Chronos joint pone a prueba si captura la dependencia cruzada y el exogeno conjuntamente.\n",
        exp_mv_cell(
            "C-E.2", "VARX dependencia fuerte",
            "seed=SEED, A=[[0.6, 0.3], [0.3, 0.6]], gamma=[0.8, 0.6], "
            "Sigma=[[1.0, 0.3], [0.3, 1.0]], sigma_x=1.0, rho_x=0.7",
            "VARMAXModel(1),\n        chronos_mv_cov1",
            "",
            VARMAX_NAME_1,
        ),
    ),
    (
        "### C-E.3 — VARX efecto exogeno debil\n\n"
        "**DGP:** $A$ baseline, $\\boldsymbol{\\gamma} = (0.2, 0.2)$.\n\n"
        "**Hipotesis:** la dinamica cruzada de $\\mathbf{Y}$ domina sobre $X$; testea si Chronos joint distingue correctamente la senal endogena del aporte marginal del exogeno cuando este es debil.\n",
        exp_mv_cell(
            "C-E.3", "VARX efecto exogeno debil",
            "seed=SEED, A=[[0.5, 0.1], [0.1, 0.5]], gamma=[0.2, 0.2], "
            "Sigma=[[1.0, 0.3], [0.3, 1.0]], sigma_x=1.0, rho_x=0.7",
            "VARMAXModel(1),\n        chronos_mv_cov1",
            "",
            VARMAX_NAME_1,
        ),
    ),
]

# ── C-F (ADL-ECM cointegracion) ─────────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-F.1 — Cointegracion ADL-ECM, correccion moderada (alpha_ecm = -0.3)\n\n"
        "**DGP:** $\\Delta Y_t = -0.3\\,(Y_{t-1} - X_{t-1}) + \\Delta X_t + \\eta_t$, "
        "$X_t = X_{t-1} + u_t$ (random walk).\n\n"
        "**Hipotesis:** ARDL-ECM (correctamente especificado) deberia dominar a horizontes largos donde la relacion de equilibrio se manifiesta. SARIMAX(1,1,0) ignora la senal de cointegracion; sirve de baseline diferenciado. Replica Exp 3.6.\n",
        # ADL-ECM tiene 3 modelos: ARDL, SARIMAX(1,1,0), Chronos. Usamos ARDL-ECM como classical en la tabla.
        f"""try:
    # C-F.1 -- ADL-ECM alpha_ecm=-0.3
    dgp_C_F_1 = ADL_ECM_DGP(seed=SEED)
    make_models_C_F_1 = lambda T: [
        ARDLModel(),
        SARIMAXModel((1, 1, 0), name_suffix='con X'),
        chronos_cov1,
    ]
    dgp_params_C_F_1 = dict(alpha_ecm=-0.3, sigma=1.0, sigma_x=1.0)
    verify_dgp_cov("C-F.1 -- ADL-ECM alpha_ecm=-0.3", dgp_C_F_1,
                   dgp_params_C_F_1, CHECKS_ADL_ECM)
    res_C_F_1 = run_exp_cov(
        dgp_C_F_1, make_models_C_F_1, dgp_params_C_F_1,
        exp_id="C-F.1",
    )
    log("\\n" + "="*60 + "\\nC-F.1 -- ADL-ECM alpha_ecm=-0.3\\n" + "="*60)
    build_grid_table(res_C_F_1, classical_name="ARDL-ECM")
    plot_simulation_cov(dgp_C_F_1, make_models_C_F_1(200), dgp_params_C_F_1,
                        title="C-F.1 -- ADL-ECM alpha_ecm=-0.3")
except Exception as _exc:
    log("\\n" + "!"*60)
    log("[CELDA C-F.1 FALLO] " + type(_exc).__name__ + ": " + str(_exc))
    log("!"*60)
    log(traceback.format_exc())
""",
    ),
    (
        "### C-F.2 — Cointegracion ADL-ECM, correccion rapida (alpha_ecm = -0.6)\n\n"
        "**DGP:** mismo sistema con $\\alpha_{\\text{ecm}} = -0.6$ — la desviacion del equilibrio se corrige el doble de rapido.\n\n"
        "**Hipotesis:** la senal de cointegracion es mas fuerte; la ventaja de ARDL-ECM sobre SARIMAX(1,1,0) y Chronos deberia ser aun mayor.\n",
        f"""try:
    # C-F.2 -- ADL-ECM alpha_ecm=-0.6
    dgp_C_F_2 = ADL_ECM_DGP(seed=SEED)
    make_models_C_F_2 = lambda T: [
        ARDLModel(),
        SARIMAXModel((1, 1, 0), name_suffix='con X'),
        chronos_cov1,
    ]
    dgp_params_C_F_2 = dict(alpha_ecm=-0.6, sigma=1.0, sigma_x=1.0)
    verify_dgp_cov("C-F.2 -- ADL-ECM alpha_ecm=-0.6", dgp_C_F_2,
                   dgp_params_C_F_2, CHECKS_ADL_ECM)
    res_C_F_2 = run_exp_cov(
        dgp_C_F_2, make_models_C_F_2, dgp_params_C_F_2,
        exp_id="C-F.2",
    )
    log("\\n" + "="*60 + "\\nC-F.2 -- ADL-ECM alpha_ecm=-0.6\\n" + "="*60)
    build_grid_table(res_C_F_2, classical_name="ARDL-ECM")
    plot_simulation_cov(dgp_C_F_2, make_models_C_F_2(200), dgp_params_C_F_2,
                        title="C-F.2 -- ADL-ECM alpha_ecm=-0.6")
except Exception as _exc:
    log("\\n" + "!"*60)
    log("[CELDA C-F.2 FALLO] " + type(_exc).__name__ + ": " + str(_exc))
    log("!"*60)
    log(traceback.format_exc())
""",
    ),
]

# ── C-G (Tendencia deterministica) ──────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-G.1 — Tendencia leve (delta = 0.05) + cov medio (beta = 0.5)\n\n"
        "**DGP:** $Y_t = 0.05\\,t + 0.6\\,Y_{t-1}^{stat} + 0.5\\,X_t + \\varepsilon_t$, "
        "con $X_t = 0.7\\,X_{t-1} + \\eta_t$.\n\n"
        "**Hipotesis:** SARIMAX con `trend='ct'` captura el drift; Chronos debe inferir la tendencia desde el contexto. Validacion MC mostro que omitir la tendencia (`trend='c'`) cuesta **+44% en RMSE** sobre 24 horizontes.\n",
        exp_uni_cell(
            "C-G.1", "Tendencia leve delta=0.05",
            "ARIMAX_TREND_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), trend='ct', name_suffix='con X+trend'),\n        chronos_cov1",
            "phi=0.6, beta=0.5, alpha=0.0, delta=0.05, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_TREND,
            checks_var="CHECKS_ARIMAX_TREND",
        ),
    ),
    (
        "### C-G.2 — Tendencia fuerte (delta = 0.10) + cov medio\n\n"
        "**DGP:** $\\delta = 0.10$ (el doble que C-G.1), $\\beta = 0.5$.\n\n"
        "**Hipotesis:** drift mas pronunciado — la senal de tendencia domina el forecast. MC mostro que misspecificar tendencia cuesta **+114%** aqui. Mide la capacidad de Chronos de extrapolar trayectorias con pendiente positiva.\n",
        exp_uni_cell(
            "C-G.2", "Tendencia fuerte delta=0.10",
            "ARIMAX_TREND_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), trend='ct', name_suffix='con X+trend'),\n        chronos_cov1",
            "phi=0.6, beta=0.5, alpha=0.0, delta=0.10, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_TREND,
            checks_var="CHECKS_ARIMAX_TREND",
        ),
    ),
    (
        "### C-G.3 — Tendencia leve + cov fuerte (beta = 0.8)\n\n"
        "**DGP:** $\\delta = 0.05$, $\\beta = 0.8$.\n\n"
        "**Hipotesis:** combinacion de las dos senales fuertes — tendencia + cov dominante. Compara con C-A.1 ($\\beta=0.8$ sin tendencia): la presencia del drift puede cambiar el patron de cruce Chronos vs SARIMAX observado.\n",
        exp_uni_cell(
            "C-G.3", "Tendencia leve + cov fuerte",
            "ARIMAX_TREND_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), trend='ct', name_suffix='con X+trend'),\n        chronos_cov1",
            "phi=0.6, beta=0.8, alpha=0.0, delta=0.05, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_TREND,
            checks_var="CHECKS_ARIMAX_TREND",
        ),
    ),
]

# ── C-H (Estacionalidad) ────────────────────────────────────────────────────
EXPERIMENTS += [
    (
        "### C-H.1 — Estacionalidad trimestral (s=4), cov medio\n\n"
        "**DGP:** SARIMA(1,0,0)(1,0,0)[4] multiplicativo: $\\phi=0.3$, $\\Phi=0.7$, $\\beta=0.5$, $\\rho_x=0.7$.\n\n"
        "**Hipotesis:** SARIMAX con `seasonal_order=(1,0,0,4)` esta correctamente especificado. Chronos debe identificar el ciclo $s=4$ desde el contexto. Validacion MC mostro **+7%** en RMSE para el bien especificado.\n",
        exp_uni_cell(
            "C-H.1", "Estacional s=4 beta=0.5",
            "SARIMAX_SEASONAL_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), seasonal_order=(1, 0, 0, 4), name_suffix='con X'),\n        chronos_cov1",
            "s=4, phi=0.3, Phi=0.7, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_SEAS_4,
            checks_var="CHECKS_ARIMAX_SEASONAL",
        ),
    ),
    (
        "### C-H.2 — Estacionalidad trimestral (s=4), cov fuerte (beta = 0.8)\n\n"
        "**DGP:** identico a C-H.1 pero $\\beta = 0.8$.\n\n"
        "**Hipotesis:** la combinacion de estacionalidad fuerte ($\\Phi=0.7$) + cov fuerte testea si Chronos puede aprovechar ambas senales simultaneamente con muestra larga (T=200), como en C-A.1.\n",
        exp_uni_cell(
            "C-H.2", "Estacional s=4 beta=0.8",
            "SARIMAX_SEASONAL_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), seasonal_order=(1, 0, 0, 4), name_suffix='con X'),\n        chronos_cov1",
            "s=4, phi=0.3, Phi=0.7, beta=0.8, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_SEAS_4,
            checks_var="CHECKS_ARIMAX_SEASONAL",
        ),
    ),
    (
        "### C-H.3 — Estacionalidad mensual (s=12), cov medio\n\n"
        "**DGP:** SARIMA(1,0,0)(1,0,0)[12] multiplicativo. **T_list restringido a [50, 100, 200]** (T=25 da solo ~2 ciclos estacionales, insuficiente para identificar $\\Phi$).\n\n"
        "**Hipotesis:** estacionalidad de baja frecuencia con muchos parametros efectivos. Modelos estacionales tipicamente necesitan T >> 2s para estimacion confiable.\n",
        exp_uni_cell(
            "C-H.3", "Estacional s=12 beta=0.5",
            "SARIMAX_SEASONAL_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), seasonal_order=(1, 0, 0, 12), name_suffix='con X'),\n        chronos_cov1",
            "s=12, phi=0.3, Phi=0.7, beta=0.5, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_SEAS_12,
            checks_var="CHECKS_ARIMAX_SEASONAL",
            T_list_override="[50, 100, 200]",
        ),
    ),
    (
        "### C-H.4 — Estacionalidad mensual (s=12), cov fuerte (beta = 0.8)\n\n"
        "**DGP:** identico a C-H.3 pero $\\beta = 0.8$. Mismo T_list restringido.\n\n"
        "**Hipotesis:** cov fuerte combinada con estacionalidad de baja frecuencia — el caso mas exigente del bloque. Verifica si Chronos extrae beneficio de la cov cuando la senal estacional es periodica y larga.\n",
        exp_uni_cell(
            "C-H.4", "Estacional s=12 beta=0.8",
            "SARIMAX_SEASONAL_DGP", "seed=SEED",
            "SARIMAXModel((1, 0, 0), seasonal_order=(1, 0, 0, 12), name_suffix='con X'),\n        chronos_cov1",
            "s=12, phi=0.3, Phi=0.7, beta=0.8, sigma_y=1.0, sigma_x=1.0, rho_x=0.7",
            SARIMAX_NAME_SEAS_12,
            checks_var="CHECKS_ARIMAX_SEASONAL",
            T_list_override="[50, 100, 200]",
        ),
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# Summary cell
# ────────────────────────────────────────────────────────────────────────────
SUMMARY = """---
## Resumen

25 experimentos x 4 valores de T x 1 R = 98 corridas Monte Carlo (R=500 cada una;
C-H.3/4 corren solo 3 T cada uno => 8 corridas en vez de 16).
Tiempo estimado: ~4-7 h en CPU (Vertex AI free tier), considerablemente menor en GPU.

**Cobertura de la grilla:**

| Bloque | Experimentos | Variacion principal                  |
|--------|--------------|--------------------------------------|
| C-A    | 5            | beta (0.2/0.5/0.8), rho_x (0/0.7/0.95)|
| C-B    | 2            | phi (0.9 / -0.6) — dinamica AR rica  |
| C-C    | 3            | n_covariables y configuracion        |
| C-D    | 3            | canal (media / varianza / ambos)     |
| C-E    | 3            | gamma y matriz A (multivariado)      |
| C-F    | 2            | alpha_ecm (cointegracion)            |
| C-G    | 3            | delta (tendencia deterministica)     |
| C-H    | 4            | s (4 / 12), beta (0.5 / 0.8) estacional |

**Output:**
- CSV por experimento en `results/covariate_v5_vertexai/exp_*.csv`
- Log de ejecucion en `results/covariate_v5_vertexai/run_*.log`

**Para re-ejecutar:** borrar el CSV correspondiente y volver a correr la celda. Los demas se cargaran desde cache.
"""


# ────────────────────────────────────────────────────────────────────────────
# Build cells list
# ────────────────────────────────────────────────────────────────────────────
cells = [md(TITLE), code(SETUP), code(HELPERS)]

# Block C-A
cells.append(md(BLOCK_CA))
for intro_md, code_src in EXPERIMENTS[:5]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-B
cells.append(md(BLOCK_CB))
for intro_md, code_src in EXPERIMENTS[5:7]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-C
cells.append(md(BLOCK_CC))
for intro_md, code_src in EXPERIMENTS[7:10]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-D
cells.append(md(BLOCK_CD))
for intro_md, code_src in EXPERIMENTS[10:13]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-E
cells.append(md(BLOCK_CE))
for intro_md, code_src in EXPERIMENTS[13:16]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-F
cells.append(md(BLOCK_CF))
for intro_md, code_src in EXPERIMENTS[16:18]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-G (trend)
cells.append(md(BLOCK_CG))
for intro_md, code_src in EXPERIMENTS[18:21]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Block C-H (seasonal)
cells.append(md(BLOCK_CH))
for intro_md, code_src in EXPERIMENTS[21:25]:
    cells.append(md(intro_md))
    cells.append(code(code_src))

# Summary
cells.append(md(SUMMARY))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {NB_PATH}")
print(f"Total cells: {len(cells)}")
print(f"Total experiments: {len(EXPERIMENTS)}")
