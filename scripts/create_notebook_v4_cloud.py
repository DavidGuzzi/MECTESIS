"""
Genera notebooks/experimentos_univariados_v4_cloud.ipynb
Fork de create_notebook_v3_cloud.py con:
  - verify_dgp + checks estadisticos antes de cada experimento
  - Logging de verificaciones al .log (usa log() en lugar de print())
  - Mismos 97 DGPs, mismo cache results/univariate_v3/

Ejecutar: python scripts/create_notebook_v4_cloud.py
"""

from pathlib import Path
import nbformat as nbf

# ─── helpers ─────────────────────────────────────────────────────────────────

def md(text: str):
    return nbf.v4.new_markdown_cell(text)

def code(src: str):
    return nbf.v4.new_code_cell(src.strip())

# ─── contenido de células ─────────────────────────────────────────────────────

SETUP = """\
import warnings
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
from statsmodels.stats.diagnostic import acorr_ljungbox

from mectesis.dgp import (
    RandomWalk, SeasonalDGP,
    AR1ARCH, AR1GARCH,
    LocalLevelDGP, LocalTrendDGP, DampedTrendDGP, LocalLevelSeasonalDGP,
    ARpDGP, MAqDGP, ARMApqDGP, ARMApqWithTrendDGP,
    SETARDGp, LSTARDGp, ESTARDGp,
)
from mectesis.models import (
    ARIMAModel, ARIMAWithTrendModel, SARIMAModel,
    ARARCHModel, ARGARCHModel,
    ETSModel, ThetaModel, ChronosModel,
)
from mectesis.simulation import MonteCarloEngine

SEED    = 3649
H_BY_T  = {25: 6, 50: 18, 100: 24, 200: 24}
H_MAX   = 24
R_LIST  = [500]
T_LIST  = [25, 50, 100, 200]
RESULTS = Path("results/univariate_v3")
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

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Cargando Chronos-2 en {device} (puede tardar ~30 s la primera vez)...")
chronos = ChronosModel(device=device)
log("Chronos-2 listo.")
"""

HELPERS = """\
# ─── Funciones auxiliares ────────────────────────────────────────────────────

def _cache_path(exp_id: str, T: int, R: int) -> Path:
    return RESULTS / f"exp_{exp_id.replace('.', '_')}_T{T}_R{R}.csv"


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


def run_exp(dgp, make_models_fn, dgp_params, exp_id,
            T_list=T_LIST, R_list=R_LIST, H_by_T=None, seed=SEED):
    if H_by_T is None:
        H_by_T = H_BY_T
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
            models = make_models_fn(T)
            engine = MonteCarloEngine(dgp, models, seed=seed)
            t0 = time.time()
            results = engine.run_monte_carlo(R, T, h, dgp_params, verbose=False)
            log(f"  T={T} H={h}, R={R}: OK ({time.time()-t0:.0f}s)")
            _save_results(results, cache)
            all_results[(T, R)] = results
    return all_results


# ─── Funciones v3 ────────────────────────────────────────────────────────────

BLOCK_DEFS = [("C", 1, 6), ("M", 7, 18), ("L", 19, 24)]
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
                     chronos_name: str = "Chronos-2"):
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


def plot_simulation_v3(dgp, models, dgp_params, title="", T_vis=100, seed=SEED):
    H_vis   = H_BY_T.get(T_vis, H_MAX)
    old_rng = dgp.rng
    dgp.rng = np.random.default_rng(seed + 99991)
    y       = dgp.simulate(T=T_vis, **dgp_params)
    dgp.rng = old_rng

    split   = T_vis - H_vis
    y_train = y[:split]
    y_test  = y[split:]
    t_train = np.arange(split)
    t_test  = np.arange(split, T_vis)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_train, y_train, color="gray", lw=1.5, label="Historico")
    ax.axvline(split - 0.5, color="black", ls="--", lw=1, alpha=0.6)
    ax.plot(t_test, y_test, color="black", lw=1.5, marker="o", ms=3, label="Observado")

    palette = ["steelblue", "darkorange", "seagreen", "purple"]
    for i, model in enumerate(models):
        try:
            model.fit(y_train)
            fcst = model.forecast(H_vis)
            c = palette[i % len(palette)]
            ax.plot(t_test, fcst, color=c, lw=1.5, ls="--", marker="s", ms=3, label=model.name)
            if getattr(model, "supports_intervals", False):
                lo, hi = model.forecast_intervals(H_vis, level=0.80)
                ax.fill_between(t_test, lo, hi, color=c, alpha=0.15)
        except Exception as e:
            log(f"  [plot] {model.name} fallo: {e}")

    ax.set(title=title, xlabel="t", ylabel="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ─── Funciones de verificacion DGP ───────────────────────────────────────────

def verify_dgp(label, dgp, dgp_params, classical_model, checks):
    log(f"{'─'*60}")
    log(f"VERIFICACION DGP: {label}")
    log(f"{'─'*60}")
    dgp_copy = copy.deepcopy(dgp)
    dgp_copy.rng = np.random.default_rng(7777)
    try:
        y_long = dgp_copy.simulate(T=1000, **dgp_params)
    except Exception as e:
        log(f"  [FAIL] simulate() lanzo excepcion: {e}")
        return
    n_fail = 0
    for check_name, check_fn in checks:
        try:
            ok, msg = check_fn(y_long, dgp, dgp_params, classical_model)
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


def chk_stationary(y, *_):
    pval = adfuller(y, autolag="AIC")[1]
    return pval < 0.05, f"ADF p={pval:.4f} (umbral 0.05)"

def chk_nonstationary(y, *_):
    pval = adfuller(y, autolag="AIC")[1]
    return pval > 0.10, f"ADF p={pval:.4f} (se espera >0.10)"

def chk_zero_mean(y, *_):
    mu = y.mean(); sigma = y.std(); T = len(y)
    tol = 3.0 * sigma / np.sqrt(T)
    return abs(mu) < tol, f"media={mu:.4f}, tol=+/-{tol:.4f}"

def chk_acf_lag1(y, dgp, dgp_params, *_):
    props = dgp.get_theoretical_properties(**dgp_params) if dgp_params else dgp.get_theoretical_properties()
    phi1 = props.get("phis", [None])[0] if props.get("phis") else None
    if phi1 is None:
        return True, "no aplica (sin phis)"
    acf_vals = acf(y, nlags=1, fft=True)
    emp = acf_vals[1]
    ok = abs(emp - phi1) < 0.15
    return ok, f"ACF[1]={emp:.4f}, phi1_DGP={phi1:.4f}, dif={abs(emp-phi1):.4f}"

def chk_ma_cutoff(y, dgp, dgp_params, *_):
    props = dgp.get_theoretical_properties(**dgp_params) if dgp_params else dgp.get_theoretical_properties()
    thetas = props.get("thetas", [])
    q = len(thetas)
    if q == 0:
        return True, "no aplica (sin thetas)"
    T = len(y)
    thr = 2.0 / np.sqrt(T)
    acf_vals = acf(y, nlags=q+2, fft=True)
    val_after = abs(acf_vals[q+1])
    ok = val_after < thr
    return ok, f"|ACF[{q+1}]|={val_after:.4f}, umbral=2/sqrt(T)={thr:.4f}"

def chk_trend_slope(y, dgp, dgp_params, *_):
    delta = dgp_params.get("delta", None)
    if delta is None:
        return True, "no aplica (sin delta)"
    T = len(y)
    slope = np.polyfit(np.arange(T), y, 1)[0]
    ok = abs(slope - delta) < 0.015
    return ok, f"pendiente_OLS={slope:.5f}, delta_DGP={delta:.5f}, dif={abs(slope-delta):.5f}"

def chk_arch_effects(y, *_):
    y2 = y ** 2
    lb = acorr_ljungbox(y2, lags=[10], return_df=True)
    pval = float(lb["lb_pvalue"].iloc[0])
    return pval < 0.05, f"LB(10) sobre y^2: p={pval:.4f} (se espera <0.05)"

def chk_rw_variance_growth(y, *_):
    T = len(y)
    v1 = np.var(y[:T//4])
    v2 = np.var(y[T//2:])
    ok = v2 > 2.0 * v1
    return ok, f"var(primera cuarta)={v1:.4f}, var(segunda mitad)={v2:.4f}"

def chk_seasonal_acf(y, dgp, dgp_params, *_):
    s = dgp_params.get("s", 4)
    T = len(y)
    thr = 2.0 / np.sqrt(T)
    acf_vals = acf(y, nlags=s+1, fft=True)
    val_s = abs(acf_vals[s])
    ok = val_s > thr
    return ok, f"|ACF[{s}]|={val_s:.4f}, umbral=2/sqrt(T)={thr:.4f}"

def chk_fit_classical(y, dgp, dgp_params, model):
    try:
        m = copy.deepcopy(model)
        m.fit(y[:800])
        fc = m.forecast(horizon=6)
        ok = len(fc) == 6 and not np.any(np.isnan(fc))
        return ok, f"fit+forecast OK, primeros valores: {fc[:3].round(4)}"
    except Exception as e:
        return False, str(e)

def chk_arma_aic(y, dgp, dgp_params, model):
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    props = dgp.get_theoretical_properties(**dgp_params) if dgp_params else dgp.get_theoretical_properties()
    p = len(props.get("phis", []))
    q = len(props.get("thetas", []))
    try:
        aic_arma = StatsARIMA(y, order=(p, 0, q)).fit(disp=False).aic
        aic_ar   = StatsARIMA(y, order=(p, 0, 0)).fit(disp=False).aic if p > 0 else np.inf
        aic_ma   = StatsARIMA(y, order=(0, 0, q)).fit(disp=False).aic if q > 0 else np.inf
        baseline = min(aic_ar, aic_ma)
        ok = aic_arma < baseline
        return ok, f"AIC_ARMA={aic_arma:.1f}, AIC_baseline={baseline:.1f}"
    except Exception as e:
        return False, str(e)

def chk_theta_fit(y, dgp, dgp_params, model):
    try:
        m = copy.deepcopy(model)
        m.fit(y[:800])
        fc = m.forecast(horizon=6)
        return True, f"OK, primeros valores: {fc[:3].round(4)}"
    except Exception as e:
        return False, str(e)

# ── grupos de checks por tipo ─────────────────────────────────────────────────

CHECKS_AR = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("Media ~ 0",             chk_zero_mean),
    ("ACF[1] ~ phi1",         chk_acf_lag1),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_AR_TREND = [
    ("Pendiente OLS ~ delta", chk_trend_slope),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_MA = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("Media ~ 0",             chk_zero_mean),
    ("Corte ACF en lag q",    chk_ma_cutoff),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_MA_TREND = [
    ("Pendiente OLS ~ delta", chk_trend_slope),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_ARMA = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("AIC ARMA < baseline",   chk_arma_aic),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_RW = [
    ("No estacionariedad",    chk_nonstationary),
    ("Varianza crece con t",  chk_rw_variance_growth),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_ARCH = [
    ("Estacionariedad media", chk_stationary),
    ("Efectos ARCH (LB y^2)", chk_arch_effects),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_SAR = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("ACF estacional signif.", chk_seasonal_acf),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_ETS = [
    ("No estacionariedad",    chk_nonstationary),
    ("Ajuste modelo clasico", chk_fit_classical),
]
CHECKS_THETA = [
    ("Ajuste Theta",          chk_theta_fit),
]
CHECKS_NONLINEAR = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("Ajuste modelo clasico", chk_fit_classical),
]
"""


# ─── Specs de experimentos ────────────────────────────────────────────────────

ARMA_SPECS = [
    ("1",  "AR(1) rho=0.30",    [0.30],                       [],                              (1, 0, 0)),
    ("2",  "AR(1) rho=0.90",    [0.90],                       [],                              (1, 0, 0)),
    ("3",  "AR(2) rho=0.30",    [0.30,  0.10],                [],                              (2, 0, 0)),
    ("4",  "AR(2) rho=0.90",    [0.90, -0.20],                [],                              (2, 0, 0)),
    ("5",  "AR(3) rho=0.30",    [0.30,  0.10,  0.05],         [],                              (3, 0, 0)),
    ("6",  "AR(3) rho=0.90",    [0.90, -0.20,  0.10],         [],                              (3, 0, 0)),
    ("7",  "AR(4) rho=0.30",    [0.30,  0.10,  0.05,  0.02],  [],                              (4, 0, 0)),
    ("8",  "AR(4) rho=0.90",    [0.90, -0.20,  0.10, -0.05],  [],                              (4, 0, 0)),
    ("9",  "MA(1) theta=0.30",  [],      [0.30],                                               (0, 0, 1)),
    ("10", "MA(1) theta=0.90",  [],      [0.90],                                               (0, 0, 1)),
    ("11", "MA(2) theta=0.30",  [],      [0.30,  0.10],                                        (0, 0, 2)),
    ("12", "MA(2) theta=0.90",  [],      [0.90,  0.10],                                        (0, 0, 2)),
    ("13", "MA(3) theta=0.30",  [],      [0.30,  0.10, -0.05],                                 (0, 0, 3)),
    ("14", "MA(3) theta=0.90",  [],      [0.90,  0.10, -0.05],                                 (0, 0, 3)),
    ("15", "MA(4) theta=0.30",  [],      [0.30,  0.10, -0.05,  0.02],                          (0, 0, 4)),
    ("16", "MA(4) theta=0.90",  [],      [0.90,  0.10, -0.05,  0.02],                          (0, 0, 4)),
    ("17", "ARMA(1,1) rho=0.30", [0.30],           [0.10],                                    (1, 0, 1)),
    ("18", "ARMA(1,1) rho=0.90", [0.90],           [0.30],                                    (1, 0, 1)),
    ("19", "ARMA(2,2) rho=0.30", [0.30,  0.10],    [0.10,  0.05],                             (2, 0, 2)),
    ("20", "ARMA(2,2) rho=0.90", [0.90, -0.20],    [0.30, -0.10],                             (2, 0, 2)),
    ("21", "ARMA(1,4) rho=0.30", [0.30],           [0.10,  0.05, -0.03,  0.01],               (1, 0, 4)),
    ("22", "ARMA(1,4) rho=0.90", [0.90],           [0.30, -0.10,  0.05, -0.02],               (1, 0, 4)),
    ("23", "ARMA(4,1) rho=0.30", [0.30,  0.10,  0.05,  0.02],  [0.10],                        (4, 0, 1)),
    ("24", "ARMA(4,1) rho=0.90", [0.90, -0.20,  0.10, -0.05],  [0.30],                        (4, 0, 1)),
]


def _dgp_class(phis, thetas):
    if not thetas:
        return "ARpDGP", f"ARpDGP(phis={phis!r}, sigma=1.0, seed=SEED)"
    if not phis:
        return "MAqDGP", f"MAqDGP(thetas={thetas!r}, sigma=1.0, seed=SEED)"
    return "ARMApqDGP", f"ARMApqDGP(phis={phis!r}, thetas={thetas!r}, sigma=1.0, seed=SEED)"


def _arima_name(order):
    p, d, q = order
    return f"ARIMA({p}, {d}, {q})"


def _checks_var_A(phis, thetas):
    if not thetas:
        return "CHECKS_AR"
    if not phis:
        return "CHECKS_MA"
    return "CHECKS_ARMA"


def _checks_var_B(phis, thetas):
    if not thetas:
        return "CHECKS_AR_TREND"
    if not phis:
        return "CHECKS_MA_TREND"
    return "CHECKS_ARMA"


def safe_wrap(exp_id: str, body: str) -> str:
    """Envuelve el cuerpo de una celda en try/except que loguea traceback y continua."""
    body_stripped = body.rstrip("\n")
    indented = "\n".join(("    " + line) if line else "" for line in body_stripped.splitlines())
    return (
        "try:\n"
        + indented + "\n"
        + "except Exception as _exc:\n"
        + '    log("\\n" + "!"*60)\n'
        + '    log("[CELDA ' + exp_id + ' FALLO] " + type(_exc).__name__ + ": " + str(_exc))\n'
        + '    log("!"*60)\n'
        + "    log(traceback.format_exc())\n"
    )


def safe_code_cell(exp_id: str, body: str):
    """Helper: devuelve un nbformat code cell con el body envuelto en try/except."""
    return code(safe_wrap(exp_id, body))


def exp_cell_A(suf, desc, phis, thetas, order):
    cl_name = _arima_name(order)
    _, dgp_expr = _dgp_class(phis, thetas)
    checks_var = _checks_var_A(phis, thetas)
    body = f"""\
# A.{suf} -- {desc}
cl  = ARIMAModel({order!r})
dgp = {dgp_expr}
verify_dgp("A.{suf} -- {desc}", dgp, {{}}, cl, {checks_var})
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {{}}, exp_id="A.{suf}")
log("\\n" + "="*60 + "\\nA.{suf} -- {desc}\\n" + "="*60)
build_grid_table(res, classical_name="{cl_name}")
plot_simulation_v3(dgp, [cl, chronos], {{}}, title="A.{suf} -- {desc}")
"""
    return safe_wrap(f"A.{suf}", body)


def exp_cell_B(suf, desc, phis, thetas, order, delta, delta_name):
    cl_name = _arima_name(order) + "+trend"
    checks_var = _checks_var_B(phis, thetas)
    body = f"""\
# B.{suf} -- {desc}  [tendencia {delta_name} delta={delta}]
cl  = ARIMAWithTrendModel({order!r}, trend="ct")
dgp = ARMApqWithTrendDGP(phis={phis!r}, thetas={thetas!r}, delta={delta}, sigma=1.0, seed=SEED)
verify_dgp("B.{suf} -- {desc} (delta={delta})", dgp, {{}}, cl, {checks_var})
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {{}}, exp_id="B.{suf}")
log("\\n" + "="*60 + "\\nB.{suf} -- {desc} (delta={delta})\\n" + "="*60)
build_grid_table(res, classical_name="{cl_name}")
plot_simulation_v3(dgp, [cl, chronos], {{}}, title="B.{suf} -- {desc} (delta={delta})")
"""
    return safe_wrap(f"B.{suf}", body)


# ─── Construir celdas ─────────────────────────────────────────────────────────

cells = []

cells.append(md(
    "# Experimentos Univariados v4 Cloud (Vertex AI)\n\n"
    "**Tesis MEC** — 97 DGPs x T in {25,50,100,200} x R=500  \n"
    "**Verificacion DGP:** cada experimento incluye seccion PASS/FAIL antes del Monte Carlo  \n"
    "**Horizonte por T:** T=25->H=6 * T=50->H=18 * T=100,200->H=24  \n"
    "**Metricas:** Bias, Varianza, RMSE, CRPS  \n"
    "**Bloques:** Corto h=1-6 * Medio h=7-18 * Largo h=19-24  \n"
    "**Logging:** dual stdout + `results/univariate_v3/run_YYYYMMDD_HHMMSS.log`  \n"
    "**Resultados:** `results/univariate_v3/` — si existen se cargan sin re-simular"
))

cells.append(code(SETUP))
cells.append(code(HELPERS))

# ── BLOQUE A ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque A — ARMA sin tendencia (24 experimentos)\n\n"
    "DGPs: AR(1-4) * MA(1-4) * ARMA(1,1) * ARMA(2,2) * ARMA(1,4) * ARMA(4,1)  \n"
    "Modelo clasico: ARIMA(p,0,q) correctamente especificado vs Chronos-2"
))

for suf, desc, phis, thetas, order in ARMA_SPECS:
    cells.append(code(exp_cell_A(suf, desc, phis, thetas, order)))

# ── BLOQUE B ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque B — ARMA con tendencia deterministica (48 experimentos)\n\n"
    "DGP: `Y_t = alpha + delta*t + ARMA_t`  \n"
    "Tendencia leve: delta=0.02 (B.1-B.24) | Tendencia fuerte: delta=0.10 (B.25-B.48)  \n"
    "Modelo clasico: ARIMA(p,0,q)+trend (trend='ct')"
))

for i, (suf, desc, phis, thetas, order) in enumerate(ARMA_SPECS):
    b_suf = str(i + 1)
    cells.append(code(exp_cell_B(b_suf, desc, phis, thetas, order, 0.02, "leve")))

for i, (suf, desc, phis, thetas, order) in enumerate(ARMA_SPECS):
    b_suf = str(i + 25)
    cells.append(code(exp_cell_B(b_suf, desc, phis, thetas, order, 0.10, "fuerte")))

# ── BLOQUE C ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque C — Random Walk (3 experimentos)\n\n"
    "DGP: `Y_t = drift + Y_{t-1} + eps_t`  \n"
    "Modelo clasico: ARIMA(0,1,0)"
))

cells.append(safe_code_cell("C.1", """\
# C.1 -- RW sin drift
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
verify_dgp("C.1 -- RW sin drift", dgp, {"drift": 0.0, "sigma": 1.0}, cl, CHECKS_RW)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {"drift": 0.0, "sigma": 1.0}, exp_id="C.1")
log("\\n" + "="*60 + "\\nC.1 -- RW sin drift\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"drift": 0.0, "sigma": 1.0}, title="C.1 -- Random Walk sin drift")
"""))

cells.append(safe_code_cell("C.2", """\
# C.2 -- RW drift leve (delta=0.05)
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
verify_dgp("C.2 -- RW drift leve (delta=0.05)", dgp, {"drift": 0.05, "sigma": 1.0}, cl, CHECKS_RW)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {"drift": 0.05, "sigma": 1.0}, exp_id="C.2")
log("\\n" + "="*60 + "\\nC.2 -- RW drift leve (delta=0.05)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"drift": 0.05, "sigma": 1.0}, title="C.2 -- Random Walk drift leve (delta=0.05)")
"""))

cells.append(safe_code_cell("C.3", """\
# C.3 -- RW drift fuerte (delta=0.20)
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
verify_dgp("C.3 -- RW drift fuerte (delta=0.20)", dgp, {"drift": 0.20, "sigma": 1.0}, cl, CHECKS_RW)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {"drift": 0.20, "sigma": 1.0}, exp_id="C.3")
log("\\n" + "="*60 + "\\nC.3 -- RW drift fuerte (delta=0.20)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"drift": 0.20, "sigma": 1.0}, title="C.3 -- Random Walk drift fuerte (delta=0.20)")
"""))

# ── BLOQUE D ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque D — Volatilidad condicional: ARCH/GARCH (4 experimentos)\n\n"
    "DGP: AR(1) en la media + proceso de varianza condicional  \n"
    "Modelo clasico: AR+ARCH o AR+GARCH correctamente especificado"
))

cells.append(safe_code_cell("D.1", """\
# D.1 -- AR(1)-ARCH(1) leve  (alpha=0.10)
cl  = ARARCHModel(ar_lags=1, p=1)
dgp = AR1ARCH(seed=SEED)
verify_dgp("D.1 -- AR(1)-ARCH(1) leve (alpha=0.10)", dgp, {"phi": 0.5, "omega": 0.5, "alpha": 0.10}, cl, CHECKS_ARCH)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.5, "alpha": 0.10}, exp_id="D.1")
log("\\n" + "="*60 + "\\nD.1 -- AR(1)-ARCH(1) leve\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.5, "alpha": 0.10}, title="D.1 -- AR(1)-ARCH(1) leve (alpha=0.10)")
"""))

cells.append(safe_code_cell("D.2", """\
# D.2 -- AR(1)-ARCH(1) fuerte  (alpha=0.50)
cl  = ARARCHModel(ar_lags=1, p=1)
dgp = AR1ARCH(seed=SEED)
verify_dgp("D.2 -- AR(1)-ARCH(1) fuerte (alpha=0.50)", dgp, {"phi": 0.5, "omega": 0.5, "alpha": 0.50}, cl, CHECKS_ARCH)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.5, "alpha": 0.50}, exp_id="D.2")
log("\\n" + "="*60 + "\\nD.2 -- AR(1)-ARCH(1) fuerte\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.5, "alpha": 0.50}, title="D.2 -- AR(1)-ARCH(1) fuerte (alpha=0.50)")
"""))

cells.append(safe_code_cell("D.3", """\
# D.3 -- AR(1)-GARCH(1,1) baja persistencia  (alpha+beta=0.50)
cl  = ARGARCHModel(ar_lags=1, p=1, q=1)
dgp = AR1GARCH(seed=SEED)
verify_dgp("D.3 -- AR(1)-GARCH(1,1) baja persistencia", dgp, {"phi": 0.5, "omega": 0.5, "alpha": 0.10, "beta": 0.40}, cl, CHECKS_ARCH)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.5, "alpha": 0.10, "beta": 0.40}, exp_id="D.3")
log("\\n" + "="*60 + "\\nD.3 -- AR(1)-GARCH(1,1) baja persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.5, "alpha": 0.10, "beta": 0.40}, title="D.3 -- AR(1)-GARCH(1,1) baja persistencia (alpha+beta=0.50)")
"""))

cells.append(safe_code_cell("D.4", """\
# D.4 -- AR(1)-GARCH(1,1) alta persistencia  (alpha+beta=0.95)
cl  = ARGARCHModel(ar_lags=1, p=1, q=1)
dgp = AR1GARCH(seed=SEED)
verify_dgp("D.4 -- AR(1)-GARCH(1,1) alta persistencia", dgp, {"phi": 0.5, "omega": 0.1, "alpha": 0.10, "beta": 0.85}, cl, CHECKS_ARCH)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.1, "alpha": 0.10, "beta": 0.85}, exp_id="D.4")
log("\\n" + "="*60 + "\\nD.4 -- AR(1)-GARCH(1,1) alta persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.1, "alpha": 0.10, "beta": 0.85}, title="D.4 -- AR(1)-GARCH(1,1) alta persistencia (alpha+beta=0.95)")
"""))

# ── BLOQUE E ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque E — ETS y Theta (8 experimentos)\n\n"
    "DGPs de espacio de estados: nivel local, tendencia, estacionalidad  \n"
    "Modelos clasicos: ETS(A,T,S) y Theta"
))

cells.append(safe_code_cell("E.1", """\
# E.1 -- Local Level  ETS(A,N,N)
cl  = ETSModel()
dgp = LocalLevelDGP(seed=SEED)
verify_dgp("E.1 -- Local Level ETS(A,N,N)", dgp, {"sigma_eps": 1.0, "sigma_eta": 0.10}, cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.10}, exp_id="E.1")
log("\\n" + "="*60 + "\\nE.1 -- Local Level ETS(A,N,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.10}, title="E.1 -- Local Level ETS(A,N,N)")
"""))

cells.append(safe_code_cell("E.2", """\
# E.2 -- Local Linear Trend leve  ETS(A,A,N)  sigma_z=0.05
cl  = ETSModel(trend="add")
dgp = LocalTrendDGP(seed=SEED)
verify_dgp("E.2 -- LLT leve ETS(A,A,N)", dgp, {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.05, "b0": 0.1}, cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.05, "b0": 0.1}, exp_id="E.2")
log("\\n" + "="*60 + "\\nE.2 -- LLT leve ETS(A,A,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.05, "b0": 0.1}, title="E.2 -- Local Linear Trend leve (sigma_z=0.05)")
"""))

cells.append(safe_code_cell("E.3", """\
# E.3 -- Local Linear Trend fuerte  ETS(A,A,N)  sigma_z=0.20
cl  = ETSModel(trend="add")
dgp = LocalTrendDGP(seed=SEED)
verify_dgp("E.3 -- LLT fuerte ETS(A,A,N)", dgp, {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.20, "b0": 0.5}, cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.20, "b0": 0.5}, exp_id="E.3")
log("\\n" + "="*60 + "\\nE.3 -- LLT fuerte ETS(A,A,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.20, "b0": 0.5}, title="E.3 -- Local Linear Trend fuerte (sigma_z=0.20)")
"""))

cells.append(safe_code_cell("E.4", """\
# E.4 -- Damped Trend  ETS(A,Ad,N)  phi_d=0.90
cl  = ETSModel(trend="add", damped_trend=True)
dgp = DampedTrendDGP(seed=SEED)
verify_dgp("E.4 -- Damped Trend ETS(A,Ad,N)", dgp, {"phi": 0.9, "sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.1}, cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.9, "sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.1}, exp_id="E.4")
log("\\n" + "="*60 + "\\nE.4 -- Damped Trend ETS(A,Ad,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.9, "sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.1}, title="E.4 -- Damped Trend ETS(A,Ad,N)")
"""))

cells.append(safe_code_cell("E.5", """\
# E.5 -- Seasonal Aditiva s=12  ETS(A,N,A)
cl  = ETSModel(seasonal="add", seasonal_periods=12)
dgp = LocalLevelSeasonalDGP(seed=SEED)
verify_dgp("E.5 -- Seasonal Aditiva s=12", dgp,
           {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1, "sigma_zeta": 0.0, "sigma_omega": 0.05, "b0": 0.0},
           cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1,
               "sigma_zeta": 0.0, "sigma_omega": 0.05, "b0": 0.0},
              exp_id="E.5", T_list=[50, 100, 200])
log("\\n" + "="*60 + "\\nE.5 -- Seasonal Aditiva s=12 ETS(A,N,A)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1, "sigma_zeta": 0.0, "sigma_omega": 0.05, "b0": 0.0}, title="E.5 -- Seasonal Aditiva s=12 ETS(A,N,A)")
"""))

cells.append(safe_code_cell("E.6", """\
# E.6 -- Trend + Seasonal s=12  ETS(A,A,A)
cl  = ETSModel(trend="add", seasonal="add", seasonal_periods=12)
dgp = LocalLevelSeasonalDGP(seed=SEED)
verify_dgp("E.6 -- Trend+Seasonal s=12 ETS(A,A,A)", dgp,
           {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1, "sigma_zeta": 0.05, "sigma_omega": 0.05, "b0": 0.1},
           cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1,
               "sigma_zeta": 0.05, "sigma_omega": 0.05, "b0": 0.1},
              exp_id="E.6", T_list=[50, 100, 200])
log("\\n" + "="*60 + "\\nE.6 -- Trend+Seasonal s=12 ETS(A,A,A)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1, "sigma_zeta": 0.05, "sigma_omega": 0.05, "b0": 0.1}, title="E.6 -- Trend+Seasonal s=12 ETS(A,A,A)")
"""))

cells.append(safe_code_cell("E.7", """\
# E.7 -- Theta leve  (b0=0.10, sigma_z=0.01)
cl  = ThetaModel()
dgp = LocalTrendDGP(seed=SEED)
verify_dgp("E.7 -- Theta leve", dgp, {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.01, "b0": 0.10}, cl, CHECKS_THETA)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.01, "b0": 0.10}, exp_id="E.7")
log("\\n" + "="*60 + "\\nE.7 -- Theta leve\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.01, "b0": 0.10}, title="E.7 -- Theta leve (b0=0.10, sigma_z=0.01)")
"""))

cells.append(safe_code_cell("E.8", """\
# E.8 -- Theta fuerte  (b0=0.50, sigma_z=0.10)
cl  = ThetaModel()
dgp = LocalTrendDGP(seed=SEED)
verify_dgp("E.8 -- Theta fuerte", dgp, {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.10, "b0": 0.50}, cl, CHECKS_THETA)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.10, "b0": 0.50}, exp_id="E.8")
log("\\n" + "="*60 + "\\nE.8 -- Theta fuerte\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.10, "b0": 0.50}, title="E.8 -- Theta fuerte (b0=0.50, sigma_z=0.10)")
"""))

# ── BLOQUE F ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque F — SARIMA (6 experimentos)\n\n"
    "DGP: SAR(1)(1)_s estacionario * (1-L)(1-L^s) integrado  \n"
    "Periodos estacionales: s=4 (trimestral) * s=12 (mensual)"
))

SARIMA_SPECS = [
    ("1", "SAR(1)(1)_4 baja persist.",  4, False, 0.3, 0.3, (1,0,0), (1,0,0,4)),
    ("2", "SAR(1)(1)_4 alta persist.",  4, False, 0.9, 0.6, (1,0,0), (1,0,0,4)),
    ("3", "SAR(1)(1)_12 baja persist.", 12, False, 0.3, 0.3, (1,0,0), (1,0,0,12)),
    ("4", "SAR(1)(1)_12 alta persist.", 12, False, 0.9, 0.6, (1,0,0), (1,0,0,12)),
    ("5", "(1-L)(1-L^4) integrado",     4, True,  None, None, (0,1,0), (0,1,0,4)),
    ("6", "(1-L)(1-L^12) integrado",    12, True,  None, None, (0,1,0), (0,1,0,12)),
]

for suf, desc, s, integrated, phi, Phi, order, sorder in SARIMA_SPECS:
    if integrated:
        dgp_params_str = f'{{"s": {s}, "sigma": 1.0, "integrated": True}}'
    else:
        dgp_params_str = f'{{"phi": {phi}, "Phi": {Phi}, "s": {s}, "sigma": 1.0, "integrated": False}}'
    t_list = "[50, 100, 200]" if s == 12 else "T_LIST"
    src = f"""\
# F.{suf} -- {desc}
cl  = SARIMAModel(order={order!r}, seasonal_order={sorder!r})
dgp = SeasonalDGP(seed=SEED)
verify_dgp("F.{suf} -- {desc}", dgp, {dgp_params_str}, cl, CHECKS_SAR)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {dgp_params_str},
              exp_id="F.{suf}", T_list={t_list})
log("\\n" + "="*60 + "\\nF.{suf} -- {desc}\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {dgp_params_str}, title="F.{suf} -- {desc}")
"""
    cells.append(safe_code_cell(f"F.{suf}", src))

# ── BLOQUE G ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque G — No linealidad de umbral: SETAR / LSTAR / ESTAR (4 experimentos)\n\n"
    "Benchmark clasico: ARIMA(1,0,0) — lineal misspecificado  \n"
    "Pregunta: detecta Chronos-2 la no-linealidad que ARIMA no puede capturar?  \n"
    "- **SETAR**: umbral deterministico y observable  \n"
    "- **LSTAR**: transicion suave asimetrica (ciclos de negocios)  \n"
    "- **ESTAR**: transicion suave simetrica (tipos de cambio con bandas)"
))

cells.append(safe_code_cell("G.1", """\
# G.1 -- SETAR(2;1) baja persistencia  phi1=0.30, phi2=-0.30, k=0
cl  = ARIMAModel((1, 0, 0))
dgp = SETARDGp(phi1=0.30, phi2=-0.30, threshold=0.0, delay=1, sigma=1.0, seed=SEED)
verify_dgp("G.1 -- SETAR(2;1) baja persistencia", dgp, {}, cl, CHECKS_NONLINEAR)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.1")
log("\\n" + "="*60 + "\\nG.1 -- SETAR(2;1) baja persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.1 -- SETAR(2;1) phi1=0.30 / phi2=-0.30")
"""))

cells.append(safe_code_cell("G.2", """\
# G.2 -- SETAR(2;1) alta persistencia  phi1=0.90, phi2=-0.50, k=0
cl  = ARIMAModel((1, 0, 0))
dgp = SETARDGp(phi1=0.90, phi2=-0.50, threshold=0.0, delay=1, sigma=1.0, seed=SEED)
verify_dgp("G.2 -- SETAR(2;1) alta persistencia", dgp, {}, cl, CHECKS_NONLINEAR)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.2")
log("\\n" + "="*60 + "\\nG.2 -- SETAR(2;1) alta persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.2 -- SETAR(2;1) phi1=0.90 / phi2=-0.50")
"""))

cells.append(safe_code_cell("G.3", """\
# G.3 -- LSTAR(1) asimetrico  phi1=0.30, phi2=0.90, gamma=2, c=0
cl  = ARIMAModel((1, 0, 0))
dgp = LSTARDGp(phi1=0.30, phi2=0.90, gamma=2.0, c=0.0, delay=1, sigma=1.0, seed=SEED)
verify_dgp("G.3 -- LSTAR(1) asimetrico", dgp, {}, cl, CHECKS_NONLINEAR)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.3")
log("\\n" + "="*60 + "\\nG.3 -- LSTAR(1) asimetrico\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.3 -- LSTAR(1) phi1=0.30 / phi2=0.90 (gamma=2)")
"""))

cells.append(safe_code_cell("G.4", """\
# G.4 -- ESTAR(1) simetrico  phi1=0.90, phi2=0.10, gamma=1, c=0
cl  = ARIMAModel((1, 0, 0))
dgp = ESTARDGp(phi1=0.90, phi2=0.10, gamma=1.0, c=0.0, delay=1, sigma=1.0, seed=SEED)
verify_dgp("G.4 -- ESTAR(1) simetrico", dgp, {}, cl, CHECKS_NONLINEAR)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.4")
log("\\n" + "="*60 + "\\nG.4 -- ESTAR(1) simetrico\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.4 -- ESTAR(1) phi1=0.90 / phi2=0.10 (gamma=1)")
"""))

# ── RESUMEN ───────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Resumen\n\n"
    "| Bloque | DGPs | T | R | Total series |\n"
    "|--------|------|---|---|-------------|\n"
    "| A — ARMA sin tendencia | 24 | 4 | 500 | 48,000 |\n"
    "| B — ARMA con tendencia | 48 | 4 | 500 | 96,000 |\n"
    "| C — Random Walk | 3 | 4 | 500 | 6,000 |\n"
    "| D — ARCH/GARCH | 4 | 4 | 500 | 8,000 |\n"
    "| E — ETS/Theta | 6* | 4 | 500 | ~12,000 |\n"
    "| F — SARIMA | 6* | <=4 | 500 | ~10,000 |\n"
    "| G — SETAR/LSTAR/ESTAR | 4 | 4 | 500 | 8,000 |\n"
    "| **Total** | **95** | — | — | **~188,000** |\n\n"
    "*Algunos experimentos usan T_list reducido (s=12 requiere T>=50)"
))

# ─── Escribir notebook ────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

out_path = Path("notebooks/experimentos_univariados_v4_cloud.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

total_cells = len(cells)
code_cells  = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook generado: {out_path}")
print(f"Total celdas: {total_cells}  ({code_cells} codigo, {total_cells - code_cells} markdown)")
