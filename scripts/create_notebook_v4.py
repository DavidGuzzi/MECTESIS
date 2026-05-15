"""
Genera notebooks/experimentos_univariados_v4.ipynb
Ejecutar: python scripts/create_notebook_v4.py

Experimentos (32 total):
  A (4)  — AR sin tendencia
  B (8)  — AR con tendencia: cruce persistencia x tendencia
  C (4)  — MA sin tendencia
  D (8)  — MA con tendencia: cruce persistencia x tendencia
  E (2)  — ARMA propuesto
  F (1)  — Random Walk
  G (2)  — AR+ARCH y AR+GARCH
  H (1)  — SAR(1)(1)_4 alta persistencia
  I (2)  — ETS(A,N,N) y Theta
"""

from pathlib import Path
import nbformat as nbf

# ─── helpers ─────────────────────────────────────────────────────────────────

def md(text: str):
    return nbf.v4.new_markdown_cell(text)

def code(src: str):
    return nbf.v4.new_code_cell(src.strip())

# ─── SETUP ───────────────────────────────────────────────────────────────────

SETUP = """\
import warnings
warnings.filterwarnings("ignore")

import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from mectesis.dgp import (
    RandomWalk, SeasonalDGP,
    AR1ARCH, AR1GARCH,
    LocalLevelDGP, LocalTrendDGP,
    ARpDGP, MAqDGP, ARMApqDGP, ARMApqWithTrendDGP,
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
RESULTS = Path("results/univariate_v4")
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)

print("Cargando Chronos-2 (puede tardar ~30 s)...")
chronos = ChronosModel(device="cpu")
print("Chronos-2 listo.")
"""

# ─── HELPERS (copiados de v3) ─────────────────────────────────────────────────

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
    combos = ", ".join(
        f"(T={t}, H={H_by_T.get(t, H_MAX)}, R={r})"
        for t in T_list for r in R_list
    )
    print(f"Exp {exp_id}: {len(T_list)*len(R_list)} ejecución(es) → {combos}")
    all_results = {}
    for T in T_list:
        h = H_by_T.get(T, H_MAX)
        for R in R_list:
            cache = _cache_path(exp_id, T, R)
            if cache.exists():
                print(f"  T={T} H={h}, R={R}: cargando {cache.name}")
                all_results[(T, R)] = _load_results(cache)
                continue
            print(f"  T={T} H={h}, R={R}: simulando...", end=" ", flush=True)
            dgp.rng = np.random.default_rng(seed)
            models = make_models_fn(T)
            engine = MonteCarloEngine(dgp, models, seed=seed)
            t0 = time.time()
            results = engine.run_monte_carlo(R, T, h, dgp_params, verbose=False)
            print(f"OK ({time.time()-t0:.0f}s)")
            _save_results(results, cache)
            all_results[(T, R)] = results
    return all_results


BLOCK_DEFS = [("C", 1, 6), ("M", 7, 18), ("L", 19, 24)]
METRICS_V3 = ["bias", "variance", "rmse", "crps"]


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
    ax.plot(t_train, y_train, color="gray", lw=1.5, label="Histórico")
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
            print(f"  [plot] {model.name} falló: {e}")
    ax.set(title=title, xlabel="t", ylabel="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
"""

# ─── VERIFY HELPERS ───────────────────────────────────────────────────────────

VERIFY = """\
# ─── Funciones de verificación DGP ───────────────────────────────────────────

def verify_dgp(label, dgp, dgp_params, classical_model, checks):
    \"\"\"Valida DGP y estimador. Imprime PASS/FAIL con motivo por cada check.\"\"\"
    print(f"\\n{'─'*60}")
    print(f"VERIFICACIÓN DGP: {label}")
    print(f"{'─'*60}")
    dgp_copy = copy.deepcopy(dgp)
    dgp_copy.rng = np.random.default_rng(7777)
    try:
        y_long = dgp_copy.simulate(T=1000, **dgp_params)
    except Exception as e:
        print(f"  [✗ FAIL] simulate() lanzó excepción: {e}")
        return
    n_fail = 0
    for check_name, check_fn in checks:
        try:
            ok, msg = check_fn(y_long, dgp, dgp_params, classical_model)
        except Exception as e:
            ok, msg = False, f"excepción inesperada → {e}"
        tag = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{tag}] {check_name}: {msg}")
        if not ok:
            n_fail += 1
    if n_fail == 0:
        print(f"  → TODAS LAS VERIFICACIONES PASARON")
    else:
        print(f"  → {n_fail} FALLO(S) — ver detalle arriba")


# ── checks reutilizables ──────────────────────────────────────────────────────

def chk_stationary(y, *_):
    \"\"\"ADF rechaza raíz unitaria (p < 0.05).\"\"\"
    pval = adfuller(y, autolag="AIC")[1]
    return pval < 0.05, f"ADF p={pval:.4f} (umbral 0.05)"

def chk_nonstationary(y, *_):
    \"\"\"ADF NO rechaza raíz unitaria (p > 0.10) — proceso integrado.\"\"\"
    pval = adfuller(y, autolag="AIC")[1]
    return pval > 0.10, f"ADF p={pval:.4f} (se espera >0.10)"

def chk_zero_mean(y, *_):
    \"\"\"Media empírica cercana a 0 (dentro de 3σ/√T).\"\"\"
    mu = y.mean(); sigma = y.std(); T = len(y)
    tol = 3.0 * sigma / np.sqrt(T)
    return abs(mu) < tol, f"media={mu:.4f}, tol=±{tol:.4f}"

def chk_acf_lag1(y, dgp, dgp_params, *_):
    \"\"\"ACF empírica en lag 1 ≈ φ₁ del DGP (tol ±0.15).\"\"\"
    props = dgp.get_theoretical_properties(**dgp_params) if dgp_params else dgp.get_theoretical_properties()
    phi1 = props.get("phis", [None])[0] if props.get("phis") else None
    if phi1 is None:
        return True, "no aplica (sin phis)"
    acf_vals = acf(y, nlags=1, fft=True)
    emp = acf_vals[1]
    ok = abs(emp - phi1) < 0.15
    return ok, f"ACF[1]={emp:.4f}, φ₁_DGP={phi1:.4f}, dif={abs(emp-phi1):.4f}"

def chk_ma_cutoff(y, dgp, dgp_params, *_):
    \"\"\"ACF se anula después del lag q (|ACF[q+1]| < 2/√T).\"\"\"
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
    return ok, f"|ACF[{q+1}]|={val_after:.4f}, umbral=2/√T={thr:.4f}"

def chk_trend_slope(y, dgp, dgp_params, *_):
    \"\"\"Pendiente OLS ≈ δ del DGP (tol ±0.015).\"\"\"
    delta = dgp_params.get("delta", None)
    if delta is None:
        return True, "no aplica (sin delta)"
    T = len(y)
    slope = np.polyfit(np.arange(T), y, 1)[0]
    ok = abs(slope - delta) < 0.015
    return ok, f"pendiente_OLS={slope:.5f}, δ_DGP={delta:.5f}, dif={abs(slope-delta):.5f}"

def chk_arch_effects(y, *_):
    \"\"\"Ljung-Box sobre y² rechaza independencia (p < 0.05) — efectos ARCH.\"\"\"
    y2 = y ** 2
    lb = acorr_ljungbox(y2, lags=[10], return_df=True)
    pval = float(lb["lb_pvalue"].iloc[0])
    return pval < 0.05, f"LB(10) sobre y²: p={pval:.4f} (se espera <0.05)"

def chk_rw_variance_growth(y, *_):
    \"\"\"Varianza crece con t: var(y[T//2:]) > 2 * var(y[:T//4]).\"\"\"
    T = len(y)
    v1 = np.var(y[:T//4])
    v2 = np.var(y[T//2:])
    ok = v2 > 2.0 * v1
    return ok, f"var(primera cuarta)={v1:.4f}, var(segunda mitad)={v2:.4f}"

def chk_seasonal_acf(y, dgp, dgp_params, *_):
    \"\"\"ACF en lag s es significativa (>2/√T).\"\"\"
    s = dgp_params.get("s", 4)
    T = len(y)
    thr = 2.0 / np.sqrt(T)
    acf_vals = acf(y, nlags=s+1, fft=True)
    val_s = abs(acf_vals[s])
    ok = val_s > thr
    return ok, f"|ACF[{s}]|={val_s:.4f}, umbral=2/√T={thr:.4f}"

def chk_fit_classical(y, dgp, dgp_params, model):
    \"\"\"El modelo clásico puede hacer fit+forecast sin error.\"\"\"
    try:
        m = copy.deepcopy(model)
        m.fit(y[:800])
        fc = m.forecast(horizon=6)
        ok = len(fc) == 6 and not np.any(np.isnan(fc))
        return ok, f"fit+forecast OK, primeros valores: {fc[:3].round(4)}"
    except Exception as e:
        return False, str(e)

def chk_arma_aic(y, dgp, dgp_params, model):
    \"\"\"AIC de ARMA(p,q) < AIC de AR(p) puro o MA(q) puro.\"\"\"
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
    \"\"\"ThetaModel se ajusta sin error.\"\"\"
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
    ("Media ≈ 0",             chk_zero_mean),
    ("ACF[1] ≈ φ₁",          chk_acf_lag1),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_AR_TREND = [
    ("Pendiente OLS ≈ δ",     chk_trend_slope),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_MA = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("Media ≈ 0",             chk_zero_mean),
    ("Corte ACF en lag q",    chk_ma_cutoff),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_MA_TREND = [
    ("Pendiente OLS ≈ δ",     chk_trend_slope),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_ARMA = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("AIC ARMA < baseline",   chk_arma_aic),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_RW = [
    ("No estacionariedad",    chk_nonstationary),
    ("Varianza crece con t",  chk_rw_variance_growth),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_ARCH = [
    ("Estacionariedad media", chk_stationary),
    ("Efectos ARCH (LB y²)",  chk_arch_effects),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_SAR = [
    ("Estacionariedad (ADF)", chk_stationary),
    ("ACF estacional signif.",chk_seasonal_acf),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_ETS = [
    ("No estacionariedad",    chk_nonstationary),
    ("Ajuste modelo clásico", chk_fit_classical),
]
CHECKS_THETA = [
    ("Ajuste Theta",          chk_theta_fit),
]
"""

# ─── Generador de celdas de experimento ──────────────────────────────────────

def exp_cell(blk_id, desc, dgp_expr, dgp_params_repr, cl_expr, cl_name,
             checks_name, t_list="T_LIST", extra_make=""):
    """Genera una celda completa: verify + run_exp + grid + plot."""
    make = f"lambda T, _cl=cl: [_cl, chronos]"
    return f"""\
# {blk_id} — {desc}
cl  = {cl_expr}
dgp = {dgp_expr}
dgp_params = {dgp_params_repr}
verify_dgp("{blk_id} — {desc}", dgp, dgp_params, cl, {checks_name})
res = run_exp(dgp, {make}, dgp_params, exp_id="{blk_id}"{', T_list=' + t_list if t_list != 'T_LIST' else ''})
print(f"\\n{{'='*60}}\\n{blk_id} — {desc}\\n{{'='*60}}")
build_grid_table(res, classical_name="{cl_name}")
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="{blk_id} — {desc}")
"""

# ─── Construir celdas ─────────────────────────────────────────────────────────

cells = []

# Título
cells.append(md(
    "# Experimentos Univariados v4\n\n"
    "**Tesis MEC** — Selección reducida y verificada: 32 DGPs × T∈{25,50,100,200} × R=500  \n"
    "**Horizonte por T:** T=25→H=6 · T=50→H=18 · T=100,200→H=24  \n"
    "**Métricas:** Bias, Varianza, RMSE, CRPS  \n"
    "**Bloques:** Corto h=1–6 · Medio h=7–18 · Largo h=19–24  \n"
    "**Verificación DGP:** cada experimento incluye sección PASS/FAIL antes del Monte Carlo  \n"
    "**Resultados:** `results/univariate_v4/`"
))
cells.append(code(SETUP))
cells.append(code(HELPERS))
cells.append(code(VERIFY))

# ── BLOQUE A — AR sin tendencia ───────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque A — AR sin tendencia (4 experimentos)\n\n"
    "DGPs: AR(1) y AR(4) con baja y alta persistencia  \n"
    "Modelo clásico: ARIMA(p,0,0) correctamente especificado vs Chronos-2"
))

AR_SPECS_PLAIN = [
    ("A.1", "AR(1) baja persist. φ=0.30",
     "ARpDGP(phis=[0.30], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((1,0,0))", "ARIMA(1, 0, 0)"),
    ("A.2", "AR(1) alta persist. φ=0.90",
     "ARpDGP(phis=[0.90], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((1,0,0))", "ARIMA(1, 0, 0)"),
    ("A.3", "AR(4) baja persist.",
     "ARpDGP(phis=[0.30, 0.10, 0.05, 0.02], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((4,0,0))", "ARIMA(4, 0, 0)"),
    ("A.4", "AR(4) alta persist.",
     "ARpDGP(phis=[0.90, -0.20, 0.10, -0.05], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((4,0,0))", "ARIMA(4, 0, 0)"),
]

for blk_id, desc, dgp_expr, dgp_params_repr, cl_expr, cl_name in AR_SPECS_PLAIN:
    cells.append(code(exp_cell(blk_id, desc, dgp_expr, dgp_params_repr,
                               cl_expr, cl_name, "CHECKS_AR")))

# ── BLOQUE B — AR con tendencia ───────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque B — AR con tendencia determinística (8 experimentos)\n\n"
    "Cruce: persistencia {baja=0.30, alta=0.90} × tendencia {baja δ=0.02, alta δ=0.10}  \n"
    "DGP: `Y_t = δ·t + AR(p)_t`  \n"
    "Modelo clásico: ARIMAWithTrend (trend='ct')"
))

AR_TREND_SPECS = [
    ("B.1", "AR(1) baja persist. + baja tend. φ=0.30 δ=0.02",
     [0.30], [], 0.02, (1, 0, 0)),
    ("B.2", "AR(1) baja persist. + alta tend. φ=0.30 δ=0.10",
     [0.30], [], 0.10, (1, 0, 0)),
    ("B.3", "AR(1) alta persist. + baja tend. φ=0.90 δ=0.02",
     [0.90], [], 0.02, (1, 0, 0)),
    ("B.4", "AR(1) alta persist. + alta tend. φ=0.90 δ=0.10",
     [0.90], [], 0.10, (1, 0, 0)),
    ("B.5", "AR(4) baja persist. + baja tend. δ=0.02",
     [0.30, 0.10, 0.05, 0.02], [], 0.02, (4, 0, 0)),
    ("B.6", "AR(4) baja persist. + alta tend. δ=0.10",
     [0.30, 0.10, 0.05, 0.02], [], 0.10, (4, 0, 0)),
    ("B.7", "AR(4) alta persist. + baja tend. δ=0.02",
     [0.90, -0.20, 0.10, -0.05], [], 0.02, (4, 0, 0)),
    ("B.8", "AR(4) alta persist. + alta tend. δ=0.10",
     [0.90, -0.20, 0.10, -0.05], [], 0.10, (4, 0, 0)),
]

for blk_id, desc, phis, thetas, delta, order in AR_TREND_SPECS:
    p, d, q = order
    dgp_expr = f"ARMApqWithTrendDGP(phis={phis!r}, thetas={thetas!r}, delta={delta}, sigma=1.0, seed=SEED)"
    dgp_params = "{}"
    cl_expr = f"ARIMAWithTrendModel({order!r}, trend='ct')"
    cl_name = f"ARIMA({p}, {d}, {q})+trend"
    cells.append(code(exp_cell(blk_id, desc, dgp_expr, dgp_params,
                               cl_expr, cl_name, "CHECKS_AR_TREND")))

# ── BLOQUE C — MA sin tendencia ───────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque C — MA sin tendencia (4 experimentos)\n\n"
    "DGPs: MA(1) y MA(4) con baja y alta persistencia  \n"
    "Modelo clásico: ARIMA(0,0,q)"
))

MA_SPECS_PLAIN = [
    ("C.1", "MA(1) baja persist. θ=0.30",
     "MAqDGP(thetas=[0.30], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((0,0,1))", "ARIMA(0, 0, 1)"),
    ("C.2", "MA(1) alta persist. θ=0.90",
     "MAqDGP(thetas=[0.90], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((0,0,1))", "ARIMA(0, 0, 1)"),
    ("C.3", "MA(4) baja persist.",
     "MAqDGP(thetas=[0.30, 0.10, -0.05, 0.02], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((0,0,4))", "ARIMA(0, 0, 4)"),
    ("C.4", "MA(4) alta persist.",
     "MAqDGP(thetas=[0.90, 0.10, -0.05, 0.02], sigma=1.0, seed=SEED)", "{}", "ARIMAModel((0,0,4))", "ARIMA(0, 0, 4)"),
]

for blk_id, desc, dgp_expr, dgp_params_repr, cl_expr, cl_name in MA_SPECS_PLAIN:
    cells.append(code(exp_cell(blk_id, desc, dgp_expr, dgp_params_repr,
                               cl_expr, cl_name, "CHECKS_MA")))

# ── BLOQUE D — MA con tendencia ───────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque D — MA con tendencia determinística (8 experimentos)\n\n"
    "Cruce: persistencia {baja=0.30, alta=0.90} × tendencia {baja δ=0.02, alta δ=0.10}  \n"
    "Modelo clásico: ARIMAWithTrend (trend='ct')"
))

MA_TREND_SPECS = [
    ("D.1", "MA(1) baja persist. + baja tend. θ=0.30 δ=0.02",
     [], [0.30], 0.02, (0, 0, 1)),
    ("D.2", "MA(1) baja persist. + alta tend. θ=0.30 δ=0.10",
     [], [0.30], 0.10, (0, 0, 1)),
    ("D.3", "MA(1) alta persist. + baja tend. θ=0.90 δ=0.02",
     [], [0.90], 0.02, (0, 0, 1)),
    ("D.4", "MA(1) alta persist. + alta tend. θ=0.90 δ=0.10",
     [], [0.90], 0.10, (0, 0, 1)),
    ("D.5", "MA(4) baja persist. + baja tend. δ=0.02",
     [], [0.30, 0.10, -0.05, 0.02], 0.02, (0, 0, 4)),
    ("D.6", "MA(4) baja persist. + alta tend. δ=0.10",
     [], [0.30, 0.10, -0.05, 0.02], 0.10, (0, 0, 4)),
    ("D.7", "MA(4) alta persist. + baja tend. δ=0.02",
     [], [0.90, 0.10, -0.05, 0.02], 0.02, (0, 0, 4)),
    ("D.8", "MA(4) alta persist. + alta tend. δ=0.10",
     [], [0.90, 0.10, -0.05, 0.02], 0.10, (0, 0, 4)),
]

for blk_id, desc, phis, thetas, delta, order in MA_TREND_SPECS:
    p, d, q = order
    dgp_expr = f"ARMApqWithTrendDGP(phis={phis!r}, thetas={thetas!r}, delta={delta}, sigma=1.0, seed=SEED)"
    dgp_params = "{}"
    cl_expr = f"ARIMAWithTrendModel({order!r}, trend='ct')"
    cl_name = f"ARIMA({p}, {d}, {q})+trend"
    cells.append(code(exp_cell(blk_id, desc, dgp_expr, dgp_params,
                               cl_expr, cl_name, "CHECKS_MA_TREND")))

# ── BLOQUE E — ARMA propuesto ─────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque E — ARMA propuesto (2 experimentos)\n\n"
    "**E.1 — ARMA(1,1) alta persistencia** φ=0.80, θ=0.40  \n"
    "Representación parsimoniosa: combinación de memoria AR y corrección MA; "
    "típica en índices económicos mensuales.  \n\n"
    "**E.2 — ARMA(2,1) moderado** φ=[0.50, 0.20], θ=0.30  \n"
    "Captura dinámica de ciclo de negocios: AR(2) genera ciclos, MA(1) corrige correlación residual."
))

cells.append(code(exp_cell(
    "E.1", "ARMA(1,1) alta persist. φ=0.80 θ=0.40",
    "ARMApqDGP(phis=[0.80], thetas=[0.40], sigma=1.0, seed=SEED)", "{}",
    "ARIMAModel((1,0,1))", "ARIMA(1, 0, 1)", "CHECKS_ARMA"
)))

cells.append(code(exp_cell(
    "E.2", "ARMA(2,1) moderado φ=[0.50,0.20] θ=0.30",
    "ARMApqDGP(phis=[0.50, 0.20], thetas=[0.30], sigma=1.0, seed=SEED)", "{}",
    "ARIMAModel((2,0,1))", "ARIMA(2, 0, 1)", "CHECKS_ARMA"
)))

# ── BLOQUE F — Random Walk ────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque F — Random Walk (1 experimento)\n\n"
    "DGP: `Y_t = Y_{t-1} + ε_t`  sin drift  \n"
    "Modelo clásico: ARIMA(0,1,0)"
))

cells.append(code("""\
# F.1 — Random Walk sin drift
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
dgp_params = {"drift": 0.0, "sigma": 1.0}
verify_dgp("F.1 — Random Walk sin drift", dgp, dgp_params, cl, CHECKS_RW)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], dgp_params, exp_id="F.1")
print("\\n" + "="*60 + "\\nF.1 — Random Walk sin drift\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="F.1 — Random Walk sin drift")
"""))

# ── BLOQUE G — ARCH / GARCH ───────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque G — Volatilidad condicional (2 experimentos)\n\n"
    "G.1: AR(1)-ARCH(1) α=0.30 — efecto ARCH moderado  \n"
    "G.2: AR(1)-GARCH(1,1) α+β=0.95 — alta persistencia de varianza  \n"
    "Modelo clásico: AR+ARCH o AR+GARCH correctamente especificado"
))

cells.append(code("""\
# G.1 — AR(1)-ARCH(1) α=0.30
cl  = ARARCHModel(ar_lags=1, p=1)
dgp = AR1ARCH(seed=SEED)
dgp_params = {"phi": 0.5, "omega": 0.5, "alpha": 0.30}
verify_dgp("G.1 — AR(1)-ARCH(1) α=0.30", dgp, dgp_params, cl, CHECKS_ARCH)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], dgp_params, exp_id="G.1")
print("\\n" + "="*60 + "\\nG.1 — AR(1)-ARCH(1) α=0.30\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="G.1 — AR(1)-ARCH(1) α=0.30")
"""))

cells.append(code("""\
# G.2 — AR(1)-GARCH(1,1) alta persistencia α+β=0.95
cl  = ARGARCHModel(ar_lags=1, p=1, q=1)
dgp = AR1GARCH(seed=SEED)
dgp_params = {"phi": 0.5, "omega": 0.1, "alpha": 0.10, "beta": 0.85}
verify_dgp("G.2 — AR(1)-GARCH(1,1) α+β=0.95", dgp, dgp_params, cl, CHECKS_ARCH)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], dgp_params, exp_id="G.2")
print("\\n" + "="*60 + "\\nG.2 — AR(1)-GARCH(1,1) alta persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="G.2 — AR(1)-GARCH(1,1) α+β=0.95")
"""))

# ── BLOQUE H — SAR ────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque H — SAR(1)(1)_4 alta persistencia (1 experimento)\n\n"
    "DGP: `(1 - 0.90·L)(1 - 0.60·L⁴)Y_t = ε_t`  \n"
    "Modelo clásico: SARIMA(1,0,0)(1,0,0)_4"
))

cells.append(code("""\
# H.1 — SAR(1)(1)_4 alta persistencia φ=0.90, Φ=0.60
cl  = SARIMAModel(order=(1,0,0), seasonal_order=(1,0,0,4))
dgp = SeasonalDGP(seed=SEED)
dgp_params = {"phi": 0.90, "Phi": 0.60, "s": 4, "sigma": 1.0, "integrated": False}
verify_dgp("H.1 — SAR(1)(1)_4 alta persist.", dgp, dgp_params, cl, CHECKS_SAR)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], dgp_params, exp_id="H.1")
print("\\n" + "="*60 + "\\nH.1 — SAR(1)(1)_4 alta persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="H.1 — SAR(1)(1)_4 φ=0.90 Φ=0.60")
"""))

# ── BLOQUE I — ETS y Theta ────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque I — ETS y Theta (2 experimentos)\n\n"
    "I.1: Local Level ETS(A,N,N) — nivel estocástico sin tendencia  \n"
    "I.2: Local Trend → Theta — tendencia suave con componente estocástica"
))

cells.append(code("""\
# I.1 — Local Level ETS(A,N,N)
cl  = ETSModel()
dgp = LocalLevelDGP(seed=SEED)
dgp_params = {"sigma_eps": 1.0, "sigma_eta": 0.10}
verify_dgp("I.1 — Local Level ETS(A,N,N)", dgp, dgp_params, cl, CHECKS_ETS)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], dgp_params, exp_id="I.1")
print("\\n" + "="*60 + "\\nI.1 — Local Level ETS(A,N,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="I.1 — Local Level ETS(A,N,N)")
"""))

cells.append(code("""\
# I.2 — Theta (DGP: Local Trend σ_ζ=0.05)
cl  = ThetaModel()
dgp = LocalTrendDGP(seed=SEED)
dgp_params = {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.05, "b0": 0.20}
verify_dgp("I.2 — Theta (Local Trend b0=0.20)", dgp, dgp_params, cl, CHECKS_THETA)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], dgp_params, exp_id="I.2")
print("\\n" + "="*60 + "\\nI.2 — Theta (Local Trend)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], dgp_params, title="I.2 — Theta (Local Trend b0=0.20)")
"""))

# ── RESUMEN ───────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Resumen\n\n"
    "| Bloque | Descripción | DGPs |\n"
    "|--------|-------------|------|\n"
    "| A | AR sin tendencia | 4 |\n"
    "| B | AR con tendencia (cruce persist.×tend.) | 8 |\n"
    "| C | MA sin tendencia | 4 |\n"
    "| D | MA con tendencia (cruce persist.×tend.) | 8 |\n"
    "| E | ARMA propuesto | 2 |\n"
    "| F | Random Walk | 1 |\n"
    "| G | AR+ARCH / AR+GARCH | 2 |\n"
    "| H | SAR(1)(1)_4 alta persist. | 1 |\n"
    "| I | ETS(A,N,N) y Theta | 2 |\n"
    "| **Total** | | **32** |\n"
))

# ─── Escribir notebook ────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (.venv_mectesis)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

out_path = Path("notebooks/experimentos_univariados_v4.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

total_cells = len(cells)
code_cells  = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook generado: {out_path}")
print(f"Total celdas: {total_cells}  ({code_cells} código, {total_cells - code_cells} markdown)")
