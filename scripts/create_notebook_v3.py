"""
Genera notebooks/experimentos_univariados_v3.ipynb
Ejecutar: python scripts/create_notebook_v3.py
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

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

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
H_BY_T  = {25: 6, 50: 18, 100: 24, 200: 24}  # horizonte por T total
H_MAX   = 24                                   # para definir los bloques
R_LIST  = [500]
T_LIST  = [25, 50, 100, 200]                   # longitud TOTAL de la serie
# T_train = T - H_BY_T[T]  (el engine hace y_train = y[:T-H])
RESULTS = Path("results/univariate_v3")
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)

print("Cargando Chronos-2 (puede tardar ~30 s)...")
chronos = ChronosModel(device="cpu")
print("Chronos-2 listo.")
"""

HELPERS = """\
# ─── Funciones auxiliares (reutilizadas de v2) ───────────────────────────────

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
    \"\"\"
    T_list: longitudes TOTALES de la serie.
    El engine hace y_train = y[:T-H] internamente.
    H_by_T: horizonte por T (por defecto H_BY_T global).
      T=25 → H=6  (Corto, T_train=19)
      T=50 → H=18 (Corto+Medio, T_train=32)
      T=100,200 → H=24 (todos los bloques)
    \"\"\"
    if H_by_T is None:
        H_by_T = H_BY_T
    n_runs = len(T_list) * len(R_list)
    combos = ", ".join(
        f"(T={t}, H={H_by_T.get(t, H_MAX)}, R={r})"
        for t in T_list for r in R_list
    )
    print(f"Exp {exp_id}: {n_runs} ejecución(es) → {combos}")
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


# ─── Funciones nuevas v3 ─────────────────────────────────────────────────────

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
    \"\"\"
    Tabla ancha: una fila por (T, Modelo).
    ratio_rmse = RMSE_clásico / RMSE_Chronos  (>1 = Chronos gana).
    \"\"\"
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
    \"\"\"1 realización del DGP: histórico + split + observado + forecasts superpuestos.\"\"\"
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
            print(f"  [plot_simulation_v3] {model.name} falló: {e}")

    ax.set(title=title, xlabel="t", ylabel="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
"""


# ─── Specs de experimentos ────────────────────────────────────────────────────

# (suffix, descripción, phis, thetas, orden ARIMA)
ARMA_SPECS = [
    ("1",  "AR(1) ρ=0.30",    [0.30],                       [],                              (1, 0, 0)),
    ("2",  "AR(1) ρ=0.90",    [0.90],                       [],                              (1, 0, 0)),
    ("3",  "AR(2) ρ=0.30",    [0.30,  0.10],                [],                              (2, 0, 0)),
    ("4",  "AR(2) ρ=0.90",    [0.90, -0.20],                [],                              (2, 0, 0)),
    ("5",  "AR(3) ρ=0.30",    [0.30,  0.10,  0.05],         [],                              (3, 0, 0)),
    ("6",  "AR(3) ρ=0.90",    [0.90, -0.20,  0.10],         [],                              (3, 0, 0)),
    ("7",  "AR(4) ρ=0.30",    [0.30,  0.10,  0.05,  0.02],  [],                              (4, 0, 0)),
    ("8",  "AR(4) ρ=0.90",    [0.90, -0.20,  0.10, -0.05],  [],                              (4, 0, 0)),
    ("9",  "MA(1) θ=0.30",    [],      [0.30],                                               (0, 0, 1)),
    ("10", "MA(1) θ=0.90",    [],      [0.90],                                               (0, 0, 1)),
    ("11", "MA(2) θ=0.30",    [],      [0.30,  0.10],                                        (0, 0, 2)),
    ("12", "MA(2) θ=0.90",    [],      [0.90,  0.10],                                        (0, 0, 2)),
    ("13", "MA(3) θ=0.30",    [],      [0.30,  0.10, -0.05],                                 (0, 0, 3)),
    ("14", "MA(3) θ=0.90",    [],      [0.90,  0.10, -0.05],                                 (0, 0, 3)),
    ("15", "MA(4) θ=0.30",    [],      [0.30,  0.10, -0.05,  0.02],                          (0, 0, 4)),
    ("16", "MA(4) θ=0.90",    [],      [0.90,  0.10, -0.05,  0.02],                          (0, 0, 4)),
    ("17", "ARMA(1,1) ρ=0.30", [0.30],           [0.10],                                    (1, 0, 1)),
    ("18", "ARMA(1,1) ρ=0.90", [0.90],           [0.30],                                    (1, 0, 1)),
    ("19", "ARMA(2,2) ρ=0.30", [0.30,  0.10],    [0.10,  0.05],                             (2, 0, 2)),
    ("20", "ARMA(2,2) ρ=0.90", [0.90, -0.20],    [0.30, -0.10],                             (2, 0, 2)),
    ("21", "ARMA(1,4) ρ=0.30", [0.30],           [0.10,  0.05, -0.03,  0.01],               (1, 0, 4)),
    ("22", "ARMA(1,4) ρ=0.90", [0.90],           [0.30, -0.10,  0.05, -0.02],               (1, 0, 4)),
    ("23", "ARMA(4,1) ρ=0.30", [0.30,  0.10,  0.05,  0.02],  [0.10],                        (4, 0, 1)),
    ("24", "ARMA(4,1) ρ=0.90", [0.90, -0.20,  0.10, -0.05],  [0.30],                        (4, 0, 1)),
]


def _dgp_class(phis, thetas):
    """Devuelve el constructor apropiado según si hay phis, thetas o ambos."""
    if not thetas:
        return "ARpDGP", f"ARpDGP(phis={phis!r}, sigma=1.0, seed=SEED)"
    if not phis:
        return "MAqDGP", f"MAqDGP(thetas={thetas!r}, sigma=1.0, seed=SEED)"
    return "ARMApqDGP", f"ARMApqDGP(phis={phis!r}, thetas={thetas!r}, sigma=1.0, seed=SEED)"


def _arima_name(order):
    p, d, q = order
    return f"ARIMA({p}, {d}, {q})"


def exp_cell_A(suf, desc, phis, thetas, order):
    cl_name = _arima_name(order)
    _, dgp_expr = _dgp_class(phis, thetas)
    return f"""\
# A.{suf} — {desc}
cl  = ARIMAModel({order!r})
dgp = {dgp_expr}
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {{}}, exp_id="A.{suf}")
print(f"\\n{'='*60}\\nA.{suf} — {desc}\\n{'='*60}")
build_grid_table(res, classical_name="{cl_name}")
plot_simulation_v3(dgp, [cl, chronos], {{}}, title="A.{suf} — {desc}")
"""


def exp_cell_B(suf, desc, phis, thetas, order, delta, delta_name):
    cl_name = _arima_name(order) + "+trend"
    return f"""\
# B.{suf} — {desc}  [tendencia {delta_name} δ={delta}]
cl  = ARIMAWithTrendModel({order!r}, trend="ct")
dgp = ARMApqWithTrendDGP(phis={phis!r}, thetas={thetas!r}, delta={delta}, sigma=1.0, seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {{}}, exp_id="B.{suf}")
print(f"\\n{'='*60}\\nB.{suf} — {desc} (δ={delta})\\n{'='*60}")
build_grid_table(res, classical_name="{cl_name}")
plot_simulation_v3(dgp, [cl, chronos], {{}}, title="B.{suf} — {desc} (δ={delta})")
"""


# ─── Construir celdas ─────────────────────────────────────────────────────────

cells = []

# Título
cells.append(md(
    "# Experimentos Univariados v3\n\n"
    "**Tesis MEC** — Grilla completa: 97 DGPs × T∈{25,50,100,200} × R=500  \n"
    "**Horizonte por T:** T=25→H=6 (Corto) · T=50→H=18 (Corto+Medio) · T=100,200→H=24 (todos)  \n"
    "**Métricas:** Bias, Varianza, RMSE, CRPS  \n"
    "**Bloques:** Corto h=1–6 · Medio h=7–18 · Largo h=19–24  \n"
    "**Ratio:** RMSE_clásico / RMSE_Chronos (>1 = Chronos gana)  \n"
    "**Resultados:** `results/univariate_v3/` — si existen se cargan sin re-simular"
))

# Setup
cells.append(code(SETUP))
cells.append(code(HELPERS))

# ── BLOQUE A ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque A — ARMA sin tendencia (24 experimentos)\n\n"
    "DGPs: AR(1–4) · MA(1–4) · ARMA(1,1) · ARMA(2,2) · ARMA(1,4) · ARMA(4,1)  \n"
    "Modelo clásico: ARIMA(p,0,q) correctamente especificado vs Chronos-2"
))

for suf, desc, phis, thetas, order in ARMA_SPECS:
    cells.append(code(exp_cell_A(suf, desc, phis, thetas, order)))

# ── BLOQUE B ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque B — ARMA con tendencia determinística (48 experimentos)\n\n"
    "DGP: `Y_t = α + δ·t + ARMA_t`  \n"
    "Tendencia leve: δ=0.02 (B.1–B.24) | Tendencia fuerte: δ=0.10 (B.25–B.48)  \n"
    "Modelo clásico: ARIMA(p,0,q)+trend (trend='ct')"
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
    "DGP: `Y_t = drift + Y_{t-1} + ε_t`  \n"
    "Modelo clásico: ARIMA(0,1,0)"
))

cells.append(code("""\
# C.1 — RW sin drift
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {"drift": 0.0, "sigma": 1.0}, exp_id="C.1")
print("\\n" + "="*60 + "\\nC.1 — RW sin drift\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"drift": 0.0, "sigma": 1.0}, title="C.1 — Random Walk sin drift")
"""))

cells.append(code("""\
# C.2 — RW drift leve (δ=0.05)
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {"drift": 0.05, "sigma": 1.0}, exp_id="C.2")
print("\\n" + "="*60 + "\\nC.2 — RW drift leve (δ=0.05)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"drift": 0.05, "sigma": 1.0}, title="C.2 — Random Walk drift leve (δ=0.05)")
"""))

cells.append(code("""\
# C.3 — RW drift fuerte (δ=0.20)
cl  = ARIMAModel((0, 1, 0))
dgp = RandomWalk(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {"drift": 0.20, "sigma": 1.0}, exp_id="C.3")
print("\\n" + "="*60 + "\\nC.3 — RW drift fuerte (δ=0.20)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"drift": 0.20, "sigma": 1.0}, title="C.3 — Random Walk drift fuerte (δ=0.20)")
"""))

# ── BLOQUE D ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque D — Volatilidad condicional: ARCH/GARCH (4 experimentos)\n\n"
    "DGP: AR(1) en la media + proceso de varianza condicional  \n"
    "Modelo clásico: AR+ARCH o AR+GARCH correctamente especificado"
))

cells.append(code("""\
# D.1 — AR(1)-ARCH(1) leve  (α=0.10)
cl  = ARARCHModel(ar_lags=1, p=1)
dgp = AR1ARCH(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.5, "alpha": 0.10}, exp_id="D.1")
print("\\n" + "="*60 + "\\nD.1 — AR(1)-ARCH(1) leve\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.5, "alpha": 0.10}, title="D.1 — AR(1)-ARCH(1) leve (α=0.10)")
"""))

cells.append(code("""\
# D.2 — AR(1)-ARCH(1) fuerte  (α=0.50)
cl  = ARARCHModel(ar_lags=1, p=1)
dgp = AR1ARCH(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.5, "alpha": 0.50}, exp_id="D.2")
print("\\n" + "="*60 + "\\nD.2 — AR(1)-ARCH(1) fuerte\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.5, "alpha": 0.50}, title="D.2 — AR(1)-ARCH(1) fuerte (α=0.50)")
"""))

cells.append(code("""\
# D.3 — AR(1)-GARCH(1,1) baja persistencia  (α+β=0.50)
cl  = ARGARCHModel(ar_lags=1, p=1, q=1)
dgp = AR1GARCH(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.5, "alpha": 0.10, "beta": 0.40}, exp_id="D.3")
print("\\n" + "="*60 + "\\nD.3 — AR(1)-GARCH(1,1) baja persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.5, "alpha": 0.10, "beta": 0.40}, title="D.3 — AR(1)-GARCH(1,1) baja persistencia (α+β=0.50)")
"""))

cells.append(code("""\
# D.4 — AR(1)-GARCH(1,1) alta persistencia  (α+β=0.95)
cl  = ARGARCHModel(ar_lags=1, p=1, q=1)
dgp = AR1GARCH(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.5, "omega": 0.1, "alpha": 0.10, "beta": 0.85}, exp_id="D.4")
print("\\n" + "="*60 + "\\nD.4 — AR(1)-GARCH(1,1) alta persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.5, "omega": 0.1, "alpha": 0.10, "beta": 0.85}, title="D.4 — AR(1)-GARCH(1,1) alta persistencia (α+β=0.95)")
"""))

# ── BLOQUE E ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque E — ETS y Theta (8 experimentos)\n\n"
    "DGPs de espacio de estados: nivel local, tendencia, estacionalidad  \n"
    "Modelos clásicos: ETS(A,T,S) y Theta"
))

cells.append(code("""\
# E.1 — Local Level  ETS(A,N,N)
cl  = ETSModel()
dgp = LocalLevelDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.10}, exp_id="E.1")
print("\\n" + "="*60 + "\\nE.1 — Local Level ETS(A,N,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.10}, title="E.1 — Local Level ETS(A,N,N)")
"""))

cells.append(code("""\
# E.2 — Local Linear Trend leve  ETS(A,A,N)  σ_ζ=0.05
cl  = ETSModel(trend="add")
dgp = LocalTrendDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.05, "b0": 0.1}, exp_id="E.2")
print("\\n" + "="*60 + "\\nE.2 — LLT leve ETS(A,A,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.05, "b0": 0.1}, title="E.2 — Local Linear Trend leve (σ_ζ=0.05)")
"""))

cells.append(code("""\
# E.3 — Local Linear Trend fuerte  ETS(A,A,N)  σ_ζ=0.20
cl  = ETSModel(trend="add")
dgp = LocalTrendDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.20, "b0": 0.5}, exp_id="E.3")
print("\\n" + "="*60 + "\\nE.3 — LLT fuerte ETS(A,A,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.20, "b0": 0.5}, title="E.3 — Local Linear Trend fuerte (σ_ζ=0.20)")
"""))

cells.append(code("""\
# E.4 — Damped Trend  ETS(A,Ad,N)  φ_d=0.90
cl  = ETSModel(trend="add", damped_trend=True)
dgp = DampedTrendDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"phi": 0.9, "sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.1}, exp_id="E.4")
print("\\n" + "="*60 + "\\nE.4 — Damped Trend ETS(A,Ad,N)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"phi": 0.9, "sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.1}, title="E.4 — Damped Trend ETS(A,Ad,N)")
"""))

cells.append(code("""\
# E.5 — Seasonal Aditiva s=12  ETS(A,N,A)  (sin tendencia: b0=0, sigma_zeta=0)
cl  = ETSModel(seasonal="add", seasonal_periods=12)
dgp = LocalLevelSeasonalDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1,
               "sigma_zeta": 0.0, "sigma_omega": 0.05, "b0": 0.0},
              exp_id="E.5", T_list=[50, 100, 200])  # T=25 muy corto para s=12
print("\\n" + "="*60 + "\\nE.5 — Seasonal Aditiva s=12 ETS(A,N,A)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1, "sigma_zeta": 0.0, "sigma_omega": 0.05, "b0": 0.0}, title="E.5 — Seasonal Aditiva s=12 ETS(A,N,A)")
"""))

cells.append(code("""\
# E.6 — Trend + Seasonal s=12  ETS(A,A,A)
cl  = ETSModel(trend="add", seasonal="add", seasonal_periods=12)
dgp = LocalLevelSeasonalDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1,
               "sigma_zeta": 0.05, "sigma_omega": 0.05, "b0": 0.1},
              exp_id="E.6", T_list=[50, 100, 200])
print("\\n" + "="*60 + "\\nE.6 — Trend+Seasonal s=12 ETS(A,A,A)\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"s": 12, "sigma_eps": 0.5, "sigma_eta": 0.1, "sigma_zeta": 0.05, "sigma_omega": 0.05, "b0": 0.1}, title="E.6 — Trend+Seasonal s=12 ETS(A,A,A)")
"""))

cells.append(code("""\
# E.7 — Theta leve  (tendencia suave b0=0.10, σ_ζ=0.01)
cl  = ThetaModel()
dgp = LocalTrendDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.01, "b0": 0.10}, exp_id="E.7")
print("\\n" + "="*60 + "\\nE.7 — Theta leve\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.1, "sigma_zeta": 0.01, "b0": 0.10}, title="E.7 — Theta leve (b0=0.10, σ_ζ=0.01)")
"""))

cells.append(code("""\
# E.8 — Theta fuerte  (tendencia marcada b0=0.50, σ_ζ=0.10)
cl  = ThetaModel()
dgp = LocalTrendDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.10, "b0": 0.50}, exp_id="E.8")
print("\\n" + "="*60 + "\\nE.8 — Theta fuerte\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {"sigma_eps": 1.0, "sigma_eta": 0.2, "sigma_zeta": 0.10, "b0": 0.50}, title="E.8 — Theta fuerte (b0=0.50, σ_ζ=0.10)")
"""))

# ── BLOQUE F ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque F — SARIMA (6 experimentos)\n\n"
    "DGP: SAR(1)(1)_s estacionario · (1-L)(1-L^s) integrado  \n"
    "Períodos estacionales: s=4 (trimestral) · s=12 (mensual)"
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
        dgp_params = f'{{"s": {s}, "sigma": 1.0, "integrated": True}}'
    else:
        dgp_params = f'{{"phi": {phi}, "Phi": {Phi}, "s": {s}, "sigma": 1.0, "integrated": False}}'
    t_list = "[50, 100, 200]" if s == 12 else "T_LIST"
    src = f"""\
# F.{suf} — {desc}
cl  = SARIMAModel(order={order!r}, seasonal_order={sorder!r})
dgp = SeasonalDGP(seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos],
              {dgp_params},
              exp_id="F.{suf}", T_list={t_list})
print("\\n" + "="*60 + "\\nF.{suf} — {desc}\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {dgp_params}, title="F.{suf} — {desc}")
"""
    cells.append(code(src))

# ── BLOQUE G ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n## Bloque G — No linealidad de umbral: SETAR / LSTAR / ESTAR (4 experimentos)\n\n"
    "Benchmark clásico: ARIMA(1,0,0) — **lineal misspecificado**  \n"
    "Pregunta: ¿detecta Chronos-2 la no-linealidad que ARIMA no puede capturar?  \n"
    "- **SETAR**: umbral determinístico y observable (vs MS-AR latente ya implementado)  \n"
    "- **LSTAR**: transición suave asimétrica (ciclos de negocios)  \n"
    "- **ESTAR**: transición suave simétrica (tipos de cambio con bandas — canónico para Argentina)"
))

cells.append(code("""\
# G.1 — SETAR(2;1) baja persistencia  φ₁=0.30, φ₂=-0.30, k=0
cl  = ARIMAModel((1, 0, 0))
dgp = SETARDGp(phi1=0.30, phi2=-0.30, threshold=0.0, delay=1, sigma=1.0, seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.1")
print("\\n" + "="*60 + "\\nG.1 — SETAR(2;1) baja persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.1 — SETAR(2;1) φ₁=0.30 / φ₂=-0.30")
"""))

cells.append(code("""\
# G.2 — SETAR(2;1) alta persistencia  φ₁=0.90, φ₂=-0.50, k=0
cl  = ARIMAModel((1, 0, 0))
dgp = SETARDGp(phi1=0.90, phi2=-0.50, threshold=0.0, delay=1, sigma=1.0, seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.2")
print("\\n" + "="*60 + "\\nG.2 — SETAR(2;1) alta persistencia\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.2 — SETAR(2;1) φ₁=0.90 / φ₂=-0.50")
"""))

cells.append(code("""\
# G.3 — LSTAR(1) asimétrico  φ₁=0.30, φ₂=0.90, γ=2, c=0
#   G≈0 (régimen bajo): persistencia leve | G≈1 (régimen alto): persistencia alta
cl  = ARIMAModel((1, 0, 0))
dgp = LSTARDGp(phi1=0.30, phi2=0.90, gamma=2.0, c=0.0, delay=1, sigma=1.0, seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.3")
print("\\n" + "="*60 + "\\nG.3 — LSTAR(1) asimétrico\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.3 — LSTAR(1) φ₁=0.30 / φ₂=0.90 (γ=2)")
"""))

cells.append(code("""\
# G.4 — ESTAR(1) simétrico  φ₁=0.90, φ₂=0.10, γ=1, c=0
#   G≈0 (cerca del equilibrio): alta persistencia | G≈1 (lejos): rápida reversión
#   Modelo canónico para tipos de cambio con bandas
cl  = ARIMAModel((1, 0, 0))
dgp = ESTARDGp(phi1=0.90, phi2=0.10, gamma=1.0, c=0.0, delay=1, sigma=1.0, seed=SEED)
res = run_exp(dgp, lambda T, _cl=cl: [_cl, chronos], {}, exp_id="G.4")
print("\\n" + "="*60 + "\\nG.4 — ESTAR(1) simétrico\\n" + "="*60)
build_grid_table(res, classical_name=cl.name)
plot_simulation_v3(dgp, [cl, chronos], {}, title="G.4 — ESTAR(1) φ₁=0.90 / φ₂=0.10 (γ=1)")
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
    "| F — SARIMA | 6* | ≤4 | 500 | ~10,000 |\n"
    "| G — SETAR/LSTAR/ESTAR | 4 | 4 | 500 | 8,000 |\n"
    "| **Total** | **95** | — | — | **~188,000** |\n\n"
    "*Algunos experimentos usan T_list reducido (s=12 requiere T≥50)"
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

out_path = Path("notebooks/experimentos_univariados_v3.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

total_cells = len(cells)
code_cells  = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook generado: {out_path}")
print(f"Total celdas: {total_cells}  ({code_cells} código, {total_cells - code_cells} markdown)")
