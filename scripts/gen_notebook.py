"""
Generates notebooks/experimentos_univariados.ipynb
Run once: python scripts/gen_notebook.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "experimentos_univariados.ipynb"
RESULTS_DIR = "results/univariate"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ─── Cell 0: Title ────────────────────────────────────────────────────────────
c0 = md(
    "# Experimentos Univariados 1.1–1.8\n\n"
    "**Tesis MEC** — Comparación TSFMs vs Modelos Clásicos bajo DGPs controlados  \n"
    "**Setup:** T ∈ {200, 1000} | H = 24 | R_LIST = [500] | Semilla = 3649  \n"
    "**Métricas:** Bias, Varianza, MSE, RMSE, MAE, Cobertura 80%/95%, Amplitud 80%/95%  \n"
    "**Resultados:** guardados en `results/univariate/` — si existen se cargan sin re-simular\n\n"
    "---"
)

# ─── Cell 1: Imports & constants ──────────────────────────────────────────────
c1 = code(
    "import warnings\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "import time\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "from IPython.display import display\n"
    "\n"
    "from mectesis.dgp import AR1, RandomWalk, AR1WithTrend, SeasonalDGP, AR1WithBreak\n"
    "from mectesis.models import (\n"
    "    ARIMAModel, ChronosModel,\n"
    "    SARIMAModel, ARIMAWithTrendModel, ARIMAWithBreakModel,\n"
    ")\n"
    "from mectesis.simulation import MonteCarloEngine\n"
    "\n"
    "# ── Parámetros globales ───────────────────────────────────────────────────\n"
    "SEED    = 3649\n"
    "H       = 24\n"
    "R_LIST  = [500]          # agregar 1000 para robustez: [500, 1000]\n"
    "T_LIST  = [200, 1000]\n"
    f'RESULTS = Path("{RESULTS_DIR}")\n'
    "RESULTS.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    'plt.rcParams.update({"figure.dpi": 110, "font.size": 10})\n'
    'pd.set_option("display.float_format", "{:.4f}".format)\n'
    "\n"
    'print("Cargando Chronos-2 (puede tardar ~30 s)...")\n'
    'chronos = ChronosModel(device="cpu")\n'
    'print("Chronos-2 listo.")'
)

# ─── Cell 2: Helper functions ─────────────────────────────────────────────────
c2 = code(
    "# ─── Funciones auxiliares ───────────────────────────────────────────────────\n"
    "\n"
    "def _cache_path(exp_id: str, T: int, R: int) -> Path:\n"
    '    return RESULTS / f"exp_{exp_id.replace(\'.\', \'_\')}_T{T}_R{R}.csv"\n'
    "\n"
    "\n"
    "def _save_results(results: dict, path: Path):\n"
    "    \"\"\"Guarda {model_name: DataFrame} como CSV con columna 'model'.\"\"\"\n"
    "    frames = []\n"
    "    for mname, df in results.items():\n"
    "        tmp = df.copy()\n"
    "        tmp.insert(0, 'model', mname)\n"
    "        frames.append(tmp)\n"
    "    pd.concat(frames, ignore_index=True).to_csv(path, index=False)\n"
    "\n"
    "\n"
    "def _load_results(path: Path) -> dict:\n"
    "    \"\"\"Carga CSV de vuelta a {model_name: DataFrame}.\"\"\"\n"
    "    df = pd.read_csv(path)\n"
    "    return {\n"
    "        mname: grp.drop(columns='model').reset_index(drop=True)\n"
    "        for mname, grp in df.groupby('model', sort=False)\n"
    "    }\n"
    "\n"
    "\n"
    "def run_exp(dgp, make_models_fn, dgp_params, exp_id,\n"
    "            T_list=T_LIST, R_list=R_LIST, H=H, seed=SEED):\n"
    "    \"\"\"\n"
    "    Corre MC para todas las combinaciones (T, R).\n"
    "    Si el CSV ya existe, lo carga sin re-simular.\n"
    "    Retorna {(T, R): {model_name: DataFrame}}.\n"
    "    \"\"\"\n"
    "    n_runs = len(T_list) * len(R_list)\n"
    "    combos = ', '.join(f'(T={t}, R={r})' for t in T_list for r in R_list)\n"
    "    print(f'Exp {exp_id}: {n_runs} ejecución(es) programada(s) → {combos}')\n"
    "\n"
    "    all_results = {}\n"
    "    for T in T_list:\n"
    "        for R in R_list:\n"
    "            cache = _cache_path(exp_id, T, R)\n"
    "            if cache.exists():\n"
    "                print(f'  T={T}, R={R}: cargando {cache.name} ...')\n"
    "                all_results[(T, R)] = _load_results(cache)\n"
    "                continue\n"
    "\n"
    "            print(f'  T={T}, R={R}: simulando ...', end=' ', flush=True)\n"
    "            dgp.rng = np.random.default_rng(seed)\n"
    "            models = make_models_fn(T)\n"
    "            engine = MonteCarloEngine(dgp, models, seed=seed)\n"
    "            t0 = time.time()\n"
    "            results = engine.run_monte_carlo(\n"
    "                R, T, H, dgp_params, verbose=False)\n"
    "            elapsed = time.time() - t0\n"
    "            print(f'OK ({elapsed:.0f}s)')\n"
    "\n"
    "            _save_results(results, cache)\n"
    "            all_results[(T, R)] = results\n"
    "\n"
    "    return all_results   # {(T, R): {model_name: DataFrame}}\n"
    "\n"
    "\n"
    "def compute_blocks(results_TR: dict):\n"
    "    \"\"\"Dado {model_name: df}, calcula promedios h=1-12 y h=13-24.\"\"\"\n"
    "    out = {}\n"
    "    for mname, df in results_TR.items():\n"
    '        df_h = df[df["horizon"] != "avg_all"].copy()\n'
    '        df_h["horizon"] = pd.to_numeric(df_h["horizon"], errors="coerce")\n'
    "        out[mname] = {\n"
    '            "h=1-12":  df_h[df_h["horizon"] <= 12].mean(numeric_only=True),\n'
    '            "h=13-24": df_h[df_h["horizon"] >= 13].mean(numeric_only=True),\n'
    "        }\n"
    "    return out\n"
    "\n"
    "\n"
    "def results_table(all_results):\n"
    "    \"\"\"Muestra tabla comparativa de métricas por bloque para todos los (T, R).\"\"\"\n"
    "    # Detect available metric columns from first DataFrame\n"
    "    sample_df = next(iter(next(iter(all_results.values())).values()))\n"
    "    numeric_cols = [c for c in sample_df.columns\n"
    "                    if c not in ('horizon',) and sample_df[c].dtype != object]\n"
    "\n"
    "    rows = []\n"
    "    for (T, R), res_TR in sorted(all_results.items()):\n"
    "        for mname, blk in compute_blocks(res_TR).items():\n"
    "            for bname, m in blk.items():\n"
    "                row = {'T': T, 'R': R, 'Modelo': mname, 'Bloque': bname}\n"
    "                for col in numeric_cols:\n"
    "                    if col in m.index:\n"
    "                        row[col] = round(float(m[col]), 4)\n"
    "                rows.append(row)\n"
    "\n"
    "    df = pd.DataFrame(rows).set_index(['T', 'R', 'Modelo', 'Bloque'])\n"
    "    grad_cols = [c for c in ['rmse', 'mae'] if c in df.columns]\n"
    "    display(df.style.format(precision=4)\n"
    "              .background_gradient(subset=grad_cols, cmap='YlOrRd'))\n"
    "\n"
    "\n"
    "def plot_rep(dgp, make_models_fn, dgp_params,\n"
    "             T=200, H=H, seed=SEED, title=''):\n"
    "    \"\"\"Visualización de una simulación representativa.\"\"\"\n"
    "    dgp_r = dgp.__class__(seed=seed)\n"
    "    y = dgp_r.simulate(T=T, **dgp_params)\n"
    "    y_train, y_test = y[:-H], y[-H:]\n"
    "    models = make_models_fn(T)\n"
    "\n"
    "    fig, ax = plt.subplots(figsize=(13, 4))\n"
    "    x_tr = np.arange(len(y_train))\n"
    "    x_te = np.arange(len(y_train), T)\n"
    "\n"
    '    ax.plot(x_tr, y_train, color="steelblue", lw=1.4, alpha=0.85, label="Histórico")\n'
    '    ax.plot(x_te, y_test, "k--", lw=1.5, label="Observado (test)")\n'
    "    ax.axvline(len(y_train) - 0.5, color='gray', ls=':', lw=1, alpha=0.6)\n"
    "\n"
    "    palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'olive']\n"
    "    for i, m in enumerate(models):\n"
    "        m.fit(y_train)\n"
    "        y_hat = m.forecast(H)\n"
    "        ax.plot(x_te, y_hat, color=palette[i % len(palette)],\n"
    "                lw=1.5, marker='o', ms=3, label=m.name)\n"
    "        if m.supports_intervals:\n"
    "            lo, hi = m.forecast_intervals(H, level=0.95)\n"
    "            ax.fill_between(x_te, lo, hi,\n"
    "                            color=palette[i % len(palette)],\n"
    "                            alpha=0.12, label='_nolegend_')\n"
    "\n"
    "    ax.set(title=title, xlabel='t', ylabel='$Y_t$')\n"
    "    ax.legend(fontsize=9)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "def plot_metrics(all_results, title='', metrics=('rmse', 'bias')):\n"
    "    \"\"\"Gráficos de métricas vs h=1..24 por modelo, subplots por (T, R).\"\"\"\n"
    "    keys = sorted(all_results.keys())\n"
    "    fig, axes = plt.subplots(\n"
    "        len(metrics), len(keys),\n"
    "        figsize=(7 * len(keys), 3.5 * len(metrics)),\n"
    "        squeeze=False,\n"
    "    )\n"
    "    palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'steelblue']\n"
    "\n"
    "    for col, (T, R) in enumerate(keys):\n"
    "        for row, metric in enumerate(metrics):\n"
    "            ax = axes[row][col]\n"
    "            for i, (mname, df) in enumerate(all_results[(T, R)].items()):\n"
    '                df_h = df[df["horizon"] != "avg_all"].copy()\n'
    '                df_h["horizon"] = pd.to_numeric(df_h["horizon"], errors="coerce")\n'
    "                if metric not in df_h.columns:\n"
    "                    continue\n"
    "                ax.plot(df_h['horizon'], df_h[metric],\n"
    "                        label=mname, color=palette[i % len(palette)], lw=1.5)\n"
    "            ax.axvline(12.5, color='gray', ls=':', lw=0.8, alpha=0.5)\n"
    "            ax.set(\n"
    "                title=f'T={T}, R={R} — {metric.upper()}',\n"
    "                xlabel='Horizonte h',\n"
    "                ylabel=metric.upper(),\n"
    "            )\n"
    "            ax.legend(fontsize=8)\n"
    "\n"
    "    fig.suptitle(title, fontsize=12)\n"
    "    plt.tight_layout()\n"
    "    plt.show()"
)

# ─── Experiments definition ───────────────────────────────────────────────────
# Updated per experiments.md (only Core models — Naive/Drift/SeasonalNaive moved to Additional)
EXPS = [
    (
        "1.1",
        "AR1",
        {"phi": 0.3},
        "lambda T: [ARIMAModel((1,0,0)), chronos]",
        (
            "**DGP:** AR(1) baja persistencia — $Y_t = 0.3\\,Y_{t-1} + \\varepsilon_t$  \n"
            "**Core:** ARIMA(1,0,0), Chronos-2  \n"
            "**Adicionales (no implementados aquí):** ETS(A,N,N), Theta, Naive, Drift"
        ),
    ),
    (
        "1.2",
        "AR1",
        {"phi": 0.9},
        "lambda T: [ARIMAModel((1,0,0)), chronos]",
        (
            "**DGP:** AR(1) alta persistencia — $Y_t = 0.9\\,Y_{t-1} + \\varepsilon_t$  \n"
            "**Core:** ARIMA(1,0,0), Chronos-2  \n"
            "**Adicionales:** ETS(A,A,N), Theta, Naive"
        ),
    ),
    (
        "1.3",
        "RandomWalk",
        {"drift": 0.0},
        "lambda T: [ARIMAModel((0,1,0)), chronos]",
        (
            "**DGP:** Random Walk I(1) sin drift — $Y_t = Y_{t-1} + \\varepsilon_t$  \n"
            "**Core:** ARIMA(0,1,0), Chronos-2  \n"
            "**Adicionales:** ETS(A,A,N), Theta, Drift"
        ),
    ),
    (
        "1.4",
        "RandomWalk",
        {"drift": 0.5},
        "lambda T: [ARIMAModel((0,1,0)), chronos]",
        (
            "**DGP:** Random Walk I(1) con drift — $Y_t = 0.5 + Y_{t-1} + \\varepsilon_t$  \n"
            "**Core:** ARIMA(0,1,0), Chronos-2  \n"
            "**Adicionales:** ETS(A,A,N), Theta, Drift"
        ),
    ),
    (
        "1.5",
        "AR1WithTrend",
        {"intercept": 5.0, "trend_coeff": 0.1, "phi": 0.6},
        "lambda T: [ARIMAWithTrendModel((1,0,0), trend='ct'), chronos]",
        (
            "**DGP:** AR(1) + tendencia — $Y_t = 5 + 0.1t + 0.6\\,Y_{t-1} + \\varepsilon_t$  \n"
            "**Core:** ARIMA(1,0,0)+trend (trend='ct'), Chronos-2  \n"
            "**Adicionales:** Holt-Winters, ETS, Theta"
        ),
    ),
    (
        "1.6",
        "SeasonalDGP",
        {"phi": 0.5, "Phi": 0.3, "s": 4, "integrated": False},
        "lambda T: [SARIMAModel((1,0,0),(1,0,0,4)), chronos]",
        (
            "**DGP:** SARIMA trimestral (s=4) — $(1-0.5L)(1-0.3L^4)Y_t = \\varepsilon_t$  \n"
            "**Core:** SARIMA(1,0,0)(1,0,0)_4, Chronos-2  \n"
            "**Adicionales:** ETS(A,A,A), Seasonal Naive"
        ),
    ),
    (
        "1.7",
        "SeasonalDGP",
        {"integrated": True, "s": 12},
        "lambda T: [SARIMAModel((0,1,0),(0,1,0,12)), chronos]",
        (
            "**DGP:** SARIMA mensual (s=12) — $(1-L)(1-L^{12})Y_t = \\varepsilon_t$  \n"
            "**Core:** SARIMA(0,1,0)(0,1,0)_12, Chronos-2  \n"
            "**Adicionales:** Holt-Winters multiplicativo, ETS, Seasonal Naive"
        ),
    ),
    (
        "1.8",
        "AR1WithBreak",
        {"phi_before": 0.3, "phi_after": 0.8},
        "lambda T: [ARIMAWithBreakModel((1,0,0), T_total=T), chronos]",
        (
            "**DGP:** AR(1) con quiebre en $T/2$ — $\\phi$ cambia de 0.3 a 0.8  \n"
            "**Core:** ARIMA(1,0,0)+break (dummy exógena), Chronos-2  \n"
            "**Adicionales:** ARIMA(1,0,0) sin quiebre, ETS"
        ),
    ),
]

cells = [c0, c1, c2]

for exp_id, dgp_cls, dgp_params, make_fn_src, md_desc in EXPS:
    slug = exp_id.replace(".", "_")

    # Markdown header
    cells.append(md(f"---\n## Experimento {exp_id}\n\n{md_desc}"))

    # Run MC (with save/load)
    cells.append(code(
        f"dgp_{slug}         = {dgp_cls}(seed=SEED)\n"
        f"make_models_{slug} = {make_fn_src}\n"
        f"dgp_params_{slug}  = {repr(dgp_params)}\n"
        f"\n"
        f"results_{slug} = run_exp(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    exp_id='{exp_id}',\n"
        f")"
    ))

    # Visualization + table + metrics
    cells.append(code(
        f"# Visualización representativa con banda de intervalo 95% (T=200)\n"
        f"plot_rep(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    T=200, title=f\"Exp {exp_id} — Simulación representativa (T=200, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Tabla de métricas por bloque\n"
        f'print("Tabla de métricas — Exp {exp_id}")\n'
        f"results_table(results_{slug})\n"
        f"\n"
        f"# Gráficos de métricas por horizonte\n"
        f"plot_metrics(\n"
        f"    results_{slug},\n"
        f'    title=f"Exp {exp_id} — Métricas por horizonte"\n'
        f")"
    ))

# ─── Build and write notebook ─────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "mectesis-venv",
            "language": "python",
            "name": "mectesis-venv",
        },
        "language_info": {"name": "python", "version": "3.12.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook escrito en: {OUT}")
print(f"Total de celdas: {len(cells)}")
