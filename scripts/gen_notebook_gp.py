"""
Genera notebooks/experimentos_gp.ipynb

Experimento independiente: Gaussian Process DGP (KernelSynth-style).
Hipótesis: Chronos gana ampliamente porque fue entrenado sobre muestras de GP
con kernels compuestos (KernelSynth, Ansari et al. 2024).

Run once:  python scripts/gen_notebook_gp.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "experimentos_gp.ipynb"
RESULTS_DIR = "results/gp"


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
    "# Experimento GP — KernelSynth: ¿Puede Chronos Ganar?\n\n"
    "**Hipótesis:** Chronos fue entrenado con datos sintéticos generados por "
    "Gaussian Processes con kernels compuestos (*KernelSynth*, Ansari et al. 2024). "
    "Para este tipo de serie, la covarianza del GP **no tiene representación ARMA de orden finito**, "
    "por lo que cualquier modelo clásico con parametrización fija especifica mal el proceso. "
    "Chronos, en cambio, puede reconocer y extrapolar el patrón sin estimación paramétrica.  \n\n"
    "**DGP:** `GPKernelSynthDGP` — muestras de GP con kernel RBF, Periódico, o su combinación  \n"
    "**Kernels:**  \n"
    "- **RBF:** $k(t,t') = \\sigma^2 \\exp\\!\\left(-\\tfrac{|t-t'|^2}{2\\ell^2}\\right)$ — tendencia suave  \n"
    "- **Periódico:** $k(t,t') = \\sigma^2 \\exp\\!\\left(-\\tfrac{2\\sin^2(\\pi|t-t'|/p)}{\\ell^2}\\right)$ — estructura estacional  \n"
    "- **RBF + Periódico:** combinación (más cercano al KernelSynth del paper)  \n\n"
    "**Setup:** T ∈ {50, 200} | H = 24 | R = 500 | Semilla = 3649  \n"
    "**Resultados:** guardados en `results/gp/` — si existen se cargan sin re-simular\n\n"
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
    "from mectesis.dgp import GPKernelSynthDGP\n"
    "from mectesis.models import (\n"
    "    ARIMAModel, ChronosModel,\n"
    "    SARIMAModel, ETSModel, ThetaModel,\n"
    ")\n"
    "from mectesis.simulation import MonteCarloEngine\n"
    "\n"
    "SEED    = 3649\n"
    "H       = 24\n"
    "R_LIST  = [500]\n"
    "T_LIST  = [50, 200]\n"
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

# ─── Cell 2: Helper functions (identical pattern to gen_notebook.py) ──────────
c2 = code(
    "def _cache_path(exp_id: str, T: int, R: int) -> Path:\n"
    '    return RESULTS / f"exp_{exp_id.replace(\'.\', \'_\')}_T{T}_R{R}.csv"\n'
    "\n"
    "\n"
    "def _save_results(results: dict, path: Path):\n"
    "    frames = []\n"
    "    for mname, df in results.items():\n"
    "        tmp = df.copy()\n"
    "        tmp.insert(0, 'model', mname)\n"
    "        frames.append(tmp)\n"
    "    pd.concat(frames, ignore_index=True).to_csv(path, index=False)\n"
    "\n"
    "\n"
    "def _load_results(path: Path) -> dict:\n"
    "    df = pd.read_csv(path)\n"
    "    return {\n"
    "        mname: grp.drop(columns='model').reset_index(drop=True)\n"
    "        for mname, grp in df.groupby('model', sort=False)\n"
    "    }\n"
    "\n"
    "\n"
    "def run_exp(dgp, make_models_fn, dgp_params, exp_id,\n"
    "            T_list=T_LIST, R_list=R_LIST, H=H, seed=SEED):\n"
    "    n_runs = len(T_list) * len(R_list)\n"
    "    combos = ', '.join(f'(T={t}, R={r})' for t in T_list for r in R_list)\n"
    "    print(f'Exp {exp_id}: {n_runs} ejecución(es) → {combos}')\n"
    "    all_results = {}\n"
    "    for T in T_list:\n"
    "        for R in R_list:\n"
    "            cache = _cache_path(exp_id, T, R)\n"
    "            if cache.exists():\n"
    "                print(f'  T={T}, R={R}: cargando {cache.name} ...')\n"
    "                all_results[(T, R)] = _load_results(cache)\n"
    "                continue\n"
    "            print(f'  T={T}, R={R}: simulando ...', end=' ', flush=True)\n"
    "            dgp.rng = np.random.default_rng(seed)\n"
    "            models = make_models_fn(T)\n"
    "            engine = MonteCarloEngine(dgp, models, seed=seed)\n"
    "            t0 = time.time()\n"
    "            results = engine.run_monte_carlo(R, T, H, dgp_params, verbose=False)\n"
    "            elapsed = time.time() - t0\n"
    "            print(f'OK ({elapsed:.0f}s)')\n"
    "            _save_results(results, cache)\n"
    "            all_results[(T, R)] = results\n"
    "    return all_results\n"
    "\n"
    "\n"
    "def compute_blocks(results_TR: dict):\n"
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
    "    seen: dict = {}\n"
    "    for res_TR in all_results.values():\n"
    "        for df in res_TR.values():\n"
    "            for c in df.columns:\n"
    "                if c not in ('horizon',) and df[c].dtype != object:\n"
    "                    seen[c] = True\n"
    "    numeric_cols = list(seen)\n"
    "    rows = []\n"
    "    for (T, R), res_TR in sorted(all_results.items()):\n"
    "        for mname, blk in compute_blocks(res_TR).items():\n"
    "            for bname, m in blk.items():\n"
    "                row = {'T': T, 'R': R, 'Modelo': mname, 'Bloque': bname}\n"
    "                for col in numeric_cols:\n"
    "                    if col in m.index:\n"
    "                        row[col] = round(float(m[col]), 4)\n"
    "                rows.append(row)\n"
    "    df = pd.DataFrame(rows).set_index(['T', 'R', 'Modelo', 'Bloque'])\n"
    "    grad_cols = [c for c in ['rmse', 'mae'] if c in df.columns]\n"
    "    display(df.style.format(precision=4)\n"
    "              .background_gradient(subset=grad_cols, cmap='YlOrRd'))\n"
    "\n"
    "\n"
    "def plot_rep(dgp, make_models_fn, dgp_params,\n"
    "             T=200, H=H, seed=SEED, title=''):\n"
    "    dgp_r = dgp.__class__(seed=seed)\n"
    "    y = dgp_r.simulate(T=T, **dgp_params)\n"
    "    y_train, y_test = y[:-H], y[-H:]\n"
    "    models = make_models_fn(T)\n"
    "    fig, ax = plt.subplots(figsize=(13, 4))\n"
    "    x_tr = np.arange(len(y_train))\n"
    "    x_te = np.arange(len(y_train), T)\n"
    '    ax.plot(x_tr, y_train, color="steelblue", lw=1.4, alpha=0.85, label="Histórico")\n'
    '    ax.plot(x_te, y_test, "k--", lw=1.5, label="Observado (test)")\n'
    "    ax.axvline(len(y_train) - 0.5, color='gray', ls=':', lw=1, alpha=0.6)\n"
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
    "    ax.set(title=title, xlabel='t', ylabel='$Y_t$')\n"
    "    ax.legend(fontsize=9)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "def plot_metrics(all_results, title='', metrics=('rmse', 'bias')):\n"
    "    keys = sorted(all_results.keys())\n"
    "    fig, axes = plt.subplots(\n"
    "        len(metrics), len(keys),\n"
    "        figsize=(7 * len(keys), 3.5 * len(metrics)),\n"
    "        squeeze=False,\n"
    "    )\n"
    "    palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'steelblue']\n"
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
    "    fig.suptitle(title, fontsize=12)\n"
    "    plt.tight_layout()\n"
    "    plt.show()"
)

# ─── Experiments ──────────────────────────────────────────────────────────────
EXPS = [
    (
        "GP.1",
        {"kernel": "rbf", "lengthscale_rbf": 30.0, "sigma_rbf": 1.0, "noise_std": 0.3},
        "lambda T: [ARIMAModel((1,1,1)), ETSModel(trend='add'), chronos]",
        (
            "**DGP:** GP — kernel RBF (tendencia suave no lineal)  \n"
            "$$f \\sim \\mathcal{GP}\\!\\left(0,\\; \\sigma^2 e^{-|t-t'|^2/2\\ell^2}\\right), "
            "\\quad y_t = f(t) + \\varepsilon_t$$\n"
            "con $\\ell=30$, $\\sigma=1$, $\\sigma_\\varepsilon=0.3$  \n\n"
            "**Modelos clásicos:** ARIMA(1,1,1), ETS(A,A,N)  \n"
            "**Hipótesis:** La tendencia suave no tiene representación ARMA finita — "
            "Chronos reconoce el patrón por su entrenamiento en GPs."
        ),
        "GP RBF — tendencia suave",
    ),
    (
        "GP.2",
        {
            "kernel": "periodic",
            "period": 12.0, "lengthscale_per": 1.0, "sigma_per": 1.0,
            "noise_std": 0.3,
        },
        "lambda T: [SARIMAModel((1,0,1),(1,0,1,12)), ETSModel(trend=None, seasonal='add', seasonal_periods=12), chronos]",
        (
            "**DGP:** GP — kernel Periódico (s=12)  \n"
            "$$f \\sim \\mathcal{GP}\\!\\left(0,\\; \\sigma^2 "
            "e^{-2\\sin^2(\\pi|t-t'|/12)/\\ell^2}\\right), "
            "\\quad y_t = f(t) + \\varepsilon_t$$\n"
            "con $p=12$, $\\ell=1$, $\\sigma=1$, $\\sigma_\\varepsilon=0.3$  \n\n"
            "**Modelos clásicos:** SARIMA(1,0,1)(1,0,1)_12, ETS con estacionalidad  \n"
            "**Hipótesis:** El kernel periódico genera estructura estacional suave que "
            "el SARIMA aproxima con parámetros lineales — Chronos la captura directamente."
        ),
        "GP Periódico — estacionalidad suave (s=12)",
    ),
    (
        "GP.3",
        {
            "kernel": "rbf+periodic",
            "lengthscale_rbf": 30.0, "sigma_rbf": 1.0,
            "period": 12.0, "lengthscale_per": 1.0, "sigma_per": 0.8,
            "noise_std": 0.3,
        },
        "lambda T: [SARIMAModel((1,1,1),(1,0,1,12)), ETSModel(trend='add', seasonal='add', seasonal_periods=12), ThetaModel(), chronos]",
        (
            "**DGP:** GP — kernel RBF + Periódico (**KernelSynth completo**)  \n"
            "$$K = \\sigma_{\\text{rbf}}^2 e^{-|t-t'|^2/2\\ell_{\\text{rbf}}^2} "
            "+ \\sigma_{\\text{per}}^2 e^{-2\\sin^2(\\pi|t-t'|/12)/\\ell_{\\text{per}}^2} "
            "+ \\sigma_\\varepsilon^2 I$$\n"
            "con $\\ell_{\\text{rbf}}=30$, $\\sigma_{\\text{rbf}}=1$, "
            "$p=12$, $\\sigma_{\\text{per}}=0.8$, $\\sigma_\\varepsilon=0.3$  \n\n"
            "**Modelos clásicos:** SARIMA(1,1,1)(1,0,1)_12, ETS(A,A,A), Theta  \n"
            "**Hipótesis:** Este DGP replica directamente el proceso de generación "
            "sintética de KernelSynth usado en el entrenamiento de Chronos. "
            "La ventaja de Chronos debería ser máxima aquí."
        ),
        "GP RBF + Periódico — KernelSynth completo",
    ),
]

cells = [c0, c1, c2]

for exp_id, dgp_params, make_fn_src, md_desc, exp_name in EXPS:
    slug = exp_id.replace(".", "_")

    cells.append(md(f"---\n## Experimento {exp_id}\n\n{md_desc}"))

    cells.append(code(
        f"dgp_{slug}         = GPKernelSynthDGP(seed=SEED)\n"
        f"make_models_{slug} = {make_fn_src}\n"
        f"dgp_params_{slug}  = {repr(dgp_params)}\n"
        f"\n"
        f"results_{slug} = run_exp(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    exp_id='{exp_id}',\n"
        f")"
    ))

    cells.append(code(
        f"plot_rep(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    T=T_LIST[0],\n"
        f"    title='Exp {exp_id}: {exp_name} — T={{T_LIST[0]}}, seed={{SEED}}'\n"
        f")\n"
        f"plot_rep(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    T=T_LIST[1],\n"
        f"    title='Exp {exp_id}: {exp_name} — T={{T_LIST[1]}}, seed={{SEED}}'\n"
        f")\n"
        f'print("Tabla de métricas — Exp {exp_id}")\n'
        f"results_table(results_{slug})\n"
        f"plot_metrics(\n"
        f"    results_{slug},\n"
        f'    title="Exp {exp_id} — Métricas por horizonte",\n'
        f'    metrics=("rmse", "crps", "winkler_95", "bias")\n'
        f")"
    ))

# ─── Build notebook ────────────────────────────────────────────────────────────
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
