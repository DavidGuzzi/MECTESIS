"""
Generates notebooks/experimentos_covariados.ipynb
Run once: python scripts/gen_notebook_covariates.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "experimentos_covariados.ipynb"
RESULTS_DIR = "results/covariates"


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
    "# Experimentos con Covariables 3.1–3.6\n\n"
    "**Tesis MEC** — Comparación TSFMs vs Modelos Clásicos bajo DGPs controlados  \n"
    "**Setup:** T ∈ {50, 200} | H = 24 | R_LIST = [500] | Semilla = 3649  \n"
    "**Métricas punto:** Bias, Varianza, MSE, RMSE, MAE  \n"
    "**Métricas probabilísticas:** CRPS, Cobertura 80%/95%, Amplitud 80%/95%, Winkler Score 80%/95%  \n"
    "**Resultados:** guardados en `results/covariates/` — si existen se cargan sin re-simular\n\n"
    "**Nota:** En todos los experimentos las covariables son *completamente observadas*: "
    "se proveen los valores históricos (`X_train`) y los valores futuros conocidos (`X_future`) "
    "a los modelos que los aceptan. Chronos-2 recibe las covariables vía la API de "
    "`past_covariates` / `future_covariates`. Los modelos sin soporte de covariables "
    "(ARIMA, VAR, Naive) reciben los mismos `X` pero los ignoran silenciosamente.\n\n"
    "---"
)

# ─── Cell 1: Imports & constants ──────────────────────────────────────────────
c1 = code(
    "import os\n"
    "import warnings\n"
    'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"\n'
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "import time\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "from IPython.display import display\n"
    "\n"
    "from mectesis.dgp import (\n"
    "    ARIMAX_DGP, ARIMAX2Cov_DGP, ARIMAX_GARCH_DGP,\n"
    "    VARX_DGP, ADL_ECM_DGP,\n"
    ")\n"
    "from mectesis.models import (\n"
    "    ARIMAModel, SARIMAXModel, VARMAXModel, ARDLModel,\n"
    "    VARModel, SeasonalNaiveModel,\n"
    "    ARGARCHModel,\n"
    "    ChronosModel, ChronosCovariateModel,\n"
    "    ChronosMultivariateModel, ChronosPerVarModel,\n"
    "    ChronosMultivariateCovariateModel,\n"
    ")\n"
    "from mectesis.simulation import (\n"
    "    CovariateMonteCarloEngine, CovariateMultivariateEngine,\n"
    ")\n"
    "\n"
    "# ── Parámetros globales ───────────────────────────────────────────────────\n"
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
    '_chronos_base       = ChronosModel(device="cpu")\n'
    "chronos_cov1        = ChronosCovariateModel(_chronos_base, n_covariates=1)\n"
    "chronos_cov2        = ChronosCovariateModel(_chronos_base, n_covariates=2,\n"
    '                          cov_names=["x0", "x1"])\n'
    "chronos_mv_cov1     = ChronosMultivariateCovariateModel(_chronos_base, n_covariates=1)\n"
    "chronos_mv          = ChronosMultivariateModel(_chronos_base)\n"
    "chronos_mv_ind      = ChronosPerVarModel(_chronos_base)\n"
    'print("Chronos-2 listo.")'
)

# ─── Cell 2: Helper functions (univariate covariates) ─────────────────────────
c2 = code(
    "# ─── Funciones auxiliares — experimentos con covariables ──────────────────\n"
    "\n"
    "def _cache_path(exp_id: str, T: int, R: int) -> Path:\n"
    '    return RESULTS / f"exp_{exp_id.replace(\'.\', \'_\')}_T{T}_R{R}.csv"\n'
    "\n"
    "\n"
    "# ── Univariate helpers ────────────────────────────────────────────────────\n"
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
    "def run_exp_cov(dgp, make_models_fn, dgp_params, exp_id,\n"
    "                T_list=T_LIST, R_list=R_LIST, H=H, seed=SEED):\n"
    "    \"\"\"\n"
    "    Corre MC con covariables para todas las combinaciones (T, R).\n"
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
    "            models  = make_models_fn(T)\n"
    "            engine  = CovariateMonteCarloEngine(dgp, models, seed=seed)\n"
    "            t0 = time.time()\n"
    "            results = engine.run_monte_carlo(R, T, H, dgp_params, verbose=False)\n"
    "            elapsed = time.time() - t0\n"
    "            print(f'OK ({elapsed:.0f}s)')\n"
    "            _save_results(results, cache)\n"
    "            all_results[(T, R)] = results\n"
    "\n"
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
    "    \"\"\"Tabla comparativa por (T, R, Modelo, Bloque) con gradiente en rmse/mae.\"\"\"\n"
    "    seen: dict = {}\n"
    "    for res_TR in all_results.values():\n"
    "        for df in res_TR.values():\n"
    "            for c in df.columns:\n"
    "                if c != 'horizon' and df[c].dtype != object:\n"
    "                    seen[c] = True\n"
    "    numeric_cols = list(seen)\n"
    "\n"
    "    rows = []\n"
    "    for (T, R), res_TR in sorted(all_results.items()):\n"
    "        for mname, blks in compute_blocks(res_TR).items():\n"
    "            for bname, m in blks.items():\n"
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
    "def plot_rep_cov(dgp, make_models_fn, dgp_params,\n"
    "                 T=200, H=H, seed=SEED, title=''):\n"
    "    \"\"\"Simulación representativa con covariables — plot Y + forecasts.\"\"\"\n"
    "    import copy\n"
    "    dgp_r = copy.deepcopy(dgp)\n"
    "    dgp_r.rng = np.random.default_rng(seed)\n"
    "    data = dgp_r.simulate(T=T, **dgp_params)\n"
    "    y, X = data['y'], data['X']\n"
    "    y_train, y_test = y[:-H], y[-H:]\n"
    "    X_train, X_future = X[:-H], X[-H:]\n"
    "    models = make_models_fn(T)\n"
    "\n"
    "    palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'olive']\n"
    "    fig, ax = plt.subplots(figsize=(13, 4))\n"
    "\n"
    "    for m in models:\n"
    "        fkw = {'X_train': X_train} if m.supports_covariates else {}\n"
    "        m.fit(y_train, **fkw)\n"
    "\n"
    "    x_tr = np.arange(len(y_train))\n"
    "    x_te = np.arange(len(y_train), T)\n"
    "\n"
    "    ax.plot(x_tr, y_train, color='steelblue', lw=1.4, alpha=0.85,\n"
    "            label='Histórico')\n"
    "    ax.plot(x_te, y_test, 'k--', lw=1.5, label='Observado (test)')\n"
    "    ax.axvline(len(y_train) - 0.5, color='gray', ls=':', lw=1, alpha=0.6)\n"
    "\n"
    "    for i, m in enumerate(models):\n"
    "        pkw = {'X_future': X_future} if m.supports_covariates else {}\n"
    "        y_hat = m.forecast(H, **pkw)\n"
    "        ax.plot(x_te, y_hat, color=palette[i % len(palette)],\n"
    "                lw=1.5, marker='o', ms=3, label=m.name)\n"
    "        if m.supports_intervals:\n"
    "            lo, hi = m.forecast_intervals(H, level=0.95, **pkw)\n"
    "            ax.fill_between(x_te, lo, hi,\n"
    "                            color=palette[i % len(palette)],\n"
    "                            alpha=0.12, label='_nolegend_')\n"
    "\n"
    "    ax.set(title=title, xlabel='t', ylabel='Y')\n"
    "    ax.legend(fontsize=9)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "def plot_metrics(all_results, title='', metrics=('rmse', 'bias')):\n"
    "    \"\"\"Grilla (metric × (T,R)) con curvas por modelo.\"\"\"\n"
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
    "    plt.show()\n"
    "\n"
    "\n"
    "# ── Multivariate helpers (re-used from gen_notebook_multivariate pattern) ──\n"
    "\n"
    "def _save_results_mv(results: dict, path: Path):\n"
    "    frames = []\n"
    "    for mname, var_dict in results.items():\n"
    "        for var_idx, df in var_dict.items():\n"
    "            tmp = df.copy()\n"
    "            tmp.insert(0, 'var', var_idx)\n"
    "            tmp.insert(0, 'model', mname)\n"
    "            frames.append(tmp)\n"
    "    pd.concat(frames, ignore_index=True).to_csv(path, index=False)\n"
    "\n"
    "\n"
    "def _load_results_mv(path: Path) -> dict:\n"
    "    df = pd.read_csv(path)\n"
    "    results = {}\n"
    "    for mname, mgrp in df.groupby('model', sort=False):\n"
    "        results[mname] = {}\n"
    "        for var_idx, vgrp in mgrp.groupby('var', sort=True):\n"
    "            results[mname][int(var_idx)] = (\n"
    "                vgrp.drop(columns=['model', 'var']).reset_index(drop=True)\n"
    "            )\n"
    "    return results\n"
    "\n"
    "\n"
    "def run_exp_mv_cov(dgp, make_models_fn, dgp_params, exp_id,\n"
    "                   T_list=T_LIST, R_list=R_LIST, H=H, seed=SEED):\n"
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
    "                all_results[(T, R)] = _load_results_mv(cache)\n"
    "                continue\n"
    "\n"
    "            print(f'  T={T}, R={R}: simulando ...', end=' ', flush=True)\n"
    "            dgp.rng = np.random.default_rng(seed)\n"
    "            models  = make_models_fn(T)\n"
    "            engine  = CovariateMultivariateEngine(dgp, models, seed=seed)\n"
    "            t0 = time.time()\n"
    "            results = engine.run_monte_carlo(R, T, H, dgp_params, verbose=False)\n"
    "            elapsed = time.time() - t0\n"
    "            print(f'OK ({elapsed:.0f}s)')\n"
    "            _save_results_mv(results, cache)\n"
    "            all_results[(T, R)] = results\n"
    "\n"
    "    return all_results\n"
    "\n"
    "\n"
    "def compute_blocks_mv(results_TR: dict):\n"
    "    out = {}\n"
    "    for mname, var_dict in results_TR.items():\n"
    "        out[mname] = {}\n"
    "        for var_idx, df in var_dict.items():\n"
    '            df_h = df[df["horizon"] != "avg_all"].copy()\n'
    '            df_h["horizon"] = pd.to_numeric(df_h["horizon"], errors="coerce")\n'
    "            out[mname][var_idx] = {\n"
    '                "h=1-12":  df_h[df_h["horizon"] <= 12].mean(numeric_only=True),\n'
    '                "h=13-24": df_h[df_h["horizon"] >= 13].mean(numeric_only=True),\n'
    "            }\n"
    "    return out\n"
    "\n"
    "\n"
    "def results_table_mv(all_results, var_names=None):\n"
    "    seen: dict = {}\n"
    "    for res_TR in all_results.values():\n"
    "        for var_dict in res_TR.values():\n"
    "            for df in var_dict.values():\n"
    "                for c in df.columns:\n"
    "                    if c not in ('horizon',) and df[c].dtype != object:\n"
    "                        seen[c] = True\n"
    "    numeric_cols = list(seen)\n"
    "\n"
    "    rows = []\n"
    "    for (T, R), res_TR in sorted(all_results.items()):\n"
    "        for mname, blk_dict in compute_blocks_mv(res_TR).items():\n"
    "            for var_idx, blks in blk_dict.items():\n"
    "                vname = var_names[var_idx] if var_names else f'Y{var_idx+1}'\n"
    "                for bname, m in blks.items():\n"
    "                    row = {'T': T, 'R': R, 'Modelo': mname,\n"
    "                           'Variable': vname, 'Bloque': bname}\n"
    "                    for col in numeric_cols:\n"
    "                        if col in m.index:\n"
    "                            row[col] = round(float(m[col]), 4)\n"
    "                    rows.append(row)\n"
    "\n"
    "    df = pd.DataFrame(rows).set_index(['T', 'R', 'Modelo', 'Variable', 'Bloque'])\n"
    "    grad_cols = [c for c in ['rmse', 'mae'] if c in df.columns]\n"
    "    display(df.style.format(precision=4)\n"
    "              .background_gradient(subset=grad_cols, cmap='YlOrRd'))\n"
    "\n"
    "\n"
    "def plot_rep_mv_cov(dgp, make_models_fn, dgp_params, var_names,\n"
    "                    T=200, H=H, seed=SEED, title=''):\n"
    "    import copy\n"
    "    dgp_r = copy.deepcopy(dgp)\n"
    "    dgp_r.rng = np.random.default_rng(seed)\n"
    "    data = dgp_r.simulate(T=T, **dgp_params)\n"
    "    y, X = data['y'], data['X']\n"
    "    k = y.shape[1]\n"
    "    y_train, y_test = y[:-H], y[-H:]\n"
    "    X_train, X_future = X[:-H], X[-H:]\n"
    "    models = make_models_fn(T)\n"
    "\n"
    "    palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'olive']\n"
    "    fig, axes = plt.subplots(k, 1, figsize=(13, 3.5 * k), squeeze=False)\n"
    "\n"
    "    for m in models:\n"
    "        fkw = {'X_train': X_train} if m.supports_covariates else {}\n"
    "        m.fit(y_train, **fkw)\n"
    "\n"
    "    x_tr = np.arange(len(y_train))\n"
    "    x_te = np.arange(len(y_train), T)\n"
    "\n"
    "    for j, ax in enumerate(axes[:, 0]):\n"
    "        vname = var_names[j] if var_names else f'Y{j+1}'\n"
    "        ax.plot(x_tr, y_train[:, j], color='steelblue', lw=1.4, alpha=0.85,\n"
    "                label='Histórico')\n"
    "        ax.plot(x_te, y_test[:, j], 'k--', lw=1.5, label='Observado (test)')\n"
    "        ax.axvline(len(y_train) - 0.5, color='gray', ls=':', lw=1, alpha=0.6)\n"
    "\n"
    "        for i, m in enumerate(models):\n"
    "            pkw = {'X_future': X_future} if m.supports_covariates else {}\n"
    "            y_hat = m.forecast(H, **pkw)\n"
    "            ax.plot(x_te, y_hat[:, j], color=palette[i % len(palette)],\n"
    "                    lw=1.5, marker='o', ms=3, label=m.name)\n"
    "            if m.supports_intervals:\n"
    "                lo, hi = m.forecast_intervals(H, level=0.95, **pkw)\n"
    "                ax.fill_between(x_te, lo[:, j], hi[:, j],\n"
    "                                color=palette[i % len(palette)],\n"
    "                                alpha=0.12, label='_nolegend_')\n"
    "\n"
    "        ax.set(title=f'{vname}', xlabel='t', ylabel=vname)\n"
    "        ax.legend(fontsize=9)\n"
    "\n"
    "    fig.suptitle(title, fontsize=12)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "def plot_metrics_mv(all_results, var_names, title='',\n"
    "                    metrics=('rmse', 'bias')):\n"
    "    first_TR    = next(iter(all_results.values()))\n"
    "    first_model = next(iter(first_TR.values()))\n"
    "    k = len(first_model)\n"
    "\n"
    "    for j in range(k):\n"
    "        vname = var_names[j] if var_names else f'Y{j+1}'\n"
    "        keys  = sorted(all_results.keys())\n"
    "        fig, axes = plt.subplots(\n"
    "            len(metrics), len(keys),\n"
    "            figsize=(7 * len(keys), 3.5 * len(metrics)),\n"
    "            squeeze=False,\n"
    "        )\n"
    "        palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'steelblue']\n"
    "\n"
    "        for col, (T, R) in enumerate(keys):\n"
    "            for row, metric in enumerate(metrics):\n"
    "                ax = axes[row][col]\n"
    "                for i, (mname, var_dict) in enumerate(\n"
    "                        all_results[(T, R)].items()):\n"
    "                    if j not in var_dict:\n"
    "                        continue\n"
    "                    df = var_dict[j]\n"
    '                    df_h = df[df["horizon"] != "avg_all"].copy()\n'
    '                    df_h["horizon"] = pd.to_numeric(\n'
    '                        df_h["horizon"], errors="coerce")\n'
    "                    if metric not in df_h.columns:\n"
    "                        continue\n"
    "                    ax.plot(df_h['horizon'], df_h[metric],\n"
    "                            label=mname, color=palette[i % len(palette)],\n"
    "                            lw=1.5)\n"
    "                ax.axvline(12.5, color='gray', ls=':', lw=0.8, alpha=0.5)\n"
    "                ax.set(\n"
    "                    title=f'T={T}, R={R} — {metric.upper()}',\n"
    "                    xlabel='Horizonte h',\n"
    "                    ylabel=metric.upper(),\n"
    "                )\n"
    "                ax.legend(fontsize=8)\n"
    "\n"
    "        fig.suptitle(f'{title} — {vname}', fontsize=12)\n"
    "        plt.tight_layout()\n"
    "        plt.show()"
)

# ─── Experiment definitions ────────────────────────────────────────────────────
# Univariate experiments (3.1–3.3, 3.5, 3.6): engine=CovariateMonteCarloEngine
# Multivariate experiment (3.4): engine=CovariateMultivariateEngine

EXPS_COV_UNIV = [
    (
        "3.1",
        "ARIMAX_DGP",
        "seed=SEED",
        "dict(phi=0.6, beta=0.8, sigma_y=1.0, sigma_x=1.0, rho_x=0.7)",
        (
            "lambda T: [\n"
            "    SARIMAXModel((1, 0, 0), name_suffix='con X'),\n"
            "    chronos_cov1,\n"
            "]"
        ),
        (
            "**DGP:** ARIMAX — efecto exógeno fuerte\n\n"
            "$$Y_t = 0.6\\,Y_{t-1} + 0.8\\,X_t + \\varepsilon_t, \\quad "
            "X_t = 0.7\\,X_{t-1} + \\eta_t$$\n\n"
            "**Modelos:** SARIMAX(1,0,0) con $X_t$, Chronos-2 con $X_t$ (past+future)\n\n"
            "**Hipótesis:** El efecto exógeno $\\beta=0.8$ es dominante; "
            "ambos modelos deben capturarlo. "
            "Chronos-2 debería aprovechar la señal exógena sin especificación explícita."
        ),
        "ARIMAX fuerte (β=0.8)",
    ),
    (
        "3.2",
        "ARIMAX_DGP",
        "seed=SEED",
        "dict(phi=0.6, beta=0.2, sigma_y=1.0, sigma_x=1.0, rho_x=0.7)",
        (
            "lambda T: [\n"
            "    SARIMAXModel((1, 0, 0), name_suffix='con X'),\n"
            "    chronos_cov1,\n"
            "]"
        ),
        (
            "**DGP:** ARIMAX — efecto exógeno débil\n\n"
            "$$Y_t = 0.6\\,Y_{t-1} + 0.2\\,X_t + \\varepsilon_t, \\quad "
            "X_t = 0.7\\,X_{t-1} + \\eta_t$$\n\n"
            "**Modelos:** SARIMAX(1,0,0) con $X_t$, Chronos-2 con $X_t$\n\n"
            "**Hipótesis:** Con $\\beta=0.2$ el efecto exógeno es débil; "
            "comparando con 3.1 se cuantifica cuán sensibles son los modelos al tamaño del efecto exógeno."
        ),
        "ARIMAX débil (β=0.2)",
    ),
    (
        "3.3",
        "ARIMAX2Cov_DGP",
        "seed=SEED",
        "dict(phi=0.6, beta1=0.8, beta2=0.4, sigma_y=1.0, sigma_x=1.0, rho_x=0.7)",
        (
            "lambda T: [\n"
            "    SARIMAXModel((1, 0, 0), name_suffix='2 cov.'),\n"
            "    chronos_cov2,\n"
            "]"
        ),
        (
            "**DGP:** ARIMAX con dos covariables independientes\n\n"
            "$$Y_t = 0.6\\,Y_{t-1} + 0.8\\,X_{1t} + 0.4\\,X_{2t} + \\varepsilon_t$$\n\n"
            "$$X_{i,t} = 0.7\\,X_{i,t-1} + \\eta_{i,t}, \\quad i=1,2$$\n\n"
            "**Modelos:** SARIMAX (con $X_1, X_2$), Chronos-2 con $(X_1, X_2)$\n\n"
            "**Hipótesis:** SARIMAX especifica explícitamente los regresores; "
            "Chronos-2 debería explotar ambas señales sin especificación paramétrica."
        ),
        "ARIMAX 2 covariables",
    ),
    (
        "3.5",
        "ARIMAX_GARCH_DGP",
        "seed=SEED",
        "dict(phi=0.4, beta_mean=0.5, omega=0.1, alpha=0.1, beta_garch=0.75, delta_var=0.1, sigma_x=1.0, rho_x=0.7)",
        (
            "lambda T: [\n"
            "    SARIMAXModel((1, 0, 0), name_suffix='con X'),\n"
            "    chronos_cov1,\n"
            "]"
        ),
        (
            "**DGP:** ARIMAX–GARCH — covariable en media y varianza\n\n"
            "Media: $Y_t = 0.4\\,Y_{t-1} + 0.5\\,X_t + \\varepsilon_t$\n\n"
            "Varianza: $\\sigma_t^2 = 0.1 + 0.1\\,\\varepsilon_{t-1}^2 + 0.75\\,\\sigma_{t-1}^2 "
            "+ 0.1\\,X_t^2$\n\n"
            "**Modelos:** SARIMAX (media con $X_t$), Chronos-2 con $X_t$\n\n"
            "**Hipótesis:** SARIMAX captura el efecto de $X_t$ en la media pero ignora su impacto en la varianza. "
            "Chronos-con-X puede adaptar implícitamente la incertidumbre al nivel de $X_t$."
        ),
        "ARIMAX–GARCH",
    ),
    (
        "3.6",
        "ADL_ECM_DGP",
        "seed=SEED",
        "dict(alpha_ecm=-0.3, sigma=1.0, sigma_x=1.0)",
        (
            "lambda T: [\n"
            "    ARDLModel(),\n"
            "    SARIMAXModel((1, 1, 0), name_suffix='dif. con X'),\n"
            "    SARIMAXModel((1, 0, 0), name_suffix='niv. con X'),\n"
            "    chronos_cov1,\n"
            "]"
        ),
        (
            "**DGP:** ADL-ECM — covariable cointegrada $X_t \\sim I(1)$\n\n"
            "$$X_t = X_{t-1} + u_t, \\quad u_t \\sim N(0, 1)$$\n\n"
            "$$\\Delta Y_t = -0.3\\,(Y_{t-1} - X_{t-1}) + \\Delta X_t + \\eta_t$$\n\n"
            "La combinación $Y_t - X_t \\sim I(0)$ (cointegración con vector $(1,-1)$).\n\n"
            "**Modelos:** ARDL-ECM (estimación correcta del ECM), "
            "SARIMAX en diferencias (pierde la relación de largo plazo), "
            "SARIMAX en niveles (potencialmente espuria), Chronos-2 con $X_t$\n\n"
            "**Hipótesis:** ARDL-ECM domina para $h$ grandes al capturar el mecanismo de corrección. "
            "SARIMAX en diferencias acumula sesgo. "
            "Chronos-2 con $X_t$ provisto puede descubrir implícitamente la tendencia común."
        ),
        "ADL-ECM cointegrado",
    ),
]

EXPS_COV_MV = [
    (
        "3.4",
        "VARX_DGP",
        (
            "seed=SEED, "
            "A=[[0.5, 0.1], [0.1, 0.5]], "
            "gamma=[0.5, 0.3], "
            "Sigma=[[1.0, 0.3], [0.3, 1.0]], "
            "sigma_x=1.0, rho_x=0.7"
        ),
        "{}",
        (
            "lambda T: [\n"
            "    VARMAXModel(1),\n"
            "    chronos_mv_cov1,\n"
            "]"
        ),
        (
            "**DGP:** VARX bivariado — efecto exógeno conjunto\n\n"
            "$$Y_t = A\\,Y_{t-1} + \\gamma\\,X_t + \\varepsilon_t, \\quad "
            "A = \\begin{pmatrix}0.5&0.1\\\\0.1&0.5\\end{pmatrix}, \\quad "
            "\\gamma = (0.5, 0.3)^{\\top}$$\n\n"
            "$$X_t = 0.7\\,X_{t-1} + \\eta_t \\quad (\\text{escalar estacionario})$$\n\n"
            "**Modelos:** VARMAX(1) con $X_t$, Chronos-2 joint con $X_t$\n\n"
            "**Hipótesis:** VARMAX especifica explícitamente el efecto exógeno en el sistema bivariado; "
            "Chronos-2 joint con $X_t$ debería capturarlo implícitamente via sus mecanismos de atención."
        ),
        "VARX bivariado",
        '["Y1", "Y2"]',
    ),
]

cells = [c0, c1, c2]

# ── Univariate covariate experiments ──────────────────────────────────────────
for (exp_id, dgp_cls, dgp_init_kwargs, dgp_params_repr,
     make_fn_src, md_desc, exp_name) in EXPS_COV_UNIV:
    slug = exp_id.replace(".", "_")

    cells.append(md(f"---\n## Experimento {exp_id}\n\n{md_desc}"))

    cells.append(code(
        f"dgp_{slug}         = {dgp_cls}({dgp_init_kwargs})\n"
        f"make_models_{slug} = {make_fn_src}\n"
        f"dgp_params_{slug}  = {dgp_params_repr}\n"
        f"\n"
        f"results_{slug} = run_exp_cov(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    exp_id='{exp_id}',\n"
        f")"
    ))

    cells.append(code(
        f"# Simulación representativa (T=T_LIST[0])\n"
        f"plot_rep_cov(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    T=T_LIST[0],\n"
        f"    title=f\"Exp {exp_id}: {exp_name} — Rep. (T={{T_LIST[0]}}, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Simulación representativa (T=T_LIST[1])\n"
        f"plot_rep_cov(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    T=T_LIST[1],\n"
        f"    title=f\"Exp {exp_id}: {exp_name} — Rep. (T={{T_LIST[1]}}, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Tabla de métricas\n"
        f'print("Tabla de métricas — Exp {exp_id}")\n'
        f"results_table(results_{slug})\n"
        f"\n"
        f"# Gráficos de métricas por horizonte\n"
        f"plot_metrics(\n"
        f"    results_{slug},\n"
        f'    title=f"Exp {exp_id} — {exp_name}",\n'
        f'    metrics=("rmse", "bias")\n'
        f")"
    ))

# ── Multivariate covariate experiment (3.4) ────────────────────────────────────
for (exp_id, dgp_cls, dgp_init_kwargs, dgp_params_repr,
     make_fn_src, md_desc, exp_name, var_names_repr) in EXPS_COV_MV:
    slug = exp_id.replace(".", "_")

    cells.append(md(f"---\n## Experimento {exp_id}\n\n{md_desc}"))

    cells.append(code(
        f"dgp_{slug}         = {dgp_cls}({dgp_init_kwargs})\n"
        f"make_models_{slug} = {make_fn_src}\n"
        f"dgp_params_{slug}  = {dgp_params_repr}\n"
        f"var_names_{slug}   = {var_names_repr}\n"
        f"\n"
        f"results_{slug} = run_exp_mv_cov(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    exp_id='{exp_id}',\n"
        f")"
    ))

    cells.append(code(
        f"# Simulación representativa (T=T_LIST[0])\n"
        f"plot_rep_mv_cov(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    var_names=var_names_{slug},\n"
        f"    T=T_LIST[0],\n"
        f"    title=f\"Exp {exp_id}: {exp_name} — Rep. (T={{T_LIST[0]}}, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Simulación representativa (T=T_LIST[1])\n"
        f"plot_rep_mv_cov(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    var_names=var_names_{slug},\n"
        f"    T=T_LIST[1],\n"
        f"    title=f\"Exp {exp_id}: {exp_name} — Rep. (T={{T_LIST[1]}}, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Tabla de métricas por variable\n"
        f'print("Tabla de métricas — Exp {exp_id}")\n'
        f"results_table_mv(results_{slug}, var_names=var_names_{slug})\n"
        f"\n"
        f"# Gráficos de métricas\n"
        f"plot_metrics_mv(\n"
        f"    results_{slug},\n"
        f"    var_names=var_names_{slug},\n"
        f'    title=f"Exp {exp_id} — {exp_name}",\n'
        f'    metrics=("rmse", "bias")\n'
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
