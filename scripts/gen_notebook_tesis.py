"""
Generate notebooks/tesis_presentacion.ipynb

Notebook de presentación para la tesis: 5 exps univariados,
3 multivariados, 3 covariados — seleccionados por relevancia narrativa.

Carga resultados desde CSVs ya cacheados (sin re-ejecutar Monte Carlo).
Corre 1 réplica representativa para las gráficas de simulación.

Run: python scripts/gen_notebook_tesis.py
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "tesis_presentacion.ipynb"

nb   = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)


# ════════════════════════════════════════════════════════════════════════════
# CELL 0 — Título
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "# Presentación — Experimentos Seleccionados para la Tesis\n\n"
    "**Experimentos:** 5 univariados (1.1, 1.4, 1.7, 1.10, 1.19) · "
    "3 multivariados (2.1, 2.5, 2.7) · 3 covariados (3.1, 3.4, 3.6)  \n"
    "**Setup:** T ∈ {50, 200} | H = 24 | R = 500 | Semilla = 3649  \n"
    "**Métricas:** RMSE, MAE, CRPS (menor = mejor) · COV_80/95 (nominal 0.80/0.95) · WINKLER_95 (menor = mejor)  \n\n"
    "Los resultados se cargan desde CSVs ya cacheados.  \n"
    "Las gráficas de simulación representativa corren 1 réplica fresca (Chronos ~30 s en total).\n\n"
    "---"
))

# ════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports, constantes, paleta, Chronos
# ════════════════════════════════════════════════════════════════════════════
cells.append(code(
    "import os, warnings\n"
    'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"\n'
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "import sys\n"
    "sys.path.insert(0, '..')\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "from IPython.display import display\n"
    "\n"
    "# ── Globales ──────────────────────────────────────────────────────────\n"
    "SEED   = 3649\n"
    "H      = 24\n"
    "R      = 500\n"
    "T_LIST = [50, 200]\n"
    "RESULTS_UNI  = Path('results/univariate')\n"
    "RESULTS_MV   = Path('results/multivariate')\n"
    "RESULTS_COV  = Path('results/covariates')\n"
    "\n"
    "# ── Paleta visual (identidad de la tesis) ─────────────────────────────\n"
    "C_HIST    = '#4C72B0'   # historia / ground truth\n"
    "C_CHRONOS = '#9672B6'   # Chronos (siempre violeta)\n"
    "C_CLASSIC = '#2A9D8F'   # modelos clásicos (verde-azulado)\n"
    "CLASSIC_LS = ['-', '--', '-.', ':']  # linestyles para múltiples clásicos\n"
    "\n"
    "def _is_chronos(name: str) -> bool:\n"
    "    return 'Chronos' in name or 'chronos' in name.lower()\n"
    "\n"
    "plt.rcParams.update({\n"
    "    'figure.dpi': 120,\n"
    "    'font.size': 9,\n"
    "    'axes.facecolor': 'white',\n"
    "    'figure.facecolor': 'white',\n"
    "    'axes.grid': True,\n"
    "    'grid.color': '#e8e8e8',\n"
    "    'grid.linewidth': 0.8,\n"
    "    'axes.spines.top': False,\n"
    "    'axes.spines.right': False,\n"
    "})\n"
    "pd.set_option('display.float_format', '{:.4f}'.format)\n"
    "\n"
    "# ── Chronos (cargado una sola vez) ────────────────────────────────────\n"
    "print('Cargando Chronos-2 (~30 s)...')\n"
    "from mectesis.models.chronos import ChronosModel\n"
    "from mectesis.models.chronos_multivariate import ChronosMultivariateModel, ChronosPerVarModel\n"
    "from mectesis.models.chronos_covariate import ChronosCovariateModel, ChronosMultivariateCovariateModel\n"
    "\n"
    "_chronos_base   = ChronosModel(device='cpu')\n"
    "chronos_uni     = _chronos_base\n"
    "chronos_mv      = ChronosMultivariateModel(_chronos_base)\n"
    "chronos_mv_ind  = ChronosPerVarModel(_chronos_base)\n"
    "chronos_cov1    = ChronosCovariateModel(_chronos_base, n_covariates=1)\n"
    "chronos_mv_cov1 = ChronosMultivariateCovariateModel(_chronos_base, n_covariates=1)\n"
    "print('Chronos-2 listo.')\n"
))

# ════════════════════════════════════════════════════════════════════════════
# CELL 2 — Helpers: load, styled_table, plot_metrics_horizon
# ════════════════════════════════════════════════════════════════════════════
cells.append(code(
    "# ══ Carga de resultados ═════════════════════════════════════════════════\n"
    "\n"
    "def _load(path):\n"
    "    df = pd.read_csv(path)\n"
    "    return {\n"
    "        m: grp.drop(columns='model').reset_index(drop=True)\n"
    "        for m, grp in df.groupby('model', sort=False)\n"
    "    }\n"
    "\n"
    "def _load_mv(path):\n"
    "    df = pd.read_csv(path)\n"
    "    out = {}\n"
    "    for m, mg in df.groupby('model', sort=False):\n"
    "        out[m] = {\n"
    "            int(v): vg.drop(columns=['model','var']).reset_index(drop=True)\n"
    "            for v, vg in mg.groupby('var', sort=True)\n"
    "        }\n"
    "    return out\n"
    "\n"
    "def load_uni(exp_id):\n"
    "    eid = exp_id.replace('.', '_')\n"
    "    return {(T, R): _load(RESULTS_UNI / f'exp_{eid}_T{T}_R{R}.csv') for T in T_LIST}\n"
    "\n"
    "def load_mv(exp_id):\n"
    "    eid = exp_id.replace('.', '_')\n"
    "    return {(T, R): _load_mv(RESULTS_MV / f'exp_{eid}_T{T}_R{R}.csv') for T in T_LIST}\n"
    "\n"
    "def load_cov(exp_id):\n"
    "    eid = exp_id.replace('.', '_')\n"
    "    return {(T, R): _load(RESULTS_COV / f'exp_{eid}_T{T}_R{R}.csv') for T in T_LIST}\n"
    "\n"
    "def load_mv_cov(exp_id):\n"
    "    eid = exp_id.replace('.', '_')\n"
    "    return {(T, R): _load_mv(RESULTS_COV / f'exp_{eid}_T{T}_R{R}.csv') for T in T_LIST}\n"
    "\n"
    "\n"
    "# ══ avg_all extractor ═══════════════════════════════════════════════════\n"
    "\n"
    "def _avg(df):\n"
    "    row = df[df['horizon'] == 'avg_all']\n"
    "    if row.empty:\n"
    "        return df.select_dtypes('number').mean()\n"
    "    return row.iloc[0]\n"
    "\n"
    "\n"
    "# ══ Tabla estilizada ════════════════════════════════════════════════════\n"
    "\n"
    "_MCOLS = ['rmse','mae','crps','cov_80','cov_95','width_95','winkler_95']\n"
    "_ERR   = ['rmse','mae','crps','winkler_95','width_95']\n"
    "_COV   = ['cov_80','cov_95']\n"
    "\n"
    "def styled_table(all_results, title='', is_mv=False):\n"
    "    rows = []\n"
    "    for (T, Rr), res_TR in sorted(all_results.items()):\n"
    "        if is_mv:\n"
    "            for mname, var_dict in res_TR.items():\n"
    "                avgs = pd.DataFrame([_avg(df) for df in var_dict.values()])\n"
    "                combined = avgs.mean(numeric_only=True)\n"
    "                row = {'T': T, 'Modelo': mname}\n"
    "                for c in _MCOLS:\n"
    "                    if c in combined.index:\n"
    "                        row[c] = round(float(combined[c]), 4)\n"
    "                rows.append(row)\n"
    "        else:\n"
    "            for mname, df in res_TR.items():\n"
    "                avg = _avg(df)\n"
    "                row = {'T': T, 'Modelo': mname}\n"
    "                for c in _MCOLS:\n"
    "                    if c in avg.index:\n"
    "                        try:    row[c] = round(float(avg[c]), 4)\n"
    "                        except: row[c] = float('nan')\n"
    "                rows.append(row)\n"
    "\n"
    "    df_out = pd.DataFrame(rows).set_index(['T', 'Modelo'])\n"
    "    err_p = [c for c in _ERR if c in df_out.columns]\n"
    "    cov_p = [c for c in _COV if c in df_out.columns]\n"
    "    sty = df_out.style.format(precision=4, na_rep='—')\n"
    "    if err_p: sty = sty.background_gradient(subset=err_p, cmap='RdYlGn_r')\n"
    "    if cov_p: sty = sty.background_gradient(subset=cov_p, cmap='RdYlGn')\n"
    "    if title: sty = sty.set_caption(title)\n"
    "    display(sty)\n"
    "    return df_out\n"
    "\n"
    "\n"
    "# ══ Métricas vs horizonte ════════════════════════════════════════════════\n"
    "\n"
    "def plot_metrics_horizon(all_results,\n"
    "                         metrics=('rmse','crps','cov_95','winkler_95'),\n"
    "                         title='', is_mv=False):\n"
    "    keys = sorted(all_results.keys())\n"
    "    fig, axes = plt.subplots(len(metrics), len(keys),\n"
    "                             figsize=(6.5*len(keys), 3.0*len(metrics)),\n"
    "                             squeeze=False)\n"
    "    for col, (T, Rr) in enumerate(keys):\n"
    "        res_TR = all_results[(T, Rr)]\n"
    "        # ── Construir series por modelo ──────────────────────────────\n"
    "        if is_mv:\n"
    "            model_dfs = {}\n"
    "            for mname, var_dict in res_TR.items():\n"
    "                dfs = [df[df['horizon'] != 'avg_all'].copy() for df in var_dict.values()]\n"
    "                for d in dfs: d['horizon'] = pd.to_numeric(d['horizon'])\n"
    "                model_dfs[mname] = (pd.concat(dfs)\n"
    "                                      .groupby('horizon').mean(numeric_only=True)\n"
    "                                      .reset_index())\n"
    "        else:\n"
    "            model_dfs = {}\n"
    "            for mname, df in res_TR.items():\n"
    "                d = df[df['horizon'] != 'avg_all'].copy()\n"
    "                d['horizon'] = pd.to_numeric(d['horizon'])\n"
    "                model_dfs[mname] = d\n"
    "\n"
    "        # ── Asignar estilos ──────────────────────────────────────────\n"
    "        classic_idx = 0\n"
    "        ls_map = {}\n"
    "        for m in model_dfs:\n"
    "            if _is_chronos(m):\n"
    "                ls_map[m] = ('-', C_CHRONOS)\n"
    "            else:\n"
    "                ls_map[m] = (CLASSIC_LS[classic_idx % len(CLASSIC_LS)], C_CLASSIC)\n"
    "                classic_idx += 1\n"
    "\n"
    "        for row, metric in enumerate(metrics):\n"
    "            ax = axes[row][col]\n"
    "            ax.axvline(12.5, color='#999999', ls=':', lw=0.9, alpha=0.7)\n"
    "            for mname, df_h in model_dfs.items():\n"
    "                if metric not in df_h.columns: continue\n"
    "                ls, color = ls_map[mname]\n"
    "                ax.plot(df_h['horizon'], df_h[metric],\n"
    "                        label=mname, color=color, ls=ls, lw=1.5, alpha=0.9)\n"
    "            ax.set_title(f'T={T} — {metric.upper()}', fontsize=9.5, pad=3)\n"
    "            ax.set_xlabel('Horizonte h', fontsize=8.5)\n"
    "            ax.set_ylabel(metric.upper(), fontsize=8.5)\n"
    "            ax.legend(fontsize=7.5)\n"
    "\n"
    "    if title: fig.suptitle(title, fontsize=11, y=1.01)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "    return fig\n"
))

# ════════════════════════════════════════════════════════════════════════════
# CELL 3 — Funciones de gráfica de simulación representativa (Fig 12/14 style)
# ════════════════════════════════════════════════════════════════════════════
cells.append(code(
    "# ══ Helpers internos ════════════════════════════════════════════════════\n"
    "\n"
    "def _assign_styles(model_names):\n"
    "    \"\"\"Retorna {name: (color, linestyle)} respetando la paleta de la tesis.\"\"\"\n"
    "    classic_idx = 0\n"
    "    styles = {}\n"
    "    for n in model_names:\n"
    "        if _is_chronos(n):\n"
    "            styles[n] = (C_CHRONOS, '-')\n"
    "        else:\n"
    "            styles[n] = (C_CLASSIC, CLASSIC_LS[classic_idx % len(CLASSIC_LS)])\n"
    "            classic_idx += 1\n"
    "    return styles\n"
    "\n"
    "def _get_name(m):\n"
    "    return m.name if hasattr(m, 'name') else str(type(m).__name__)\n"
    "\n"
    "def _draw_context(ax, x_tr, y_tr_1d, x_te, y_te_1d, n_context):\n"
    "    \"\"\"Dibuja los últimos n_context puntos del histórico + ground truth futuro.\"\"\"\n"
    "    ax.plot(x_tr[-n_context:], y_tr_1d[-n_context:],\n"
    "            color=C_HIST, lw=1.3, alpha=0.85, label='Histórico')\n"
    "    ax.plot(x_te, y_te_1d,\n"
    "            color=C_HIST, lw=1.3, ls='--', alpha=0.6, label='Observado')\n"
    "    ax.axvline(x_tr[-1] + 0.5, color='#999999', ls=':', lw=0.9, alpha=0.7)\n"
    "\n"
    "\n"
    "# ══ Univariado ══════════════════════════════════════════════════════════\n"
    "\n"
    "def plot_rep_uni(dgp, make_models, dgp_params, T_list=T_LIST,\n"
    "                 H=H, seed=SEED, title='', n_context=50):\n"
    "    \"\"\"\n"
    "    Figura 12/14 style: filas = modelos, columnas = T.\n"
    "    Intervalo de predicción al 80% sombreado.\n"
    "    \"\"\"\n"
    "    n_models = len(make_models(T_list[0]))\n"
    "    fig, axes = plt.subplots(n_models, len(T_list),\n"
    "                             figsize=(12, 3.5 * n_models), squeeze=False)\n"
    "    fig.patch.set_facecolor('white')\n"
    "\n"
    "    for col, T in enumerate(T_list):\n"
    "        dgp.rng = np.random.default_rng(seed)\n"
    "        y = dgp.simulate(T=T, **dgp_params)\n"
    "        y_train, y_test = y[:-H], y[-H:]\n"
    "        x_tr = np.arange(len(y_train))\n"
    "        x_te = np.arange(len(y_train), T)\n"
    "        models  = make_models(T)\n"
    "        mnames  = [_get_name(m) for m in models]\n"
    "        styles  = _assign_styles(mnames)\n"
    "\n"
    "        for row, m in enumerate(models):\n"
    "            ax = axes[row][col]\n"
    "            m.fit(y_train)\n"
    "            y_hat = m.forecast(H)\n"
    "            mname = mnames[row]\n"
    "            color, ls = styles[mname]\n"
    "            rmse_v = float(np.sqrt(np.nanmean((y_hat - y_test)**2)))\n"
    "            _draw_context(ax, x_tr, y_train, x_te, y_test, n_context)\n"
    "            ax.plot(x_te, y_hat, color=color, ls=ls, lw=1.5, label=mname)\n"
    "            if getattr(m, 'supports_intervals', False):\n"
    "                lo, hi = m.forecast_intervals(H, level=0.80)\n"
    "                ax.fill_between(x_te, lo, hi, color=color, alpha=0.15)\n"
    "            ax.set_title(f'{mname}  |  T={T}    RMSE={rmse_v:.3f}', fontsize=9)\n"
    "            ax.set_xlabel('t', fontsize=8)\n"
    "            ax.legend(fontsize=7.5, loc='upper left')\n"
    "\n"
    "    if title: fig.suptitle(title, fontsize=11, y=1.01)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "    return fig\n"
    "\n"
    "\n"
    "# ══ Multivariado ════════════════════════════════════════════════════════\n"
    "\n"
    "def plot_rep_mv(dgp, make_models, T_list=T_LIST,\n"
    "                H=H, seed=SEED, title='', var_idx=0, n_context=50):\n"
    "    \"\"\"\n"
    "    Multivariado: DGP.simulate() → (T, k). Muestra variable var_idx.\n"
    "    \"\"\"\n"
    "    n_models = len(make_models(T_list[0]))\n"
    "    fig, axes = plt.subplots(n_models, len(T_list),\n"
    "                             figsize=(12, 3.5 * n_models), squeeze=False)\n"
    "    fig.patch.set_facecolor('white')\n"
    "\n"
    "    for col, T in enumerate(T_list):\n"
    "        dgp.rng = np.random.default_rng(seed)\n"
    "        y = dgp.simulate(T=T)\n"
    "        y_train, y_test = y[:-H], y[-H:]\n"
    "        x_tr = np.arange(len(y_train))\n"
    "        x_te = np.arange(len(y_train), T)\n"
    "        y_tr_v = y_train[:, var_idx]\n"
    "        y_te_v = y_test[:, var_idx]\n"
    "        models  = make_models(T)\n"
    "        mnames  = [_get_name(m) for m in models]\n"
    "        styles  = _assign_styles(mnames)\n"
    "\n"
    "        for row, m in enumerate(models):\n"
    "            ax = axes[row][col]\n"
    "            m.fit(y_train)\n"
    "            y_hat_full = m.forecast(H)  # (H, k)\n"
    "            y_hat = y_hat_full[:, var_idx]\n"
    "            mname = mnames[row]\n"
    "            color, ls = styles[mname]\n"
    "            rmse_v = float(np.sqrt(np.nanmean((y_hat - y_te_v)**2)))\n"
    "            _draw_context(ax, x_tr, y_tr_v, x_te, y_te_v, n_context)\n"
    "            ax.plot(x_te, y_hat, color=color, ls=ls, lw=1.5, label=mname)\n"
    "            if getattr(m, 'supports_intervals', False):\n"
    "                lo_f, hi_f = m.forecast_intervals(H, level=0.80)  # (H, k)\n"
    "                ax.fill_between(x_te, lo_f[:, var_idx], hi_f[:, var_idx],\n"
    "                                color=color, alpha=0.15)\n"
    "            ax.set_title(f'{mname}  |  T={T}    RMSE Y{var_idx}={rmse_v:.3f}', fontsize=9)\n"
    "            ax.set_xlabel('t', fontsize=8)\n"
    "            ax.legend(fontsize=7.5, loc='upper left')\n"
    "\n"
    "    extra = f'  (variable Y{var_idx} mostrada)'\n"
    "    if title: fig.suptitle(title + extra, fontsize=11, y=1.01)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "    return fig\n"
    "\n"
    "\n"
    "# ══ Covariado ═══════════════════════════════════════════════════════════\n"
    "\n"
    "def plot_rep_cov(dgp, make_models, dgp_params, T_list=T_LIST,\n"
    "                 H=H, seed=SEED, title='', is_mv=False, var_idx=0, n_context=50):\n"
    "    \"\"\"\n"
    "    Covariado: DGP.simulate() → {'y': ..., 'X': ...}.\n"
    "    is_mv=True si y es (T,k); muestra variable var_idx.\n"
    "    \"\"\"\n"
    "    n_models = len(make_models(T_list[0]))\n"
    "    fig, axes = plt.subplots(n_models, len(T_list),\n"
    "                             figsize=(12, 3.5 * n_models), squeeze=False)\n"
    "    fig.patch.set_facecolor('white')\n"
    "\n"
    "    for col, T in enumerate(T_list):\n"
    "        dgp.rng = np.random.default_rng(seed)\n"
    "        data    = dgp.simulate(T=T, **dgp_params)\n"
    "        y_full  = data['y']\n"
    "        X_full  = data['X']\n"
    "        y_train_f, y_test_f = y_full[:-H], y_full[-H:]\n"
    "        X_train,   X_future = X_full[:-H], X_full[-H:]\n"
    "\n"
    "        if is_mv:\n"
    "            y_tr_v = y_train_f[:, var_idx]\n"
    "            y_te_v = y_test_f[:, var_idx]\n"
    "        else:\n"
    "            y_tr_v = y_train_f\n"
    "            y_te_v = y_test_f\n"
    "\n"
    "        x_tr = np.arange(len(y_tr_v))\n"
    "        x_te = np.arange(len(y_tr_v), T)\n"
    "        models  = make_models(T)\n"
    "        mnames  = [_get_name(m) for m in models]\n"
    "        styles  = _assign_styles(mnames)\n"
    "\n"
    "        for row, m in enumerate(models):\n"
    "            ax = axes[row][col]\n"
    "            m.fit(y_train_f, X_train)\n"
    "            y_hat_full = m.forecast(H, X_future=X_future)\n"
    "            if is_mv and np.ndim(y_hat_full) > 1:\n"
    "                y_hat = y_hat_full[:, var_idx]\n"
    "            else:\n"
    "                y_hat = np.asarray(y_hat_full).ravel()\n"
    "            mname  = mnames[row]\n"
    "            color, ls = styles[mname]\n"
    "            valid  = ~np.isnan(y_hat)\n"
    "            rmse_v = (float(np.sqrt(np.mean((y_hat[valid] - y_te_v[valid])**2)))\n"
    "                      if valid.any() else float('nan'))\n"
    "            _draw_context(ax, x_tr, y_tr_v, x_te, y_te_v, n_context)\n"
    "            ax.plot(x_te, y_hat, color=color, ls=ls, lw=1.5, label=mname)\n"
    "            if getattr(m, 'supports_intervals', False):\n"
    "                lo_f, hi_f = m.forecast_intervals(H, level=0.80, X_future=X_future)\n"
    "                if is_mv and np.ndim(lo_f) > 1:\n"
    "                    lo_f, hi_f = lo_f[:, var_idx], hi_f[:, var_idx]\n"
    "                ax.fill_between(x_te, lo_f, hi_f, color=color, alpha=0.15)\n"
    "            ax.set_title(f'{mname}  |  T={T}    RMSE={rmse_v:.3f}', fontsize=9)\n"
    "            ax.set_xlabel('t', fontsize=8)\n"
    "            ax.legend(fontsize=7.5, loc='upper left')\n"
    "\n"
    "    extra = f'  (Y{var_idx})' if is_mv else ''\n"
    "    if title: fig.suptitle(title + extra, fontsize=11, y=1.01)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "    return fig\n"
))

# ════════════════════════════════════════════════════════════════════════════
# CELL 4 — Imports de DGPs y modelos clásicos para réplicas representativas
# ════════════════════════════════════════════════════════════════════════════
cells.append(code(
    "# ── DGPs univariados ──────────────────────────────────────────────────\n"
    "from mectesis.dgp.ar       import AR1\n"
    "from mectesis.dgp.rw       import RandomWalk\n"
    "from mectesis.dgp.seasonal import SeasonalDGP\n"
    "from mectesis.dgp.garch    import AR1GARCH\n"
    "from mectesis.dgp.ar_trend import AR1WithTrend\n"
    "\n"
    "# ── DGPs multivariados ────────────────────────────────────────────────\n"
    "from mectesis.dgp.var_dgp  import VARDGP\n"
    "from mectesis.dgp.vecm_dgp import VECMBivariateDGP\n"
    "\n"
    "# ── DGPs covariados ───────────────────────────────────────────────────\n"
    "from mectesis.dgp.arimax_dgp  import ARIMAX_DGP\n"
    "from mectesis.dgp.varx_dgp    import VARX_DGP\n"
    "from mectesis.dgp.adl_ecm_dgp import ADL_ECM_DGP\n"
    "\n"
    "# ── Modelos clásicos ──────────────────────────────────────────────────\n"
    "from mectesis.models.arima          import ARIMAModel\n"
    "from mectesis.models.sarima_model   import SARIMAModel\n"
    "from mectesis.models.garch_model    import ARGARCHModel\n"
    "from mectesis.models.theta_model    import ThetaModel\n"
    "from mectesis.models.var_model      import VARModel, VECMModel\n"
    "from mectesis.models.sarimax_model  import SARIMAXModel\n"
    "from mectesis.models.varmax_model   import VARMAXModel\n"
    "from mectesis.models.ardl_model     import ARDLModel\n"
    "\n"
    "print('Imports OK.')\n"
))

# ════════════════════════════════════════════════════════════════════════════
# CELL 5 — Instancias DGP para las réplicas representativas
# ════════════════════════════════════════════════════════════════════════════
cells.append(code(
    "# ── Univariados ───────────────────────────────────────────────────────\n"
    "dgp_1_1  = AR1(seed=SEED)\n"
    "dgp_1_4  = RandomWalk(seed=SEED)\n"
    "dgp_1_7  = SeasonalDGP(seed=SEED)\n"
    "dgp_1_10 = AR1GARCH(seed=SEED)\n"
    "dgp_1_19 = AR1WithTrend(seed=SEED)\n"
    "\n"
    "# ── Multivariados ─────────────────────────────────────────────────────\n"
    "_A1_21  = np.array([[0.5, 0.1], [0.1, 0.5]])\n"
    "_Sig2   = np.array([[1.0, 0.3], [0.3, 1.0]])\n"
    "dgp_2_1  = VARDGP(seed=SEED, A_list=[_A1_21], Sigma=_Sig2)\n"
    "\n"
    "_A1_25  = np.array([[0.3, 0.05, 0.0,  0.0,  0.0 ],\n"
    "                    [0.05, 0.3, 0.05, 0.0,  0.0 ],\n"
    "                    [0.0, 0.05, 0.3,  0.05, 0.0 ],\n"
    "                    [0.0,  0.0, 0.05, 0.3,  0.05],\n"
    "                    [0.0,  0.0, 0.0,  0.05, 0.3 ]])\n"
    "_Sig5   = np.array([[1.0, 0.2, 0.0, 0.0, 0.0],\n"
    "                    [0.2, 1.0, 0.2, 0.0, 0.0],\n"
    "                    [0.0, 0.2, 1.0, 0.2, 0.0],\n"
    "                    [0.0, 0.0, 0.2, 1.0, 0.2],\n"
    "                    [0.0, 0.0, 0.0, 0.2, 1.0]])\n"
    "dgp_2_5  = VARDGP(seed=SEED, A_list=[_A1_25], Sigma=_Sig5)\n"
    "\n"
    "dgp_2_7  = VECMBivariateDGP(seed=SEED)  # parámetros por defecto del exp 2.7\n"
    "\n"
    "# ── Covariados ────────────────────────────────────────────────────────\n"
    "dgp_3_1  = ARIMAX_DGP(seed=SEED)\n"
    "dgp_3_4  = VARX_DGP(\n"
    "    seed=SEED,\n"
    "    A=[[0.5, 0.1], [0.1, 0.5]],\n"
    "    gamma=[0.5, 0.3],\n"
    "    Sigma=[[1.0, 0.3], [0.3, 1.0]],\n"
    "    sigma_x=1.0, rho_x=0.7,\n"
    ")\n"
    "dgp_3_6  = ADL_ECM_DGP(seed=SEED)\n"
    "\n"
    "print('DGPs creados.')\n"
))


# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN UNIVARIADOS ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "---\n\n"
    "# Experimentos Univariados\n\n"
    "Seleccionados: **1.1** (baseline) · **1.4** (misspecificación) · "
    "**1.7** (límite estructural) · **1.10** (brecha CRPS) · **1.19** (colapso clásico)  \n\n"
    "Modelos clásicos correctamente especificados vs Chronos-2 (zero-shot)."
))

# ── Exp 1.1 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 1.1 — AR(1) φ=0.3, baja persistencia\n\n"
    "**DGP:** $Y_t = 0.3\\,Y_{t-1} + \\varepsilon_t, \\quad \\varepsilon_t \\sim \\mathcal{N}(0,1)$\n\n"
    "**Caso base.** ARIMA correctamente especificado vs Chronos zero-shot. "
    "Evalúa si Chronos recupera la estructura AR(1) estacionaria sin estimarla explícitamente.\n\n"
    "**Hipótesis:** ARIMA domina en RMSE y CRPS. Chronos sobrecobertura (~2× ancho de intervalo)."
))
cells.append(code(
    "res_1_1 = load_uni('1.1')\n"
    "styled_table(res_1_1, title='Exp 1.1 — AR(1) φ=0.3')\n"
    "plot_metrics_horizon(res_1_1, title='Exp 1.1 — AR(1) φ=0.3')\n"
))
cells.append(code(
    "make_models_1_1 = lambda T: [ARIMAModel((1,0,0)), chronos_uni]\n"
    "plot_rep_uni(dgp_1_1, make_models_1_1, {'phi': 0.3},\n"
    "             title='Exp 1.1 — AR(1) φ=0.3')\n"
))

# ── Exp 1.4 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 1.4 — Random Walk con drift=0.5 *(inversión: Chronos gana)*\n\n"
    "**DGP:** $Y_t = 0.5 + Y_{t-1} + \\varepsilon_t, \\quad \\varepsilon_t \\sim \\mathcal{N}(0,1)$\n\n"
    "**El resultado más relevante del bloque.** ARIMA(0,1,0) sin constante no captura el drift: "
    "el sesgo crece linealmente con el horizonte ($\\text{bias} \\approx 0.5h$). "
    "Chronos identifica el drift desde el contexto de la serie (zero-shot). "
    "La ventaja de Chronos se amplía con T (RMSE cae 32% de T=50 a T=200).\n\n"
    "**Nota:** La misspecificación del drift es un error frecuente en la práctica.\n\n"
    "**Hipótesis:** Chronos domina en RMSE, MAE y Winkler en ambos T."
))
cells.append(code(
    "res_1_4 = load_uni('1.4')\n"
    "styled_table(res_1_4, title='Exp 1.4 — RW drift=0.5')\n"
    "plot_metrics_horizon(res_1_4, title='Exp 1.4 — RW drift=0.5')\n"
))
cells.append(code(
    "make_models_1_4 = lambda T: [ARIMAModel((0,1,0)), chronos_uni]\n"
    "plot_rep_uni(dgp_1_4, make_models_1_4, {'drift': 0.5},\n"
    "             title='Exp 1.4 — RW drift=0.5 (ARIMA sin constante)')\n"
))

# ── Exp 1.7 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 1.7 — Seasonal I(1)×I(1)₁₂ *(limitación estructural de Chronos)*\n\n"
    "**DGP:** $(1-L)(1-L^{12})\\,Y_t = \\varepsilon_t$\n\n"
    "**El peor resultado de Chronos en el bloque.** SARIMA domina +40–45% en RMSE, "
    "brecha **estable** con T (no se cierra con más contexto). "
    "La doble integración estacional es la limitación estructural más clara del foundation model: "
    "no maneja no-estacionariedad de largo alcance en dos dimensiones.\n\n"
    "**Hipótesis:** SARIMA domina todas las métricas; Chronos no mejora con T=200."
))
cells.append(code(
    "res_1_7 = load_uni('1.7')\n"
    "styled_table(res_1_7, title='Exp 1.7 — Seasonal I(1)×I(1)₁₂')\n"
    "plot_metrics_horizon(res_1_7, title='Exp 1.7 — Seasonal I(1)×I(1)₁₂')\n"
))
cells.append(code(
    "make_models_1_7 = lambda T: [\n"
    "    SARIMAModel((0,1,0), (0,1,0,12)),\n"
    "    chronos_uni,\n"
    "]\n"
    "plot_rep_uni(dgp_1_7, make_models_1_7, {'s': 12, 'integrated': True},\n"
    "             title='Exp 1.7 — Seasonal I(1)×I(1)₁₂')\n"
))

# ── Exp 1.10 ─────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 1.10 — AR(1)–GARCH(1,1), α+β=0.9 *(RMSE ≠ CRPS)*\n\n"
    "**DGP:** $Y_t = 0.3\\,Y_{t-1} + \\varepsilon_t, \\quad "
    "\\sigma_t^2 = 0.1 + 0.1\\,\\varepsilon_{t-1}^2 + 0.8\\,\\sigma_{t-1}^2$\n\n"
    "**Representa el bloque GARCH (1.9–1.12).** A T=200: paridad casi perfecta en RMSE (~1%) "
    "pero la brecha en CRPS persiste (~23%). "
    "Chronos produce distribuciones predictivas más anchas de lo necesario aun cuando el "
    "error puntual converge. La dinámica GARCH afecta la varianza condicional, no la media.\n\n"
    "**Hipótesis:** Empate en RMSE a T=200; ventaja persistente del modelo clásico en CRPS y calibración."
))
cells.append(code(
    "res_1_10 = load_uni('1.10')\n"
    "styled_table(res_1_10, title='Exp 1.10 — AR(1)+GARCH(1,1)')\n"
    "plot_metrics_horizon(res_1_10, title='Exp 1.10 — AR(1)+GARCH(1,1)')\n"
))
cells.append(code(
    "make_models_1_10 = lambda T: [ARGARCHModel(), chronos_uni]\n"
    "plot_rep_uni(dgp_1_10, make_models_1_10,\n"
    "             {'phi': 0.3, 'omega': 0.1, 'alpha': 0.1, 'beta': 0.8},\n"
    "             title='Exp 1.10 — AR(1)+GARCH(1,1)')\n"
))

# ── Exp 1.19 ─────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 1.19 — Tendencia lineal / Theta *(colapso catastrófico del modelo clásico)*\n\n"
    "**DGP:** $Y_t = 0.1\\,t + \\varepsilon_t, \\quad \\varepsilon_t \\sim \\mathcal{N}(0,1)$\n\n"
    "**La mayor inversión del bloque.** A T=50: Theta domina (26% mejor RMSE). "
    "A T=200: **Theta colapsa** — RMSE=7.89, bias=−7.63, cov_95=0.640. "
    "Chronos domina por factor 7.7× en RMSE. "
    "El mecanismo: el optimizador de Theta converge a la media histórica (~10) "
    "ignorando que el nivel actual es ~20. Complementa el mensaje de Exp 1.4: "
    "los modelos clásicos también pueden fallar estructuralmente.\n\n"
    "**Hipótesis:** Inversión dramática entre T=50 y T=200. Chronos aprende el slope con suficiente historia."
))
cells.append(code(
    "res_1_19 = load_uni('1.19')\n"
    "styled_table(res_1_19, title='Exp 1.19 — Tendencia lineal (Theta)')\n"
    "plot_metrics_horizon(res_1_19, metrics=('rmse','bias','cov_95','winkler_95'),\n"
    "                     title='Exp 1.19 — Tendencia lineal (Theta)')\n"
))
cells.append(code(
    "make_models_1_19 = lambda T: [ThetaModel(), chronos_uni]\n"
    "plot_rep_uni(dgp_1_19, make_models_1_19,\n"
    "             {'intercept': 0.0, 'trend_coeff': 0.1, 'phi': 0.0},\n"
    "             title='Exp 1.19 — Tendencia lineal (Theta)')\n"
))


# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN MULTIVARIADOS ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "---\n\n"
    "# Experimentos Multivariados\n\n"
    "Seleccionados: **2.1** (baseline k=2) · **2.5** (paridad RMSE k=5) · "
    "**2.7** (VECM cointegrado)  \n\n"
    "Métricas reportadas como promedio sobre las k variables y los H=24 horizontes.  \n"
    "Gráficas de simulación muestran la variable Y₀."
))

# ── Exp 2.1 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 2.1 — VAR(1) bivariado, baja interdependencia (k=2)\n\n"
    "**DGP:** $Y_t = A_1 Y_{t-1} + \\varepsilon_t, \\quad "
    "A_1 = \\begin{pmatrix}0.5&0.1\\\\0.1&0.5\\end{pmatrix}, \\quad "
    "\\Sigma = \\begin{pmatrix}1.0&0.3\\\\0.3&1.0\\end{pmatrix}$\n\n"
    "**Caso base multivariado.** VAR domina en RMSE (+7% T=50, +1% T=200) y CRPS (+73%/+23%). "
    "Chronos-joint y Chronos-ind son prácticamente iguales: con baja interdependencia, "
    "la API conjunta no aporta sobre k=2 forecasts independientes.\n\n"
    "**Hipótesis:** VAR domina en todas las métricas; Chronos-joint ≈ Chronos-ind."
))
cells.append(code(
    "res_2_1 = load_mv('2.1')\n"
    "styled_table(res_2_1, title='Exp 2.1 — VAR(1) bivariado k=2', is_mv=True)\n"
    "plot_metrics_horizon(res_2_1, title='Exp 2.1 — VAR(1) bivariado k=2', is_mv=True)\n"
))
cells.append(code(
    "make_models_2_1 = lambda T: [VARModel(1), chronos_mv, chronos_mv_ind]\n"
    "plot_rep_mv(dgp_2_1, make_models_2_1, title='Exp 2.1 — VAR(1) k=2')\n"
))

# ── Exp 2.5 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 2.5 — VAR(1) pentavariado, k=5 *(paridad en RMSE)*\n\n"
    "**DGP:** $Y_t = A_1 Y_{t-1} + \\varepsilon_t$, $k=5$, "
    "$A_1$ tridiagonal (diagonal 0.3, off-diagonal 0.05), "
    "$\\Sigma$ tridiagonal (diagonal 1.0, off-diagonal 0.2)\n\n"
    "**El resultado más sorprendente del bloque.** RMSE en paridad a **ambos** T "
    "(+0.3% T=50, +0.5% T=200): la maldición de dimensionalidad no se manifiesta en RMSE "
    "con coeficientes cruzados débiles. "
    "Sin embargo, la brecha en CRPS persiste (+40% T=50, +21% T=200): "
    "incluso cuando el nivel de error converge, Chronos produce distribuciones más anchas.\n\n"
    "**Hipótesis:** Empate RMSE; VAR con ventaja sistemática en CRPS."
))
cells.append(code(
    "res_2_5 = load_mv('2.5')\n"
    "styled_table(res_2_5, title='Exp 2.5 — VAR(1) pentavariado k=5', is_mv=True)\n"
    "plot_metrics_horizon(res_2_5, title='Exp 2.5 — VAR(1) pentavariado k=5', is_mv=True)\n"
))
cells.append(code(
    "make_models_2_5 = lambda T: [VARModel(1), chronos_mv]\n"
    "plot_rep_mv(dgp_2_5, make_models_2_5, title='Exp 2.5 — VAR(1) k=5')\n"
))

# ── Exp 2.7 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 2.7 — VECM bivariado cointegrado (k=2, rango 1)\n\n"
    "**DGP:** $(1-L)Y_t = \\alpha(\\beta' Y_{t-1}) + \\Gamma_1 (1-L)Y_{t-1} + \\varepsilon_t$, "
    "$\\beta=(1,-1)'$, $\\alpha=(-0.4,0.2)'$\n\n"
    "**El experimento más dramático del bloque.** VAR(1) colapsa a T=50 (+76% RMSE): "
    "aplica VAR en niveles a datos I(1) cointegrados sin restricción de rango → regresión espuria. "
    "A T=200, VAR se recupera parcialmente (+6%). "
    "Chronos-joint supera a VAR(1) en RMSE a T=200 ($-0.8\\%$): "
    "**única victoria de Chronos en RMSE** en el bloque, posible por la misspecificación del benchmark. "
    "VECM correctamente especificado domina en CRPS en ambos T.\n\n"
    "**Replica la lección de Exp 1.4:** Chronos > modelo clásico *misspecificado*, "
    "pero no > modelo *correctamente especificado*."
))
cells.append(code(
    "res_2_7 = load_mv('2.7')\n"
    "styled_table(res_2_7, title='Exp 2.7 — VECM cointegrado k=2', is_mv=True)\n"
    "plot_metrics_horizon(res_2_7, title='Exp 2.7 — VECM cointegrado k=2', is_mv=True)\n"
))
cells.append(code(
    "make_models_2_7 = lambda T: [\n"
    "    VECMModel(coint_rank=1),\n"
    "    VARModel(1),\n"
    "    chronos_mv,\n"
    "    chronos_mv_ind,\n"
    "]\n"
    "plot_rep_mv(dgp_2_7, make_models_2_7, title='Exp 2.7 — VECM cointegrado k=2')\n"
))


# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN COVARIADOS ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "---\n\n"
    "# Experimentos con Covariables\n\n"
    "Seleccionados: **3.1** (inversión β fuerte) · **3.4** (VARX sin inversión) · "
    "**3.6** (ADL-ECM cointegrado)  \n\n"
    "**Supuesto oráculo:** los valores futuros de X se proveen a los modelos — "
    "se evalúa la utilidad del covariante dado que su futuro es observable."
))

# ── Exp 3.1 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 3.1 — ARIMAX AR(1) con covariante fuerte (β=0.8) *(inversión a T=200)*\n\n"
    "**DGP:** $Y_t = 0.6\\,Y_{t-1} + 0.8\\,X_t + \\varepsilon_t$, "
    "$X_t = 0.7\\,X_{t-1} + \\eta_t$\n\n"
    "**La inversión más limpia del bloque.** A T=50: SARIMAX domina (+19% RMSE). "
    "A T=200: **Chronos gana** (−24% RMSE, −10% CRPS, mejor Winkler). "
    "Mecanismo: Chronos aprende implícitamente la relación entre covariante y target "
    "desde el contexto cuando la señal es suficientemente fuerte. "
    "SARIMAX tiene un techo de mejora limitado por los errores de estimación a largo horizonte.\n\n"
    "**Hipótesis:** Inversión a T=200 impulsada por señal fuerte del covariante."
))
cells.append(code(
    "res_3_1 = load_cov('3.1')\n"
    "styled_table(res_3_1, title='Exp 3.1 — ARIMAX β=0.8')\n"
    "plot_metrics_horizon(res_3_1, title='Exp 3.1 — ARIMAX β=0.8')\n"
))
cells.append(code(
    "make_models_3_1 = lambda T: [\n"
    "    SARIMAXModel((1,0,0)),\n"
    "    chronos_cov1,\n"
    "]\n"
    "plot_rep_cov(dgp_3_1, make_models_3_1,\n"
    "             dgp_params={'phi': 0.6, 'beta': 0.8, 'sigma_y': 1.0, 'sigma_x': 1.0, 'rho_x': 0.7},\n"
    "             title='Exp 3.1 — ARIMAX β=0.8')\n"
))

# ── Exp 3.4 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 3.4 — VARX bivariado (VAR con covariante) *(sin inversión)*\n\n"
    "**DGP:** $Y_t = A\\,Y_{t-1} + \\gamma\\,X_t + \\varepsilon_t$, "
    "$A = \\begin{pmatrix}0.5&0.1\\\\0.1&0.5\\end{pmatrix}$, "
    "$\\gamma=(0.5, 0.3)^{\\top}$, $X_t = 0.7\\,X_{t-1} + \\eta_t$\n\n"
    "**Contraste directo con Exp 3.1: sin inversión.** VARMAX-OLS domina en todas las métricas "
    "en ambos T (+21%/+12% RMSE a T=50, +8% a T=200). "
    "La estructura VARX con covariante beneficia al modelo paramétrico: "
    "las interdependencias cross-variable están explícitamente representadas en el VAR, "
    "mientras Chronos trata cada variable de forma más independiente.\n\n"
    "**Hipótesis:** VARMAX domina sin inversión. La dimensionalidad multivariante protege al modelo clásico."
))
cells.append(code(
    "res_3_4 = load_mv_cov('3.4')\n"
    "styled_table(res_3_4, title='Exp 3.4 — VARX bivariado', is_mv=True)\n"
    "plot_metrics_horizon(res_3_4, title='Exp 3.4 — VARX bivariado', is_mv=True)\n"
))
cells.append(code(
    "make_models_3_4 = lambda T: [\n"
    "    VARMAXModel(1),\n"
    "    chronos_mv_cov1,\n"
    "]\n"
    "plot_rep_cov(dgp_3_4, make_models_3_4,\n"
    "             dgp_params={},  # VARX_DGP: parámetros en __init__\n"
    "             is_mv=True, var_idx=0,\n"
    "             title='Exp 3.4 — VARX bivariado')\n"
))

# ── Exp 3.6 ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Exp 3.6 — ADL-ECM cointegrado *(el costo de la misspecificación del orden de integración)*\n\n"
    "**DGP:** $X_t = X_{t-1} + u_t$, "
    "$\\Delta Y_t = -0.3\\,(Y_{t-1} - X_{t-1}) + \\Delta X_t + \\eta_t$ "
    "(cointegración con vector $(1,-1)$)\n\n"
    "**El más rico metodológicamente.** Ranking RMSE: "
    "ARDL-ECM < SARIMAX-niv < Chronos < SARIMAX-dif. "
    "Sobrediferenciar es el error más costoso: SARIMAX(1,1,0) "
    "es +82% peor en RMSE a T=50 y +40% a T=200 vs ARDL-ECM. "
    "SARIMAX en niveles, sin conocer la cointegración, es mucho mejor que SARIMAX diferenciado. "
    "Chronos se posiciona entre ambas especificaciones de SARIMAX — "
    "captura implícitamente parte de la dinámica de largo plazo.\n\n"
    "**Nota:** ARDL-ECM no produce intervalos de predicción.\n\n"
    "**Hipótesis:** La correcta especificación del orden de integración es el factor dominante."
))
cells.append(code(
    "res_3_6 = load_cov('3.6')\n"
    "styled_table(res_3_6, title='Exp 3.6 — ADL-ECM cointegrado')\n"
    "plot_metrics_horizon(res_3_6, metrics=('rmse','crps','cov_95','winkler_95'),\n"
    "                     title='Exp 3.6 — ADL-ECM cointegrado')\n"
))
cells.append(code(
    "make_models_3_6 = lambda T: [\n"
    "    ARDLModel(),\n"
    "    SARIMAXModel((1,1,0), name_suffix='dif. con X'),\n"
    "    SARIMAXModel((1,0,0), name_suffix='niv. con X'),\n"
    "    chronos_cov1,\n"
    "]\n"
    "plot_rep_cov(dgp_3_6, make_models_3_6,\n"
    "             dgp_params={'alpha_ecm': -0.3, 'sigma': 1.0, 'sigma_x': 1.0},\n"
    "             title='Exp 3.6 — ADL-ECM cointegrado')\n"
))


# ════════════════════════════════════════════════════════════════════════════
# Escribir notebook
# ════════════════════════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.9.0"
    }
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook escrito en: {OUT}")
print(f"Celdas: {len(cells)}")
