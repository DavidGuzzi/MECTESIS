"""
Generates notebooks/experimentos_multivariados.ipynb
Run once: python scripts/gen_notebook_multivariate.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "experimentos_multivariados.ipynb"
RESULTS_DIR = "results/multivariate"


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
    "# Experimentos Multivariados 2.1–2.7\n\n"
    "**Tesis MEC** — Comparación TSFMs vs Modelos Clásicos bajo DGPs controlados  \n"
    "**Setup:** T ∈ {50, 200} | H = 24 | R_LIST = [500] | Semilla = 3649  \n"
    "**Métricas punto:** Bias, Varianza, MSE, RMSE, MAE  \n"
    "**Métricas probabilísticas:** CRPS, Cobertura 80%/95%, Amplitud 80%/95%, Winkler Score 80%/95%  \n"
    "**Resultados:** guardados en `results/multivariate/` — si existen se cargan sin re-simular\n\n"
    "**Nota:** Experimentos multivariados. Los modelos clásicos (VAR, VECM) modelan "
    "las variables **conjuntamente** de forma paramétrica. Chronos-2 se usa en su modo "
    "**multivariado nativo** (`Chronos-2 (joint)`): pasa todas las k variables juntas "
    "en una sola inferencia — `(1, k, T_train)` → `(k, H, n_quantiles)`. "
    "En los experimentos 2.1 y 2.7 se agrega `Chronos-2 (ind.)` como baseline secundario "
    "(k forecasts univariados independientes) para cuantificar cuánto aporta el modo joint.\n\n"
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
    "from mectesis.dgp import VARDGP, VARGARCHDiagonalDGP, VECMBivariateDGP\n"
    "from mectesis.models import (\n"
    "    VARModel, VECMModel, VARGARCHDiagonalModel,\n"
    "    ChronosMultivariateModel, ChronosPerVarModel,\n"
    "    ChronosModel,\n"
    ")\n"
    "from mectesis.simulation import MultivariateMonteCarloEngine\n"
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
    '_chronos_base  = ChronosModel(device="cpu")\n'
    "chronos_mv     = ChronosMultivariateModel(_chronos_base)  # joint — API nativa\n"
    "chronos_mv_ind = ChronosPerVarModel(_chronos_base)         # independiente — baseline\n"
    'print("Chronos-2 listo.")'
)

# ─── Cell 2: Helper functions ─────────────────────────────────────────────────
c2 = code(
    "# ─── Funciones auxiliares multivariadas ─────────────────────────────────────\n"
    "\n"
    "def _cache_path(exp_id: str, T: int, R: int) -> Path:\n"
    '    return RESULTS / f"exp_{exp_id.replace(\'.\', \'_\')}_T{T}_R{R}.csv"\n'
    "\n"
    "\n"
    "def _save_results_mv(results: dict, path: Path):\n"
    "    \"\"\"Guarda {model: {var_idx: DataFrame}} como CSV con columnas 'model','var'.\"\"\"\n"
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
    "    \"\"\"Carga CSV de vuelta a {model: {var_idx: DataFrame}}.\"\"\"\n"
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
    "def run_exp_mv(dgp, make_models_fn, dgp_params, exp_id,\n"
    "               T_list=T_LIST, R_list=R_LIST, H=H, seed=SEED):\n"
    "    \"\"\"\n"
    "    Corre MC multivariado para todas las combinaciones (T, R).\n"
    "    Si el CSV ya existe, lo carga sin re-simular.\n"
    "    Retorna {(T, R): {model_name: {var_idx: DataFrame}}}.\n"
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
    "                all_results[(T, R)] = _load_results_mv(cache)\n"
    "                continue\n"
    "\n"
    "            print(f'  T={T}, R={R}: simulando ...', end=' ', flush=True)\n"
    "            dgp.rng = np.random.default_rng(seed)\n"
    "            models = make_models_fn(T)\n"
    "            engine = MultivariateMonteCarloEngine(dgp, models, seed=seed)\n"
    "            t0 = time.time()\n"
    "            results = engine.run_monte_carlo(\n"
    "                R, T, H, dgp_params, verbose=False)\n"
    "            elapsed = time.time() - t0\n"
    "            print(f'OK ({elapsed:.0f}s)')\n"
    "\n"
    "            _save_results_mv(results, cache)\n"
    "            all_results[(T, R)] = results\n"
    "\n"
    "    return all_results\n"
    "\n"
    "\n"
    "def compute_blocks_mv(results_TR: dict):\n"
    "    \"\"\"Dado {model: {var_idx: df}}, calcula promedios h=1-12 y h=13-24.\"\"\"\n"
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
    "    \"\"\"Tabla comparativa por (T, R, Modelo, Variable, Bloque).\"\"\"\n"
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
    "def plot_rep_mv(dgp, make_models_fn, dgp_params, var_names,\n"
    "                T=200, H=H, seed=SEED, title=''):\n"
    "    \"\"\"Visualización de una simulación representativa (k subplots verticales).\"\"\"\n"
    "    import copy\n"
    "    dgp_r = copy.deepcopy(dgp)\n"
    "    dgp_r.rng = np.random.default_rng(seed)\n"
    "    y = dgp_r.simulate(T=T, **dgp_params)   # (T, k)\n"
    "    k = y.shape[1]\n"
    "    y_train, y_test = y[:-H], y[-H:]\n"
    "    models = make_models_fn(T)\n"
    "\n"
    "    palette = ['crimson', 'darkorange', 'seagreen', 'purple', 'teal', 'olive']\n"
    "    fig, axes = plt.subplots(k, 1, figsize=(13, 3.5 * k), squeeze=False)\n"
    "\n"
    "    # Fit all models once\n"
    "    for m in models:\n"
    "        m.fit(y_train)\n"
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
    "            y_hat = m.forecast(H)    # (H, k)\n"
    "            ax.plot(x_te, y_hat[:, j], color=palette[i % len(palette)],\n"
    "                    lw=1.5, marker='o', ms=3, label=m.name)\n"
    "            if m.supports_intervals:\n"
    "                lo, hi = m.forecast_intervals(H, level=0.95)\n"
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
    "    \"\"\"Grilla (metric × (T,R)) con curvas por modelo, una figura por variable.\"\"\"\n"
    "    # Infer k from first available result\n"
    "    first_TR = next(iter(all_results.values()))\n"
    "    first_model = next(iter(first_TR.values()))\n"
    "    k = len(first_model)\n"
    "\n"
    "    for j in range(k):\n"
    "        vname = var_names[j] if var_names else f'Y{j+1}'\n"
    "        keys = sorted(all_results.keys())\n"
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
# Each tuple: (exp_id, dgp_cls_name, dgp_init_kwargs_repr, dgp_params_repr,
#              make_models_src, md_desc, exp_name, var_names_repr)

_A1_21 = "[[0.5,0.1],[0.1,0.5]]"
_A1_22 = "[[0.4,0.4],[0.4,0.4]]"
_A1_23 = "[[0.5,0.2],[0.1,0.4]]"
_A2_23 = "[[0.1,0.0],[0.0,0.1]]"
_Sigma_2 = "[[1.0,0.3],[0.3,1.0]]"
_A1_24 = "[[0.5,0.1,0.0],[0.1,0.5,0.1],[0.0,0.1,0.5]]"
_Sigma_3 = "[[1.0,0.2,0.0],[0.2,1.0,0.2],[0.0,0.2,1.0]]"
_A1_25 = ("[[0.3,0.05,0.0,0.0,0.0],"
          "[0.05,0.3,0.05,0.0,0.0],"
          "[0.0,0.05,0.3,0.05,0.0],"
          "[0.0,0.0,0.05,0.3,0.05],"
          "[0.0,0.0,0.0,0.05,0.3]]")
_Sigma_5 = ("[[1.0,0.2,0.0,0.0,0.0],"
            "[0.2,1.0,0.2,0.0,0.0],"
            "[0.0,0.2,1.0,0.2,0.0],"
            "[0.0,0.0,0.2,1.0,0.2],"
            "[0.0,0.0,0.0,0.2,1.0]]")


EXPS_MV = [
    (
        "2.1",
        "VARDGP",
        f"seed=SEED, A_list=[{_A1_21}], Sigma={_Sigma_2}",
        "{}",
        "lambda T: [VARModel(1), chronos_mv, chronos_mv_ind]",
        (
            "**DGP:** VAR(1) bivariado — baja interdependencia\n\n"
            "$$Y_t = A_1 Y_{t-1} + \\varepsilon_t, \\quad "
            "A_1 = \\begin{pmatrix}0.5 & 0.1 \\\\ 0.1 & 0.5\\end{pmatrix}, \\quad "
            "\\Sigma = \\begin{pmatrix}1 & 0.3 \\\\ 0.3 & 1\\end{pmatrix}$$\n\n"
            "**Modelos:** VAR(1), Chronos-2 (joint — API nativa multivariada), "
            "Chronos-2 (ind. — k forecasts univariados independientes)\n\n"
            "**Hipótesis:** Con interdependencia baja, la ventaja de VAR sobre Chronos-joint "
            "es pequeña; Chronos-ind debería ser el más débil al no usar información cruzada."
        ),
        "VAR(1) baja interdependencia",
        '["Y1","Y2"]',
    ),
    (
        "2.2",
        "VARDGP",
        f"seed=SEED, A_list=[{_A1_22}], Sigma={_Sigma_2}",
        "{}",
        "lambda T: [VARModel(1), chronos_mv]",
        (
            "**DGP:** VAR(1) bivariado — alta interdependencia\n\n"
            "$$Y_t = A_1 Y_{t-1} + \\varepsilon_t, \\quad "
            "A_1 = \\begin{pmatrix}0.4 & 0.4 \\\\ 0.4 & 0.4\\end{pmatrix}$$\n\n"
            "**Modelos:** VAR(1), Chronos-2 (joint)\n\n"
            "**Hipótesis:** Con alta interdependencia, el beneficio del modelado conjunto "
            "debería ser más pronunciado que en 2.1 — tanto para VAR como para Chronos-joint."
        ),
        "VAR(1) alta interdependencia",
        '["Y1","Y2"]',
    ),
    (
        "2.3",
        "VARDGP",
        f"seed=SEED, A_list=[{_A1_23},{_A2_23}], Sigma={_Sigma_2}",
        "{}",
        "lambda T: [VARModel(2), VARModel(1), chronos_mv]",
        (
            "**DGP:** VAR(2) bivariado\n\n"
            "$$Y_t = A_1 Y_{t-1} + A_2 Y_{t-2} + \\varepsilon_t$$\n\n"
            "$$A_1 = \\begin{pmatrix}0.5 & 0.2 \\\\ 0.1 & 0.4\\end{pmatrix}, \\quad "
            "A_2 = \\begin{pmatrix}0.1 & 0 \\\\ 0 & 0.1\\end{pmatrix}$$\n\n"
            "**Modelos:** VAR(2) [orden correcto], VAR(1) [misspecificado], "
            "Chronos-2 (joint)\n\n"
            "**Hipótesis:** VAR(2) domina; Chronos-joint puede capturar implícitamente "
            "dinámicas de orden 2 via su contexto largo, a diferencia de VAR(1)."
        ),
        "VAR(2) bivariado",
        '["Y1","Y2"]',
    ),
    (
        "2.4",
        "VARDGP",
        f"seed=SEED, A_list=[{_A1_24}], Sigma={_Sigma_3}",
        "{}",
        "lambda T: [VARModel(1), chronos_mv]",
        (
            "**DGP:** VAR(1) trivariado (3 variables)\n\n"
            "$$A_1 = \\begin{pmatrix}0.5&0.1&0\\\\0.1&0.5&0.1\\\\0&0.1&0.5\\end{pmatrix}$$\n\n"
            "**Modelos:** VAR(1), Chronos-2 (joint)\n\n"
            "**Hipótesis:** A mayor dimensión, el VAR tiene más parámetros que estimar "
            "(curse of dimensionality) mientras Chronos-joint escala mejor."
        ),
        "VAR(1) 3 variables",
        '["Y1","Y2","Y3"]',
    ),
    (
        "2.5",
        "VARDGP",
        f"seed=SEED, A_list=[{_A1_25}], Sigma={_Sigma_5}",
        "{}",
        "lambda T: [VARModel(1), chronos_mv]",
        (
            "**DGP:** VAR(1) 5 variables\n\n"
            "Matriz $5\\times5$ tridiagonal con diagonal $0.3$ y off-diagonal $0.05$.\n\n"
            "**Modelos:** VAR(1), Chronos-2 (joint)\n\n"
            "**Nota computacional:** Con T=50 y 5 variables, el VAR puede tener "
            "dificultades de estimación (pocos grados de libertad). Las réplicas fallidas "
            "se excluyen automáticamente de las métricas."
        ),
        "VAR(1) 5 variables",
        '["Y1","Y2","Y3","Y4","Y5"]',
    ),
    (
        "2.6",
        "VARGARCHDiagonalDGP",
        (
            "seed=SEED, "
            f"A1={_A1_21}, "
            "omegas=[0.1,0.1], alphas=[0.1,0.15], betas=[0.8,0.75]"
        ),
        "{}",
        "lambda T: [VARGARCHDiagonalModel(), VARModel(1), chronos_mv]",
        (
            "**DGP:** VAR(1) bivariado con volatilidad condicional GARCH(1,1) diagonal\n\n"
            "Media: $Y_t = A_1 Y_{t-1} + u_t$ con $A_1$ como en 2.1.\n\n"
            "Ruido: $u_{it} = \\sigma_{it}\\,z_{it}$, con "
            "$\\sigma_{it}^2 = \\omega_i + \\alpha_i u_{i,t-1}^2 + \\beta_i \\sigma_{i,t-1}^2$.\n\n"
            "**Modelos:** VAR(1)+GARCH-diag (intervalos por simulación), "
            "VAR(1) estándar (ignora heteroscedasticidad), Chronos-2 (joint)\n\n"
            "**Hipótesis:** El VAR+GARCH-diag mejora la calibración de intervalos; "
            "Chronos-joint puede capturar heteroscedasticidad implícitamente."
        ),
        "VAR(1) + GARCH diagonal",
        '["Y1","Y2"]',
    ),
    (
        "2.7",
        "VECMBivariateDGP",
        "seed=SEED",
        "{}",
        "lambda T: [VECMModel(coint_rank=1), VARModel(1), chronos_mv, chronos_mv_ind]",
        (
            "**DGP:** VECM bivariado — cointegración rango 1\n\n"
            "$$\\Delta Y_t = \\alpha\\,\\beta^{\\top}\\,Y_{t-1} + \\Gamma_1\\,\\Delta Y_{t-1} + \\varepsilon_t$$\n\n"
            "$$\\beta=(1,-1)^{\\top},\\quad \\alpha=(-0.4,0.2)^{\\top},\\quad "
            "\\Gamma_1=\\text{diag}(0.3,0.3)$$\n\n"
            "Cada serie individualmente es $I(1)$; la combinación $Y_1-Y_2$ es $I(0)$.\n\n"
            "**Modelos:** VECM(r=1) [Johansen], VAR(1) en niveles "
            "[ignora cointegración], Chronos-2 (joint), Chronos-2 (ind.)\n\n"
            "**Hipótesis:** Chronos-2 sin restricción de cointegración muestra sesgo "
            "creciente en $h \\geq 6$ al no reproducir la relación de largo plazo. "
            "El modo joint puede capturar la cointegración implícitamente mejor que el independiente."
        ),
        "VECM bivariado rango 1",
        '["Y1","Y2"]',
    ),
]

cells = [c0, c1, c2]

for (exp_id, dgp_cls, dgp_init_kwargs, dgp_params_repr,
     make_fn_src, md_desc, exp_name, var_names_repr) in EXPS_MV:
    slug = exp_id.replace(".", "_")

    # Markdown header
    cells.append(md(f"---\n## Experimento {exp_id}\n\n{md_desc}"))

    # Run MC (with save/load)
    cells.append(code(
        f"dgp_{slug}         = {dgp_cls}({dgp_init_kwargs})\n"
        f"make_models_{slug} = {make_fn_src}\n"
        f"dgp_params_{slug}  = {dgp_params_repr}\n"
        f"var_names_{slug}   = {var_names_repr}\n"
        f"\n"
        f"results_{slug} = run_exp_mv(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    exp_id='{exp_id}',\n"
        f")"
    ))

    # Visualization + table + metrics
    cells.append(code(
        f"# Simulación representativa (T=T_LIST[0])\n"
        f"plot_rep_mv(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    var_names=var_names_{slug},\n"
        f"    T=T_LIST[0],\n"
        f"    title=f\"Exp {exp_id}: {exp_name} — Rep. (T={{T_LIST[0]}}, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Simulación representativa (T=T_LIST[1])\n"
        f"plot_rep_mv(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f"    var_names=var_names_{slug},\n"
        f"    T=T_LIST[1],\n"
        f"    title=f\"Exp {exp_id}: {exp_name} — Rep. (T={{T_LIST[1]}}, seed={{SEED}})\"\n"
        f")\n"
        f"\n"
        f"# Tabla de métricas por bloque\n"
        f'print("Tabla de métricas — Exp {exp_id}")\n'
        f"results_table_mv(results_{slug}, var_names=var_names_{slug})\n"
        f"\n"
        f"# Gráficos de métricas por horizonte (por variable)\n"
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
