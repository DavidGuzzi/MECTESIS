"""
Generates notebooks/experimentos_univariados.ipynb
Run once: python scripts/gen_notebook.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "experimentos_univariados.ipynb"


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
    "**Setup:** T ∈ {200, 1000} | H = 24 | R = 500 | Semilla = 3649 | Modelos Core únicamente\n\n"
    "---"
)

# ─── Cell 1: Imports & constants ──────────────────────────────────────────────
c1 = code(
    "import warnings\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from IPython.display import display\n"
    "\n"
    "from mectesis.dgp import AR1, RandomWalk, AR1WithTrend, SeasonalDGP, AR1WithBreak\n"
    "from mectesis.models import (\n"
    "    ARIMAModel, ChronosModel,\n"
    "    NaiveModel, DriftModel, SeasonalNaiveModel,\n"
    "    SARIMAModel, ARIMAWithTrendModel, ARIMAWithBreakModel,\n"
    ")\n"
    "from mectesis.simulation import MonteCarloEngine\n"
    "\n"
    "SEED   = 3649\n"
    "H      = 24\n"
    "R      = 500\n"
    "T_LIST = [200, 1000]\n"
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
    "def run_exp(dgp, make_models_fn, dgp_params,\n"
    "            T_list=T_LIST, H=H, n_sim=R, seed=SEED):\n"
    "    \"\"\"Corre MC para todos los T. make_models_fn(T) retorna lista de modelos.\"\"\"\n"
    "    all_results = {}\n"
    "    for T in T_list:\n"
    "        dgp.rng = np.random.default_rng(seed)  # reset para reproducibilidad\n"
    "        models = make_models_fn(T)\n"
    "        engine = MonteCarloEngine(dgp, models, seed=seed)\n"
    "        print(f\"  T={T}: {n_sim} reps × {len(models)} modelos...\",\n"
    "              end=\" \", flush=True)\n"
    "        all_results[T] = engine.run_monte_carlo(\n"
    "            n_sim, T, H, dgp_params, verbose=False)\n"
    '        print("OK")\n'
    "    return all_results   # {T: {model_name: DataFrame}}\n"
    "\n"
    "\n"
    "def compute_blocks(results_T):\n"
    "    \"\"\"Dado {model_name: df}, calcula promedios para h=1-12 y h=13-24.\"\"\"\n"
    "    out = {}\n"
    "    for mname, df in results_T.items():\n"
    '        df_h = df[df["horizon"] != "avg_all"].copy()\n'
    '        df_h["horizon"] = df_h["horizon"].astype(int)\n'
    "        out[mname] = {\n"
    '            "h=1-12":  df_h[df_h["horizon"] <= 12].mean(numeric_only=True),\n'
    '            "h=13-24": df_h[df_h["horizon"] >= 13].mean(numeric_only=True),\n'
    "        }\n"
    "    return out\n"
    "\n"
    "\n"
    "def results_table(all_results):\n"
    "    \"\"\"Muestra tabla comparativa de métricas por bloque.\"\"\"\n"
    "    rows = []\n"
    "    for T, res_T in sorted(all_results.items()):\n"
    "        for mname, blk in compute_blocks(res_T).items():\n"
    "            for bname, m in blk.items():\n"
    "                rows.append({\n"
    '                    "T": T, "Modelo": mname, "Bloque": bname,\n'
    '                    "Bias":    round(float(m["bias"]),     4),\n'
    '                    "Varianza":round(float(m["variance"]), 4),\n'
    '                    "MSE":     round(float(m["mse"]),      4),\n'
    '                    "RMSE":    round(float(m["rmse"]),     4),\n'
    "                })\n"
    '    df = pd.DataFrame(rows).set_index(["T", "Modelo", "Bloque"])\n'
    '    display(df.style.format(precision=4)\n'
    '              .background_gradient(subset=["RMSE"], cmap="YlOrRd"))\n'
    "\n"
    "\n"
    "def plot_rep(dgp, make_models_fn, dgp_params,\n"
    "             T=200, H=H, seed=SEED, title=\"\"):\n"
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
    "    ax.axvline(len(y_train) - 0.5, color=\"gray\", ls=\":\", lw=1, alpha=0.6)\n"
    "\n"
    '    palette = ["crimson", "darkorange", "seagreen", "purple", "teal", "olive"]\n'
    "    for i, m in enumerate(models):\n"
    "        m.fit(y_train)\n"
    "        ax.plot(x_te, m.forecast(H),\n"
    "                color=palette[i % len(palette)],\n"
    '                lw=1.5, marker="o", ms=3, label=m.name)\n'
    "\n"
    '    ax.set(title=title, xlabel="t", ylabel="$Y_t$")\n'
    "    ax.legend(fontsize=9)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "\n"
    'def plot_metrics(all_results, title="", metrics=("rmse", "bias")):\n'
    "    \"\"\"Gráficos de métricas vs horizonte h=1..24 por modelo.\"\"\"\n"
    "    T_list = sorted(all_results.keys())\n"
    "    fig, axes = plt.subplots(\n"
    "        len(metrics), len(T_list),\n"
    "        figsize=(7 * len(T_list), 3.5 * len(metrics)),\n"
    "        squeeze=False,\n"
    "    )\n"
    '    palette = ["crimson", "darkorange", "seagreen", "purple", "teal", "steelblue"]\n'
    "\n"
    "    for col, T in enumerate(T_list):\n"
    "        for row, metric in enumerate(metrics):\n"
    "            ax = axes[row][col]\n"
    "            for i, (mname, df) in enumerate(all_results[T].items()):\n"
    '                df_h = df[df["horizon"] != "avg_all"].copy()\n'
    '                df_h["horizon"] = df_h["horizon"].astype(int)\n'
    "                ax.plot(df_h[\"horizon\"], df_h[metric],\n"
    "                        label=mname, color=palette[i % len(palette)], lw=1.5)\n"
    "            ax.axvline(12.5, color=\"gray\", ls=\":\", lw=0.8, alpha=0.5)\n"
    "            ax.set(\n"
    "                title=f\"T={T} — {metric.upper()}\",\n"
    '                xlabel="Horizonte h",\n'
    "                ylabel=metric.upper(),\n"
    "            )\n"
    "            ax.legend(fontsize=8)\n"
    "\n"
    "    fig.suptitle(title, fontsize=12)\n"
    "    plt.tight_layout()\n"
    "    plt.show()"
)

# ─── Experiments definition ───────────────────────────────────────────────────
# Each tuple: (exp_id, dgp_cls_name, dgp_params, make_models_src, md_desc)
EXPS = [
    (
        "1.1",
        "AR1",
        {"phi": 0.3},
        "lambda T: [ARIMAModel((1,0,0)), NaiveModel(), DriftModel(), chronos]",
        (
            "**DGP:** AR(1) baja persistencia — $Y_t = 0.3\\,Y_{t-1} + \\varepsilon_t$  \n"
            "**Modelos core:** ARIMA(1,0,0), Naive, Drift, Chronos-2"
        ),
    ),
    (
        "1.2",
        "AR1",
        {"phi": 0.9},
        "lambda T: [ARIMAModel((1,0,0)), NaiveModel(), chronos]",
        (
            "**DGP:** AR(1) alta persistencia — $Y_t = 0.9\\,Y_{t-1} + \\varepsilon_t$  \n"
            "**Modelos core:** ARIMA(1,0,0), Naive, Chronos-2"
        ),
    ),
    (
        "1.3",
        "RandomWalk",
        {"drift": 0.0},
        "lambda T: [ARIMAModel((0,1,0)), DriftModel(), chronos]",
        (
            "**DGP:** Random Walk I(1) sin drift — $Y_t = Y_{t-1} + \\varepsilon_t$  \n"
            "**Modelos core:** ARIMA(0,1,0), Drift, Chronos-2"
        ),
    ),
    (
        "1.4",
        "RandomWalk",
        {"drift": 0.5},
        "lambda T: [ARIMAModel((0,1,0)), DriftModel(), chronos]",
        (
            "**DGP:** Random Walk I(1) con drift — $Y_t = 0.5 + Y_{t-1} + \\varepsilon_t$  \n"
            "**Modelos core:** ARIMA(0,1,0), Drift, Chronos-2"
        ),
    ),
    (
        "1.5",
        "AR1WithTrend",
        {"intercept": 5.0, "trend_coeff": 0.1, "phi": 0.6},
        "lambda T: [ARIMAWithTrendModel((1,0,0), trend='ct'), chronos]",
        (
            "**DGP:** AR(1) + tendencia — $Y_t = 5 + 0.1t + 0.6\\,Y_{t-1} + \\varepsilon_t$  \n"
            "**Modelos core:** ARIMA(1,0,0)+trend (trend='ct'), Chronos-2"
        ),
    ),
    (
        "1.6",
        "SeasonalDGP",
        {"phi": 0.5, "Phi": 0.3, "s": 4, "integrated": False},
        "lambda T: [SARIMAModel((1,0,0),(1,0,0,4)), SeasonalNaiveModel(4), chronos]",
        (
            "**DGP:** SARIMA trimestral (s=4) — $(1-0.5L)(1-0.3L^4)Y_t = \\varepsilon_t$  \n"
            "**Modelos core:** SARIMA(1,0,0)(1,0,0)_4, SeasonalNaive(s=4), Chronos-2"
        ),
    ),
    (
        "1.7",
        "SeasonalDGP",
        {"integrated": True, "s": 12},
        "lambda T: [SARIMAModel((0,1,0),(0,1,0,12)), SeasonalNaiveModel(12), chronos]",
        (
            "**DGP:** SARIMA mensual (s=12) — $(1-L)(1-L^{12})Y_t = \\varepsilon_t$  \n"
            "**Modelos core:** SARIMA(0,1,0)(0,1,0)_12, SeasonalNaive(s=12), Chronos-2"
        ),
    ),
    (
        "1.8",
        "AR1WithBreak",
        {"phi_before": 0.3, "phi_after": 0.8},
        "lambda T: [ARIMAWithBreakModel((1,0,0), T_total=T), chronos]",
        (
            "**DGP:** AR(1) con quiebre estructural en $T/2$ — "
            "$\\phi$ cambia de 0.3 a 0.8  \n"
            "**Modelos core:** ARIMA(1,0,0)+break (dummy exógena), Chronos-2"
        ),
    ),
]

cells = [c0, c1, c2]

for exp_id, dgp_cls, dgp_params, make_fn_src, md_desc in EXPS:
    slug = exp_id.replace(".", "_")

    # — Markdown header for experiment
    cells.append(md(
        f"---\n## Experimento {exp_id}\n\n{md_desc}"
    ))

    # — Run Monte Carlo
    cells.append(code(
        f"dgp_{slug}         = {dgp_cls}(seed=SEED)\n"
        f"make_models_{slug} = {make_fn_src}\n"
        f"dgp_params_{slug}  = {repr(dgp_params)}\n"
        f"\n"
        f'print("\\n=== Experimento {exp_id} ===")\n'
        f"results_{slug} = run_exp(\n"
        f"    dgp_{slug},\n"
        f"    make_models_{slug},\n"
        f"    dgp_params_{slug},\n"
        f")"
    ))

    # — Visualization + table + metrics
    cells.append(code(
        f"# Visualización representativa (T=200)\n"
        f"plot_rep(\n"
        f"    dgp_{slug}, make_models_{slug}, dgp_params_{slug},\n"
        f'    T=200, title="Exp {exp_id} — Simulación representativa (T=200, seed=SEED)"\n'
        f")\n"
        f"\n"
        f"# Tabla de métricas por bloque\n"
        f'print("Tabla de métricas — Exp {exp_id}")\n'
        f"results_table(results_{slug})\n"
        f"\n"
        f"# Gráficos de métricas por horizonte\n"
        f"plot_metrics(\n"
        f"    results_{slug},\n"
        f'    title="Exp {exp_id} — Métricas por horizonte (R={{}})".format(R)\n'
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
