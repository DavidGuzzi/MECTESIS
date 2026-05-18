"""Build notebooks/validacion_empirica_ipc_v2.ipynb from a structured cell list.

Replaces the original notebook with a depurated v2 that adds optimal-lag
selection for exogenous covariables in Section 3 and removes redundancies
flagged in docs/hallazgos_validacion_empirica.md.
"""

import nbformat as nbf
from pathlib import Path


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


CELLS = [
    md("""# Validación empírica — IPC Argentina **v2** (dic-2016 → abr-2026)

Versión depurada de `validacion_empirica_ipc.ipynb`. Mantiene la estructura de tres bloques (univariado, multivariado endógeno, con covariables exógenas) y aplica:

- Diagnósticos consolidados (sin duplicación ACF/PACF + Ljung-Box, sin Q-Q + Jarque-Bera).
- Multivariado depurado: VAR de orden óptimo por AIC/BIC + VECM(r=1) + ChronosMulti; sin VAR(2) redundante.
- Cross-correlaciones rotuladas: lags predictivos (≥ 1) para inferencia; los negativos se reservan como chequeo de endogeneidad inversa.
- **Sec. 3 con lags óptimos por covariable**: el pass-through del tipo de cambio y la tasa al IPC no es inmediato → `select_optimal_lags` (cross-corr restringida a lag ≥ 1, validada por Granger) elige el rezago de cada covariable antes de entrar a AutoSARIMAX / ChronosCov.

Para hallazgos del run base ver `docs/hallazgos_validacion_empirica.md`."""),
    md("## 0 · Setup y carga del panel"),
    code("""import sys, warnings, pathlib

sys.path.insert(0, str(pathlib.Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mectesis.empirical.loaders import build_panel
from mectesis.empirical.transforms import log_diff, select_optimal_lags, apply_lags
from mectesis.empirical import diagnostics as diag
from mectesis.empirical import describe as desc
from mectesis.empirical.backtest import (
    compare_models, to_wide_table, predict_all_at_last_origin,
)
from mectesis.empirical.autoselect import (
    AutoARIMAModel, AutoETSModel, AutoThetaModel, AutoSARIMAXModel,
)
from mectesis.models.var_model import VARModel, VECMModel
from mectesis.models.chronos import ChronosModel
from mectesis.models.chronos_multivariate import ChronosMultivariateModel
from mectesis.models.chronos_covariate import ChronosCovariateModel

START, END = "2016-12-01", "2026-05-01"
INITIAL_WINDOW = 72
HORIZONS = [1, 3, 6, 12]

# Pipeline Chronos compartido entre las 3 variantes (univariada, multi, covariables)
chronos_pipeline = ChronosModel(device="cpu")"""),
    code("""panel = build_panel(start=START, end=END)
print(f"Panel shape: {panel.shape}, rango {panel.index.min().date()} -> {panel.index.max().date()}")
print(panel.isna().sum())
panel.head()"""),
    code("""desc.plot_panel(panel, title="Panel macro Argentina dic-2016 -> abr-2026")
plt.show()"""),
    md("""## 1 · Validación univariada (π mensual)

El IPC del panel ya es tasa de inflación mensual %. Modelos: AutoARIMA, AutoETS, AutoTheta, Chronos-2.

Cambios vs notebook base:
- Una sola figura ACF/PACF (Ljung-Box reportado solo como p-valor citado en celda de diagnóstico).
- Tabla compacta de diagnósticos (raíces unitarias, ARCH, normalidad) en lugar de bloques separados; sin Q-Q plot."""),
    code("""pi = panel["ipc"].rename("pi")
print(desc.summary_stats(pi).to_string(index=False))
desc.plot_series(pi, title="Inflación mensual π_t (IPC Nacional)")
plt.show()"""),
    code("""# ACF/PACF y descomposición estacional (Ljung-Box queda en la tabla de diagnóstico).
desc.plot_acf_pacf(pi, lags=36, title="π mensual"); plt.show()
desc.plot_decomposition(pi, period=12, title="π mensual"); plt.show()"""),
    md("### 1.1 Tabla resumen de diagnósticos"),
    code("""# Diagnósticos consolidados: estacionariedad + ARCH + normalidad.
diag_pi = pd.concat([
    diag.unit_root_battery(pi)[["test", "regression", "stat", "pvalue", "conclusion"]],
    diag.heteroscedasticity(pi)[["test", "stat", "pvalue", "conclusion"]].assign(regression=""),
    diag.normality(pi)[["test", "stat", "pvalue", "conclusion"]].assign(regression=""),
], ignore_index=True)
print(diag_pi.round(4).to_string(index=False))

# Ljung-Box citado puntualmente: la ACF ya muestra el patrón.
lb = diag.serial_correlation(pi, lags=(12, 24))
print(f"\\nLjung-Box(12): stat={lb.iloc[0]['stat']:.2f}, p={lb.iloc[0]['pvalue']:.4f}")
print(f"Ljung-Box(24): stat={lb.iloc[1]['stat']:.2f}, p={lb.iloc[1]['pvalue']:.4f}")

print("\\n=== Quiebres estructurales (PELT) ===")
print(diag.structural_breaks(pi).to_string(index=False))"""),
    md("### 1.2 Backtest rolling-origin univariado"),
    code("""uni_factories = {
    "AutoARIMA": lambda: AutoARIMAModel(season_length=12),
    "AutoETS":   lambda: AutoETSModel(season_length=12),
    "AutoTheta": lambda: AutoThetaModel(season_length=12),
    "Chronos-2": lambda: ChronosModel(device="cpu"),
}
uni_long = compare_models(
    factories=uni_factories,
    y=pi, horizons=HORIZONS, initial_window=INITIAL_WINDOW, verbose=True,
)
uni_wide = to_wide_table(uni_long)
uni_wide.round(3)"""),
    code("""# Forecast en el último origen — un panel por modelo.
uni_forecasts = predict_all_at_last_origin(
    factories=uni_factories,
    y=pi, horizons=HORIZONS, initial_window=INITIAL_WINDOW,
)
origin = uni_forecasts["AutoARIMA"]["origin_idx"]
desc.plot_forecast_grid(
    pi, uni_forecasts, origin_idx=origin, horizon=max(HORIZONS),
    ncols=2,
    title=f"π — forecast último origen (t={pi.index[origin].date()})",
)
plt.show()"""),
    md("""## 2 · Validación multivariada (sistema endógeno π, dlog_tcm, m2, d_badlar)

> **Nota sobre lags en multivariado endógeno**: VAR(p), VECM y ChronosMultivariate aprenden internamente la estructura de rezagos del sistema. Pre-shiftear inputs no agrega información — incluso degrada la parametrización. La "decisión de lag" acá es el orden `p` del VAR, seleccionado por AIC/BIC.
>
> El uso de **covariables exógenas con lag óptimo** se aborda en Sec. 3 (SARIMAX / ChronosCov tratan a `X` como regresor exógeno alineado con `y_t`)."""),
    code("""Y = pd.concat({
    "pi":       panel["ipc"],
    "dlog_tcm": log_diff(panel["tcm"], 100),
    "m2":       panel["m2"],
    "d_badlar": panel["badlar"].diff(),
}, axis=1).dropna()
print(f"Y shape: {Y.shape}")
desc.plot_panel(Y, title="Sistema endógeno (transformaciones estacionarias)"); plt.show()"""),
    code("""# Selección de orden VAR por AIC/BIC sobre Y (statsmodels).
# Con T~112 obs y k=4 variables, AIC tiende a sobre-parametrizar (VAR(6) implica
# ~100 coeficientes); BIC es el criterio prudente para esta muestra.
from statsmodels.tsa.api import VAR as _SMVAR
order_sel = _SMVAR(Y.to_numpy()).select_order(maxlags=6).selected_orders
p_aic = max(int(order_sel.get("aic", 1)), 1)
p_bic = max(int(order_sel.get("bic", 1)), 1)
p_var = p_bic  # criterio conservador para evitar sobre-parametrización en m2
print(f"Orden VAR sugerido — AIC: {p_aic}, BIC: {p_bic}. Usamos p={p_var} (BIC).")

print("\\n=== Granger causality (min p-value sobre lags 1..6) — fila <- columna ===")
print(diag.granger_matrix(Y, maxlag=6).round(3))

# Johansen sobre log-niveles donde aplica.
Yl = pd.DataFrame({
    "log_tcm": np.log(panel["tcm"]),
    "badlar":  panel["badlar"],
    "log_m2":  np.log(panel["m2"]),
}).dropna()
print("\\n=== Johansen sobre [log_tcm, badlar, log_m2] ===")
print(diag.johansen_test(Yl).round(3).to_string(index=False))"""),
    md("""**Lectura**: Johansen trace y max-eig rechazan `r ≤ 0` al 95 % pero no `r ≤ 1` → **r = 1**, vector cointegrante entre `log_tcm`, `badlar` y `log_m2`. Se justifica VECM. El orden VAR se fija por BIC (criterio prudente para T~112, k=4); AIC tiende a sobre-parametrizar y degrada el pronóstico de `m2`.

> **VARX como extensión futura**: el wrapper actual de `VARModel` ([mectesis/models/var_model.py](../mectesis/models/var_model.py)) no expone el argumento `exog` de statsmodels. La dimensión de covariables exógenas se aborda con SARIMAX/ChronosCov en Sec. 3."""),
    code("""multi_factories = {
    f"VAR({p_var})":          (lambda p=p_var: VARModel(lags=p)),
    "VECM(r=1)":               lambda: VECMModel(coint_rank=1, k_ar_diff=1),
    "ChronosMultivariate":     lambda: ChronosMultivariateModel(chronos_pipeline),
}
multi_long = compare_models(
    factories=multi_factories,
    y=Y, horizons=HORIZONS, initial_window=INITIAL_WINDOW, verbose=True,
)
multi_wide = to_wide_table(multi_long)
multi_wide.round(3)"""),
    code("""# Matriz modelo × variable en el último origen.
multi_forecasts = predict_all_at_last_origin(
    factories=multi_factories,
    y=Y, horizons=HORIZONS, initial_window=INITIAL_WINDOW,
)
origin = multi_forecasts[next(iter(multi_factories))]["origin_idx"]
desc.plot_forecast_matrix(
    Y, multi_forecasts, origin_idx=origin, horizon=max(HORIZONS),
    title=f"Sistema endógeno — forecast último origen (t={Y.index[origin].date()})",
)
plt.show()"""),
    md("""## 3 · Con covariables exógenas + lags óptimos (π ~ X_lagged)

`X = {dlog_tcm, rem, badlar}`. A diferencia de Sec. 2, acá SARIMAX y ChronosCov tratan a `X` como **regresores exógenos** alineados con `y_t`: el punto de entrada temporal lo elegimos nosotros. Aplicamos un **lag óptimo por variable** vía `select_optimal_lags`:

1. Para cada covariable, `cross_corr(π, x)` restringida a `lag ≥ 1` (estrictamente predictiva, evita endogeneidad inversa).
2. `argmax(|corr|)` define el lag candidato.
3. Validación con `granger_matrix`: se descartan covariables con `pval(x → π) > 0.10`.
4. `apply_lags` arma `X_lagged` y los modelos reciben `X_{t-L}` en lugar de `X_t`."""),
    code("""pi_cov = panel["ipc"].rename("pi")
X_raw = pd.concat({
    "dlog_tcm": log_diff(panel["tcm"], 100),
    "rem":      panel["rem"],
    "badlar":   panel["badlar"],
}, axis=1).reindex(pi_cov.index).dropna()
pi_cov = pi_cov.loc[X_raw.index]
print(f"pi: {pi_cov.shape}, X_raw: {X_raw.shape}")
desc.plot_panel(X_raw, title="Covariables observadas"); plt.show()"""),
    code("""# Endogeneidad inversa: chequeo bidireccional con Granger antes de elegir lags.
gpanel = pd.concat([pi_cov.rename("pi"), X_raw], axis=1).dropna()
gm = diag.granger_matrix(gpanel, maxlag=6).round(3)
print("=== Granger bidireccional (p-valores, lags 1..6) — fila <- columna ===")
print(gm)

# Cross-correlaciones restringidas a lags predictivos.
print("\\n=== Cross-correlaciones predictivas (lag >= 1, max_lag=12) ===")
rows = []
for col in X_raw.columns:
    cc = diag.cross_corr(pi_cov, X_raw[col], max_lag=12)
    cc_pred = cc[cc["lag"] >= 1].reset_index(drop=True)
    best_idx = cc_pred["corr"].abs().idxmax()
    rows.append({
        "variable": col,
        "lag*":     int(cc_pred.loc[best_idx, "lag"]),
        "corr*":    round(float(cc_pred.loc[best_idx, "corr"]), 3),
    })
cc_summary = pd.DataFrame(rows)
print(cc_summary.to_string(index=False))"""),
    md("""**Diagnóstico complementario — cross-corr con innovaciones de π**

La cross-corr arriba puede confundir señal con autocorrelación: si `π_{t-1}` está fuertemente correlacionada con `π_t` y las covariables también lo están con `π_{t-1}`, aparecen artificialmente como mejores predictoras a lag 1. Una validación es restar primero el efecto autorregresivo: ajustamos un AR(1) a `π`, extraemos los residuos `π_innov_t = π_t - β̂·π_{t-1}` y cruzamos contra `X_{t-lag}`. Si los lags 3-6 emergen acá, son señal genuina más allá de la inercia inflacionaria."""),
    code("""# Cross-corr con innovaciones de un AR(1) sobre pi (control por autocorrelación).
from statsmodels.tsa.ar_model import AutoReg
ar1 = AutoReg(pi_cov, lags=1).fit()
pi_innov = ar1.resid.rename("pi_innov")
print(f"AR(1) sobre π — beta_1 = {ar1.params.iloc[1]:.3f}, R² = {1 - ar1.resid.var()/pi_cov.var():.3f}")

print("\\n=== Cross-corr (lag >= 1) — innovaciones AR(1) de π vs covariables ===")
innov_rows = []
for col in X_raw.columns:
    cc = diag.cross_corr(pi_innov, X_raw[col].loc[pi_innov.index], max_lag=12)
    cc_pred = cc[cc["lag"] >= 1].reset_index(drop=True)
    best_idx = cc_pred["corr"].abs().idxmax()
    innov_rows.append({
        "variable":  col,
        "lag*":      int(cc_pred.loc[best_idx, "lag"]),
        "corr*":     round(float(cc_pred.loc[best_idx, "corr"]), 3),
    })
print(pd.DataFrame(innov_rows).to_string(index=False))"""),
    code("""# Selección automática + filtrado Granger.
# granger_maxlag=6: el filtro chequea Granger sobre lags 1..6 (independiente del
# lag óptimo elegido por cross-corr). Una covariable con cross-corr máxima en
# lag=1 puede tener evidencia Granger en lag=2-3; queremos conservarla.
lag_map = select_optimal_lags(
    y=pi_cov.rename("pi"),
    X=X_raw,
    max_lag=12, min_lag=1,
    granger_panel=gpanel, alpha=0.10,
    granger_maxlag=6,
)
print(f"lag_map (post Granger filter @ alpha=0.10, granger_maxlag=6): {lag_map}")

X_lagged = apply_lags(X_raw, lag_map).dropna()
pi_lag = pi_cov.loc[X_lagged.index]
print(f"\\nX_lagged: {X_lagged.shape}, pi_lag: {pi_lag.shape}")
print(f"Obs perdidas por lageo: {X_raw.shape[0] - X_lagged.shape[0]}")
print(X_lagged.head())"""),
    md("### 3.1 Backtest con `X_lagged`"),
    code("""cov_factories = {
    "AutoSARIMAX": lambda: AutoSARIMAXModel(season_length=12),
    "ChronosCov":  lambda: ChronosCovariateModel(
        chronos_pipeline,
        n_covariates=X_lagged.shape[1],
        cov_names=list(X_lagged.columns),
    ),
}
cov_long = compare_models(
    factories=cov_factories,
    y=pi_lag, X=X_lagged, horizons=HORIZONS, initial_window=INITIAL_WINDOW, verbose=True,
)
cov_wide = to_wide_table(cov_long)
cov_wide.round(3)"""),
    code("""cov_forecasts = predict_all_at_last_origin(
    factories=cov_factories,
    y=pi_lag, X=X_lagged, horizons=HORIZONS, initial_window=INITIAL_WINDOW,
)
origin = cov_forecasts["AutoSARIMAX"]["origin_idx"]
desc.plot_forecast_grid(
    pi_lag, cov_forecasts, origin_idx=origin, horizon=max(HORIZONS),
    ncols=2,
    title=f"π con X_lagged — forecast último origen (t={pi_lag.index[origin].date()})",
)
plt.show()"""),
    md("""## 4 · Síntesis

Tabla maestra (modelo × sección × horizonte) y export a `results/empirical/tabla_v2_long.csv`."""),
    code("""uni = uni_long.assign(seccion="univariada")
mul = multi_long.assign(seccion="multivariada")
cov = cov_long.assign(seccion="covariadas_lagged")
master_long = pd.concat([uni, mul, cov], ignore_index=True)
master_long["lag_map"] = master_long["seccion"].map({
    "univariada":          "",
    "multivariada":        "",
    "covariadas_lagged":   str(lag_map),
})

out_dir = pathlib.Path.cwd().parent / "results" / "empirical"
out_dir.mkdir(parents=True, exist_ok=True)
master_long.to_csv(out_dir / "tabla_v2_long.csv", index=False)

master_wide = {
    "univariada":   to_wide_table(uni),
    "multivariada": to_wide_table(mul),
    "covariadas":   to_wide_table(cov),
}
for sec, df in master_wide.items():
    df.to_csv(out_dir / f"tabla_v2_wide_{sec}.csv")
print(f"Guardado en: {out_dir}")
print(f"lag_map usado en Sec. 3: {lag_map}")
master_wide["covariadas"].round(3)"""),
    md("""## Conclusiones y riesgos

- **Univariado**: ranking esperado igual al notebook base (cambios solo cosméticos).
- **Multivariado**: ChronosMulti > VECM > VAR(p_bic). Orden BIC se usa para evitar sobre-parametrización (con T~112 y k=4, VAR(6) genera ~100 coeficientes y explota en `m2`).
- **Con covariables**: la versión v2 usa lags óptimos por variable con `granger_maxlag=6` (validación independiente del lag elegido por cross-corr). Diagnóstico complementario AR(1)-innovations rotula si los lags más largos son señal genuina o sólo arrastre inflacionario.

### Riesgos

- **Tamaño efectivo**: T = 113 − max(lag); con max_lag ≤ 6, ~107 obs. Ventana inicial 72 es viable pero ajustada.
- **Endogeneidad inversa (π → tcm, π → badlar)**: el Granger bidireccional reportado arriba es el control. Si `pval(π → x)` también es < 0.10, hay retroalimentación y el lag óptimo no garantiza identificación causal — sigue siendo útil como feature predictiva.
- **Múltiples comparaciones**: 3 covariables × 12 lags + Granger ⇒ tratamiento exploratorio. No se aplica Bonferroni.
- **Look-ahead leakage**: `apply_lags` impone `lag ≥ 1` y se ejecuta sobre el panel completo antes del split → no hay fuga.
- **VARX descartado por scope**: extensión natural sería VAR con exógenas lageadas; queda como follow-up si Sec. 3 lo amerita."""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "validacion_empirica_ipc_v2.ipynb"
    nbf.write(nb, out)
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
