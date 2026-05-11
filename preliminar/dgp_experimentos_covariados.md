# Procesos generadores de datos — Experimentos con covariables 3.1–3.6

**Experimentos 3.1–3.6:** $T \in \{50, 200\}$, $H=24$, $R=500$, semilla $=3649$.  
**Supuesto clave:** Los valores futuros de la covariable $X_{T+1},\ldots,X_{T+H}$ se asumen **conocidos** en el momento de la predicción (*oracle forecast*). Esto evalúa el límite superior del beneficio de incluir información exógena, aislando el efecto del regresor externo de los errores de predicción de $X$.  
**Experimentos 3.1–3.3, 3.5:** Objetivo univariante $Y_t$, covariable(s) $X_t$. Motor: `CovariateMonteCarloEngine`.  
**Experimento 3.4:** Objetivo bivariante $Y_t \in \mathbb{R}^2$, covariable escalar $X_t$. Motor: `CovariateMultivariateEngine`.  
**Experimento 3.6:** Objetivo univariante cointegrado con $X_t \sim I(1)$. Motor: `CovariateMonteCarloEngine`.

---

## Exp 3.1 — ARIMAX con efecto exógeno fuerte

$$Y_t = 0.6\,Y_{t-1} + 0.8\,X_t + \varepsilon_t, \qquad X_t = 0.7\,X_{t-1} + \eta_t$$

donde $\varepsilon_t \sim \mathcal{N}(0,1)$, $\eta_t \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** $Y_t$ estacionario ($|\phi|<1$). El covariate $X_t$ es un AR(1) estacionario con persistencia media ($\rho_x=0.7$). Efecto exógeno fuerte: $\beta=0.8$ implica que $X_t$ explica ~55% de la varianza no-AR de $Y_t$. $\text{Var}(X) = \sigma_x^2/(1-\rho_x^2) \approx 1.96$; $\text{Var}(Y) \approx 3.52$. El pronóstico óptimo dado $X_{T+h}$ requiere el conocimiento conjunto de $\phi$ y $\beta$.

**Por qué es relevante:** Evalúa si los modelos explotan eficientemente una covariable de efecto grande cuando su valor futuro es conocido. Contrasta con exp 3.2 (efecto débil).

**DGP — `mectesis/dgp/arimax_dgp.py` · `ARIMAX_DGP`**
```python
ARIMAX_DGP(seed=SEED).simulate(T=200, phi=0.6, beta=0.8, sigma_y=1.0, sigma_x=1.0, rho_x=0.7)
# Retorna {"y": (T,), "X": (T, 1)}
# Burn-in=200; X inicializa en 0; Y inicializa en 0
```

**Modelo — `mectesis/models/sarimax_model.py` · `SARIMAXModel`**
```python
# Librería: statsmodels
m = SARIMAXModel(order=(1, 0, 0), name_suffix='con X')

# Estimación:
m.fit(y_train, X_train=X_train)
# Internamente: SARIMAX(y_train, order=(1,0,0), exog=X_train,
#                        trend='c', enforce_stationarity=False).fit(disp=False)
# → MLE; estima φ̂, β̂, σ̂², constante

# Pronóstico puntual con X_future conocido:
y_hat = m.forecast(H, X_future=X_future)
# → fitted.forecast(steps=H, exog=X_future)

# Intervalos:
lo, hi = m.forecast_intervals(H, level=0.95, X_future=X_future)
# → fitted.get_forecast(steps=H, exog=X_future).conf_int(alpha=0.05)
```

---

## Exp 3.2 — ARIMAX con efecto exógeno débil

$$Y_t = 0.6\,Y_{t-1} + 0.2\,X_t + \varepsilon_t, \qquad X_t = 0.7\,X_{t-1} + \eta_t$$

donde $\varepsilon_t \sim \mathcal{N}(0,1)$, $\eta_t \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** Idéntico al exp 3.1 excepto $\beta=0.2$. El covariate contribuye $\beta^2\,\text{Var}(X) \approx 0.078$ a la varianza de $Y$ (vs 1.255 en 3.1) — ~7% de la varianza total. La señal exógena es difícil de extraer del contexto observado.

**Por qué es relevante:** Con efecto débil, la ventaja de disponer de $X_{\text{future}}$ es pequeña. Evalúa si los modelos estiman correctamente un $\beta$ pequeño y si Chronos puede extraerlo sin estimación paramétrica.

**DGP — `ARIMAX_DGP`**
```python
ARIMAX_DGP(seed=SEED).simulate(T=200, phi=0.6, beta=0.2, sigma_y=1.0, sigma_x=1.0, rho_x=0.7)
```

**Modelo — `SARIMAXModel`** (idéntico al exp 3.1):
```python
m = SARIMAXModel(order=(1, 0, 0), name_suffix='con X')
```

---

## Exp 3.3 — ARIMAX con dos covariables

$$Y_t = 0.6\,Y_{t-1} + 0.8\,X_{1t} + 0.4\,X_{2t} + \varepsilon_t$$
$$X_{jt} = 0.7\,X_{j,t-1} + \eta_{jt}, \quad j=1,2 \quad (\text{independientes})$$

donde $\varepsilon_t, \eta_{1t}, \eta_{2t} \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** Dos covariables independientes con efectos asimétricos ($\beta_1=0.8 > \beta_2=0.4$). El efecto combinado es mayor que en 3.1. SARIMAX estima tres parámetros adicionales ($\beta_1, \beta_2$, más la constante) con la misma muestra.

**Por qué es relevante:** Evalúa el costo de estimación cuando hay múltiples covariables y la robustez de Chronos al recibir dos covariables en el API de `past_covariates`/`future_covariates`.

**DGP — `mectesis/dgp/arimax_dgp.py` · `ARIMAX2Cov_DGP`**
```python
ARIMAX2Cov_DGP(seed=SEED).simulate(T=200, phi=0.6, beta1=0.8, beta2=0.4,
                                    sigma_y=1.0, sigma_x=1.0, rho_x=0.7)
# Retorna {"y": (T,), "X": (T, 2)}
```

**Modelo — `SARIMAXModel`**
```python
m = SARIMAXModel(order=(1, 0, 0), name_suffix='2 cov.')
# Pasa X_train (T_train, 2) y X_future (H, 2) a SARIMAX(exog=...)
# SARIMAX estima β₁ y β₂ conjuntamente vía MLE
```

---

## Exp 3.4 — VARX bivariante

$$\mathbf{Y}_t = A\,\mathbf{Y}_{t-1} + \boldsymbol{\gamma}\,X_t + \boldsymbol{\varepsilon}_t, \qquad X_t = 0.7\,X_{t-1} + \eta_t$$

$$A = \begin{pmatrix}0.5 & 0.1\\0.1 & 0.5\end{pmatrix}, \quad \boldsymbol{\gamma} = \begin{pmatrix}0.5\\0.3\end{pmatrix}, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}\!\left(\mathbf{0},\,\Sigma\right), \quad \Sigma = \begin{pmatrix}1.0 & 0.3\\0.3 & 1.0\end{pmatrix}$$

**Propiedades:** VAR(1) bivariante estacionario (valores propios de $A$: 0.6 y 0.4). Covarianza positiva entre componentes ($\Sigma_{12}=0.3$). La covariable afecta más a $Y_1$ ($\gamma_1=0.5$) que a $Y_2$ ($\gamma_2=0.3$).

**Por qué es relevante:** Único experimento multivariante del bloque. Evalúa VAR-OLS vs Chronos-2 joint en presencia de covariable escalar compartida. El VAR-OLS puede producir estimados con radio espectral $\geq 1$ en muestras pequeñas; se aplica un chequeo de estacionariedad post-OLS que descarta esos reps.

**DGP — `mectesis/dgp/varx_dgp.py` · `VARX_DGP`**
```python
VARX_DGP(seed=SEED).simulate(T=200,
    A=[[0.5,0.1],[0.1,0.5]], gamma=[0.5,0.3],
    Sigma=[[1.0,0.3],[0.3,1.0]], sigma_x=1.0, rho_x=0.7)
# Retorna {"y": (T, 2), "X": (T, 1)}
# Usa descomposición de Cholesky para innovaciones correlacionadas
# Burn-in=200
```

**Modelo — `mectesis/models/varmax_model.py` · `VARMAXModel`**
```python
# Estimación OLS ecuación por ecuación (igual que VARModel para VAR sin X)
m = VARMAXModel(order=1)

m.fit(y_train, X_train=X_train)
# Construye matriz de diseño Z (n, 4): [1, Y1_{t-1}, Y2_{t-1}, X_t]
# OLS: beta = lstsq(Z, Y[1:]) → shape (4, 2)
# Sigma_eps = (resid.T @ resid) / (n - 4)
# Chequeo post-OLS: si radio_espectral(A_estimada) >= 1.0 → fit_failed = True

y_hat = m.forecast(H, X_future=X_future)
# Pronóstico recursivo; usa X_future conocido en cada paso h

lo, hi = m.forecast_intervals(H, level=0.95, X_future=X_future)
# MSE por horizonte via recursión de matriz compañera:
# M_h = M_{h-1} + F^{h-1} · Sigma_comp · F'^{h-1}
# Intervalo h: ŷ_h ± z_{α/2} · sqrt(diag(M_h))
```

**Modelo — `mectesis/models/chronos_covariate.py` · `ChronosMultivariateCovariateModel`**
```python
# Cada componente de Y tratado como una serie 1D separada con la misma covariable
m = ChronosMultivariateCovariateModel(_chronos_base, n_covariates=1)

m.fit(y_train, X_train=X_train)  # y_train: (T_train, 2)

y_hat = m.forecast(H, X_future=X_future)
# Pasa k dicts separados al pipeline (uno por variable):
# inputs = [
#   {"target": y_train[:,0], "past_covariates": {"x0": X_train[:,0]},
#    "future_covariates": {"x0": X_future[:,0]}},
#   {"target": y_train[:,1], "past_covariates": {"x0": X_train[:,0]},
#    "future_covariates": {"x0": X_future[:,0]}},
# ]
# predict_quantiles(inputs, prediction_length=H, quantile_levels=[0.025,0.1,0.5,0.9,0.975])
# Devuelve (horizon, k)
```

---

## Exp 3.5 — ARIMAX-GARCH: covariable en media y varianza

$$Y_t = 0.4\,Y_{t-1} + 0.5\,X_t + \varepsilon_t, \qquad \varepsilon_t = \sigma_t\,z_t, \quad z_t \sim \mathcal{N}(0,1)$$
$$\sigma_t^2 = 0.1 + 0.1\,\varepsilon_{t-1}^2 + 0.75\,\sigma_{t-1}^2 + 0.1\,X_t^2$$

donde $X_t = 0.7\,X_{t-1} + \eta_t$, $\eta_t \sim \mathcal{N}(0,1)$.

**Propiedades:** La covariable entra en la ecuación de **media** ($\beta_\text{mean}=0.5$) y en la ecuación de **varianza** ($\delta_\text{var}=0.1$). Persistencia GARCH: $\alpha+\beta_\text{GARCH}=0.85$. La covariable infla la varianza condicional en períodos de alta actividad.

**Por qué es relevante:** Evalúa si SARIMAX (que solo modela la ecuación de media) es penalizado por ignorar el efecto exógeno en la varianza. Contrasta con los experimentos GARCH del bloque 1.

**DGP — `mectesis/dgp/arimax_dgp.py` · `ARIMAX_GARCH_DGP`**
```python
ARIMAX_GARCH_DGP(seed=SEED).simulate(T=200,
    phi=0.4, beta_mean=0.5, omega=0.1, alpha=0.1,
    beta_garch=0.75, delta_var=0.1, sigma_x=1.0, rho_x=0.7)
# Retorna {"y": (T,), "X": (T, 1)}
# Inicializa sigma2[0] = omega/(1-alpha-beta_garch) = 1.0
# Burn-in=500
```

**Modelo — `SARIMAXModel`**
```python
m = SARIMAXModel(order=(1, 0, 0), name_suffix='con X')
# Solo modela la ecuación de media: Y_t = φ·Y_{t-1} + β·X_t + ε_t (homosc.)
# Ignora heterocedasticidad condicional y el efecto de X_t en σ_t²
```

---

## Exp 3.6 — VECM: cointegración Y–X

$$X_t = X_{t-1} + u_t, \quad u_t \sim \mathcal{N}(0,\,\sigma_x^2)$$
$$\Delta Y_t = -0.3\,(Y_{t-1} - X_{t-1}) + \Delta X_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0,\,\sigma^2)$$

**Propiedades:** $X_t \sim I(1)$; $Y_t \sim I(1)$ pero $Y_t - X_t \sim I(0)$ (cointegrados con vector $(1,-1)$). La velocidad de ajuste es $\alpha_\text{ecm}=-0.3$ (moderada). $\Delta X_t$ aparece como control de corto plazo. El modelo ECM es el correctamente especificado; un modelo en niveles sin la relación de cointegración o en diferencias sin la corrección de error serán subóptimos.

**Por qué es relevante:** Evalúa el valor de la especificación ECM correcta frente a alternativas: SARIMAX en diferencias (pierde información de cointegración), SARIMAX en niveles (ignora la no-estacionariedad) y Chronos zero-shot (sin estructura paramétrica). También es el único experimento donde se comparan 4 modelos.

**DGP — `mectesis/dgp/adl_ecm_dgp.py` · `ADL_ECM_DGP`**
```python
ADL_ECM_DGP(seed=SEED).simulate(T=200, alpha_ecm=-0.3, sigma=1.0, sigma_x=1.0)
# Retorna {"y": (T,), "X": (T, 1)}
# Burn-in=50; inicializa Y[0]=X[0]=0
```

**Modelo 1 — `mectesis/models/ardl_model.py` · `ARDLModel`**
```python
# ECM estimado por OLS
m = ARDLModel()

m.fit(y_train, X_train=X_train)
# Construye: ΔY_t, (Y_{t-1} - X_{t-1}), ΔX_t
# OLS: [ΔY] ~ [1, Y_{t-1}-X_{t-1}, ΔX_t]
# Estima: α̂_ecm, β̂, constante

y_hat = m.forecast(H, X_future=X_future)
# Pronóstico recursivo de ΔŷT+h usando los ΔX_future conocidos y la posición de Y

# Nota: no produce intervalos ni CRPS (supports_intervals=False)
```

**Modelo 2 — `SARIMAXModel` en diferencias**
```python
m = SARIMAXModel(order=(1, 1, 0), name_suffix='dif. con X')
# d=1 → diferencia Y antes de estimar; pasa ΔX_future como exog_future
# Modelo: ΔY_t = φ·ΔY_{t-1} + β·X_t + ε_t
# Pierde la relación de cointegración en niveles
```

**Modelo 3 — `SARIMAXModel` en niveles**
```python
m = SARIMAXModel(order=(1, 0, 0), name_suffix='niv. con X')
# d=0 → no diferencia; Y e X son I(1) → regresión espuria potencial
# Pero captura implícitamente la cointegración en los coeficientes
```

---

## Chronos-2 (todos los experimentos univariantes)

**Modelo — `mectesis/models/chronos_covariate.py` · `ChronosCovariateModel`**
```python
# Exp 3.1–3.3, 3.5, 3.6: covariable(s) pasada via dict API de Chronos-2
chronos_cov1 = ChronosCovariateModel(_chronos_base, n_covariates=1)
chronos_cov2 = ChronosCovariateModel(_chronos_base, n_covariates=2,
                   cov_names=["x0", "x1"])

m.fit(y_train, X_train=X_train)  # almacena contexto

y_hat = m.forecast(H, X_future=X_future)
# pipeline.predict_quantiles(
#     inputs=[{
#         "target": y_train,
#         "past_covariates":   {"x0": X_train[:,0], ...},
#         "future_covariates": {"x0": X_future[:,0], ...}
#     }],
#     prediction_length=H,
#     quantile_levels=[0.025, 0.10, 0.50, 0.90, 0.975]
# )
# Punto: cuantil 0.5

lo, hi = m.forecast_intervals(H, level=0.95, X_future=X_future)
# 95%: cuantiles 0.025 / 0.975
# 80%: cuantiles 0.10  / 0.90
# Los cinco cuantiles se obtienen en una única llamada y se cachean
```

---

## Infraestructura común

```
mectesis/simulation/covariate_engine.py
    CovariateMonteCarloEngine     ← exps 3.1–3.3, 3.5–3.6 (Y univariante)
    CovariateMultivariateEngine   ← exp 3.4 (Y bivariante)
```

**Split train/test:**
```python
y_train, y_test = y[:T - H], y[T - H:]   # T_train = T - 24
X_train, X_future = X[:T - H], X[T - H:] # X_future conocido (oracle)
```

**Ciclo Monte Carlo:**
```python
for s in range(R):
    data = dgp.simulate(T=T, **dgp_params)
    y, X = data["y"], data["X"]
    y_train, y_test = y[:T-H], y[-H:]
    X_train, X_future = X[:T-H], X[-H:]
    for model in models:
        fkw = {"X_train": X_train} if model.supports_covariates else {}
        pkw = {"X_future": X_future} if model.supports_covariates else {}
        model.fit(y_train, **fkw)
        y_hat = model.forecast(H, **pkw)
        # acumula errores, cobertura, ancho, Winkler, CRPS
```

**Nota sobre reps descartadas (exp 3.4):** `VARMAXModel` descarta reps donde el estimado OLS tiene radio espectral $\geq 1$, retornando NaN. Las métricas de VARMAX se computan sobre el subconjunto de reps estacionarias.
