# Procesos generadores de datos — Experimentos univariados 1.1–1.19

**Experimentos 1.1–1.12:** $T \in \{200, 500\}$, $H=24$, $R=500$, semilla $=3649$.  
**Experimentos 1.13–1.19:** $T \in \{50, 200\}$, $H=24$, $R=500$, semilla $=3649$.  
**Experimentos 1.1–1.8:** $\varepsilon_t \sim \mathcal{N}(0,\sigma^2)$ i.i.d.  
**Experimentos 1.9–1.12:** $z_t \sim \mathcal{N}(0,1)$ i.i.d.; $\varepsilon_t = \sigma_t z_t$ con $\sigma_t^2$ determinada por la ecuación GARCH correspondiente.

---

## Exp 1.1 — AR(1) baja persistencia

$$Y_t = 0.3\,Y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**Propiedades:** Estacionario ($|\phi|<1$). Memoria corta: la autocorrelación decae rápido ($\rho_k = 0.3^k$). El proceso revierte a su media incondicional (0) en pocos períodos. El error de pronóstico óptimo es $\hat{Y}_{T+h} = 0.3^h Y_T \to 0$.

**Por qué es relevante:** Caso base. Evalúa si Chronos recupera la estructura AR sin estimarla explícitamente.

**DGP — `mectesis/dgp/ar.py` · `AR1`**
```python
# Librería: numpy
AR1(seed=SEED).simulate(T=200, phi=0.3)
# Bucle: y[t] = phi * y[t-1] + rng.normal(0, sigma=1)
# Inicializa en mu=0; descarta burn_in=200 observaciones
```

**Modelo — `mectesis/models/arima.py` · `ARIMAModel`**
```python
# Librería: statsmodels
m = ARIMAModel(order=(1, 0, 0))

# Estimación:
m.fit(y_train)
# Internamente: ARIMA(y_train, order=(1,0,0)).fit()
# → MLE vía filtro de Kalman; estima φ̂, σ̂²

# Pronóstico puntual:
y_hat = m.forecast(H)          # shape (H,): [ŷ_{T+1}, …, ŷ_{T+H}]
# → fitted.get_forecast(steps=H).predicted_mean

# Intervalos:
lo, hi = m.forecast_intervals(H, level=0.95)   # shape (H,) cada uno
# → fitted.get_forecast(steps=H).conf_int(alpha=0.05)[["lower y","upper y"]]
```

---

## Exp 1.2 — AR(1) alta persistencia

$$Y_t = 0.9\,Y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**Propiedades:** Estacionario pero cercano a la raíz unitaria. La varianza incondicional es $\sigma^2/(1-\phi^2) \approx 5.3$. La reversión a la media es muy lenta; en muestras finitas el proceso parece no estacionario. El error de pronóstico óptimo crece hasta $\sigma^2/(1-\phi^2)$ conforme $h \to \infty$.

**Por qué es relevante:** Distingue modelos que ajustan bien la persistencia de los que la subestiman o confunden con I(1).

**DGP — `AR1`**
```python
AR1(seed=SEED).simulate(T=200, phi=0.9)
# Mismo código que 1.1; solo cambia phi
```

**Modelo — `ARIMAModel`**
```python
m = ARIMAModel(order=(1, 0, 0))
m.fit(y_train)
y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)
# Ídem exp 1.1; φ̂ estimado ≈ 0.9 → intervalos se ensanchan rápido con h
```

---

## Exp 1.3 — Random Walk sin drift — I(1)

$$Y_t = Y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**Propiedades:** No estacionario. Varianza crece linealmente con $t$. El pronóstico óptimo para todo horizonte es $\hat{Y}_{T+h} = Y_T$ (el último valor). Diferencia fundamental con 1.1/1.2: **no existe media de largo plazo**.

**Por qué es relevante:** Si un modelo aplica AR estacionario a un I(1), sobreestima la reversión a la media y tiene RMSE creciente con h.

**DGP — `mectesis/dgp/rw.py` · `RandomWalk`**
```python
# Librería: numpy
RandomWalk(seed=SEED).simulate(T=200, drift=0.0)
# Bucle: y[t] = drift + y[t-1] + eps[t]
# Sin burn-in (proceso no estacionario)
```

**Modelo — `ARIMAModel`**
```python
m = ARIMAModel(order=(0, 1, 0))
m.fit(y_train)
# Internamente: ARIMA(y_train, order=(0,1,0)).fit()
# Diferencia la serie antes de estimar: Δy_t ~ WN(0, σ²)
# Pronóstico: ŷ_{T+h} = y_T para todo h (sin constante)

y_hat = m.forecast(H)          # array constante = y_train[-1]
lo, hi = m.forecast_intervals(H, level=0.95)
# Ancho del intervalo crece como sqrt(h)
```

---

## Exp 1.4 — Random Walk con drift — I(1) con tendencia

$$Y_t = 0.5 + Y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**Propiedades:** I(1) con deriva determinista. El pronóstico óptimo es $\hat{Y}_{T+h} = Y_T + 0.5\,h$. La tendencia es **estocástica** (acumulación de choques) más **determinista** (drift constante).

**Por qué es relevante:** Evalúa si los modelos capturan el drift. ARIMA(0,1,0) con constante es el modelo correcto; sin constante subestima sistemáticamente (sesgo $\approx 0.5 h$ en h).

**DGP — `RandomWalk`**
```python
RandomWalk(seed=SEED).simulate(T=200, drift=0.5)
```

**Modelo — `ARIMAModel`**
```python
m = ARIMAModel(order=(0, 1, 0))
m.fit(y_train)
# ARIMA(0,1,0) sin constante → NO captura el drift
# → sesgo sistemático = 0.5*h en cada horizonte
# El modelo correcto requeriría trend='c' (constante post-diferenciación)

y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)
```

---

## Exp 1.5 — AR(1) con tendencia lineal determinista

$$Y_t = 5 + 0.1\,t + 0.6\,Y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**Propiedades:** **Estacionario alrededor de una tendencia lineal** (diferencia clave con 1.4). La media condicional crece, pero los choques no se acumulan. El proceso es I(0) con media $\mu_t = (5 + 0.1t)/(1-0.6)$. Un modelo AR sin tendencia estima parámetros contaminados y produce pronósticos sesgados a largo plazo.

**Por qué es relevante:** Distingue tendencia determinista (exp 1.5) de tendencia estocástica (exp 1.4).

**DGP — `mectesis/dgp/ar_trend.py` · `AR1WithTrend`**
```python
# Librería: numpy
AR1WithTrend(seed=SEED).simulate(T=200, intercept=5.0, trend_coeff=0.1, phi=0.6)
# Bucle: y[t] = intercept + trend_coeff*t_actual + phi*y[t-1] + eps[t]
# Durante burn_in=50, t_actual es negativo para anclar el AR pre-período visible
```

**Modelo — `mectesis/models/arima_ext.py` · `ARIMAWithTrendModel`**
```python
# Librería: statsmodels
m = ARIMAWithTrendModel(order=(1, 0, 0), trend='ct')
m.fit(y_train)
# Internamente: ARIMA(y_train, order=(1,0,0), trend='ct').fit()
# trend='ct' agrega constante + tendencia lineal al modelo de estado
# → estima intercepto, coeficiente de tendencia, y φ

y_hat = m.forecast(H)
# La tendencia se extrapola automáticamente h períodos adelante:
# ŷ_{T+h} = â + b̂*(T+h) + φ̂*ŷ_{T+h-1}

lo, hi = m.forecast_intervals(H, level=0.95)
```

---

## Exp 1.6 — Seasonal AR, trimestral (s=4)

$$(1 - 0.5\,L)(1 - 0.3\,L^4)\,Y_t = \varepsilon_t$$

Expandido: $Y_t = 0.5\,Y_{t-1} + 0.3\,Y_{t-4} - 0.15\,Y_{t-5} + \varepsilon_t$

**Propiedades:** Estacionario. Estacionalidad **estocástica**: el patrón estacional tiene amplitud y fase que varían entre realizaciones. El espectro tiene un pico en la frecuencia $\omega = 2\pi/4$ pero no es una línea discreta. El proceso tiene autocorrelaciones significativas en rezagos 1, 4 y 5.

**Por qué es relevante:** Evalúa si SARIMA y Chronos identifican la estructura estacional sin diferenciar.

**DGP — `mectesis/dgp/seasonal.py` · `SeasonalDGP`**
```python
# Librería: numpy
SeasonalDGP(seed=SEED).simulate(T=200, phi=0.5, Phi=0.3, s=4, integrated=False)
# Bucle: y[t] = phi*y[t-1] + Phi*y[t-4] - phi*Phi*y[t-5] + eps[t]
# Burn-in=200; requiere s+1 rezagos de padding inicial
```

**Modelo — `mectesis/models/sarima_model.py` · `SARIMAModel`**
```python
# Librería: statsmodels
m = SARIMAModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 4))
m.fit(y_train)
# Internamente:
# SARIMAX(y_train, order=(1,0,0), seasonal_order=(1,0,0,4),
#         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

y_hat = m.forecast(H)
# → fitted.forecast(steps=H)

lo, hi = m.forecast_intervals(H, level=0.95)
# → fitted.get_forecast(steps=H).conf_int(alpha=0.05)
```

---

## Exp 1.7 — Seasonal I(1)×I(1)₁₂, mensual

$$(1-L)(1-L^{12})\,Y_t = \varepsilon_t$$

**Propiedades:** Doblemente integrado: integración regular (diferencia de orden 1) y estacional (diferencia estacional de orden 12). La varianza crece tanto por tendencia como por estacionalidad acumulada. Representa series económicas típicas como ventas minoristas o producción industrial.

**Por qué es relevante:** SARIMA(0,1,0)(0,1,0)₁₂ es el modelo correcto. Si no se aplica diferenciación estacional, los residuos tienen autocorrelación estacional fuerte.

**DGP — `SeasonalDGP` con `integrated=True`**
```python
SeasonalDGP(seed=SEED).simulate(T=200, s=12, integrated=True)
# Bucle: y[t] = y[t-1] + y[t-12] - y[t-13] + eps[t]
# No estacionario → sin burn-in; inicializa con s+1 ceros de padding
```

**Modelo — `SARIMAModel`**
```python
m = SARIMAModel(order=(0, 1, 0), seasonal_order=(0, 1, 0, 12))
m.fit(y_train)
# d=1, D=1 → aplica (1-L)(1-L^12) internamente antes de estimar

y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)
```

---

## Exp 1.8 — AR(1) con quiebre estructural en T/2

$$Y_t = \begin{cases} 0.3\,Y_{t-1} + \varepsilon_t & t < T/2 \\ 0.8\,Y_{t-1} + \varepsilon_t & t \geq T/2 \end{cases}$$

**Propiedades:** Dos regímenes estacionarios con persistencias muy distintas. Un modelo AR estimado sobre toda la muestra produce $\hat\phi$ contaminado entre 0.3 y 0.8, subestimando la persistencia post-quiebre. El período de pronóstico (últimas 24 obs) es siempre post-quiebre.

**Por qué es relevante:** Evalúa robustez ante no-estacionariedad paramétrica. El modelo con dummy exógena es el correcto especificado; Chronos no tiene mecanismo explícito para detectar el quiebre.

**DGP — `mectesis/dgp/ar_break.py` · `AR1WithBreak`**
```python
# Librería: numpy
AR1WithBreak(seed=SEED).simulate(T=200, phi_before=0.3, phi_after=0.8)
# Bucle: phi = phi_before if t < break_idx else phi_after
# break_idx calculado sobre serie extendida (T + burn_in) para que el
# quiebre visible quede en T//2 de la serie retornada
```

**Modelo — `mectesis/models/arima_ext.py` · `ARIMAWithBreakModel`**
```python
# Librería: statsmodels (SARIMAX con regresor exógeno)
m = ARIMAWithBreakModel(order=(1, 0, 0), T_total=200, break_fraction=0.5)
m.fit(y_train)
# Internamente:
#   break_idx  = int(T_total * break_fraction)   # = 100
#   break_exog = (np.arange(len(y_train)) >= break_idx).astype(float)[:,None]
#   SARIMAX(y_train, order=(1,0,0), exog=break_exog).fit(disp=False)
# Dummy: 0 antes del quiebre, 1 desde el quiebre

y_hat = m.forecast(H)
# exog_future = np.ones((H, 1))   ← todo el forecast es post-quiebre
# fitted.forecast(steps=H, exog=exog_future)

lo, hi = m.forecast_intervals(H, level=0.95)
```

---

## Exp 1.9 — AR(1)–ARCH(1)

$$Y_t = 0.3\,Y_{t-1} + \varepsilon_t, \qquad \sigma_t^2 = 0.1 + 0.3\,\varepsilon_{t-1}^2$$

donde $\varepsilon_t = \sigma_t z_t$, $z_t \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** Media AR(1) con varianza heteroscedástica de baja persistencia. Varianza incondicional $\bar\sigma^2 = \omega/(1-\alpha) = 0.1/0.7 \approx 0.143$ → $\sigma_\varepsilon \approx 0.38$. Sin término $\beta$: tras un choque, $\sigma_t^2$ regresa al nivel incondicional en un único período. El proceso no tiene clustering de volatilidad prolongado.

**Por qué es relevante:** Introduce heterocedasticidad simple. Evalúa si el modelo ARCH la captura y si Chronos la ignora sin penalización en RMSE.

**DGP — `mectesis/dgp/garch.py` · `AR1ARCH`**
```python
# Librería: numpy
AR1ARCH(seed=SEED).simulate(T=200, phi=0.3, omega=0.1, alpha=0.3)
# Inicialización: sigma2[0] = omega/(1-alpha) = 0.143
# Bucle (burn_in=500):
#   sigma2[t] = omega + alpha * eps[t-1]**2
#   eps[t]    = sqrt(sigma2[t]) * z[t]
#   y[t]      = phi * y[t-1] + eps[t]
```

**Modelo — `mectesis/models/garch_model.py` · `ARARCHModel`**
```python
# Librería: arch (Kevin Sheppard)
m = ARARCHModel()   # ar_lags=1, p=1; q=0 fijo → ARCH(1) = GARCH(1,0)
m.fit(y_train)
# Internamente (escala ×100 por estabilidad numérica):
#   arch_model(y_train*100, mean='AR', lags=1,
#              vol='GARCH', p=1, q=0, rescale=False).fit(disp='off')
# Estima: constante AR, φ̂, ω̂, α̂ vía MLE (BHHH / L-BFGS-B)

y_hat = m.forecast(H)
# result.forecast(horizon=H, reindex=False).mean.values[-1] / 100
# shape (H,): pronóstico puntual del componente AR (φ̂^h * y_{T})

lo, hi = m.forecast_intervals(H, level=0.95)
# var_fc = result.forecast(...).variance.values[-1] / 10000  → (H,)
# std_fc = sqrt(max(var_fc, 0))
# lo, hi = mean_fc ∓ z_{0.975} * std_fc
# Nota: sigma_h^2 → sigma_incondicional^2 conforme h crece (sin clustering)
```

---

## Exp 1.10 — AR(1)–GARCH(1,1), alta persistencia

$$Y_t = 0.3\,Y_{t-1} + \varepsilon_t, \qquad \sigma_t^2 = 0.1 + 0.1\,\varepsilon_{t-1}^2 + 0.8\,\sigma_{t-1}^2$$

donde $\varepsilon_t = \sigma_t z_t$, $z_t \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** Persistencia $\alpha + \beta = 0.9$ → la volatilidad condicional decae lentamente. Varianza incondicional $\bar\sigma^2 = \omega/(1-\alpha-\beta) = 0.1/0.1 = 1$ → $\sigma_\varepsilon = 1$. El proceso tiene clustering de volatilidad pronunciado: los períodos de alta volatilidad tienden a agruparse. $\text{Var}(Y_t) = \bar\sigma^2/(1-\phi^2) \approx 1.10$.

**Por qué es relevante:** Evalúa si la alta persistencia de la volatilidad penaliza más a Chronos que a GARCH en la construcción de intervalos. El pronóstico de nivel es idéntico para ambos (mismo AR); la diferencia se concentra en los intervalos.

**DGP — `mectesis/dgp/garch.py` · `AR1GARCH`**
```python
AR1GARCH(seed=SEED).simulate(T=200, phi=0.3, omega=0.1, alpha=0.1, beta=0.8)
# Inicialización: sigma2[0] = omega/(1-alpha-beta) = 1.0
# Bucle (burn_in=500):
#   sigma2[t] = omega + alpha*eps[t-1]**2 + beta*sigma2[t-1]
#   eps[t]    = sqrt(sigma2[t]) * z[t]
#   y[t]      = phi * y[t-1] + eps[t]
```

**Modelo — `mectesis/models/garch_model.py` · `ARGARCHModel`**
```python
# Librería: arch
m = ARGARCHModel()   # ar_lags=1, p=1, q=1
m.fit(y_train)
# Internamente:
#   arch_model(y_train*100, mean='AR', lags=1,
#              vol='GARCH', p=1, q=1, rescale=False).fit(disp='off')
# Estima: φ̂, ω̂, α̂, β̂

y_hat = m.forecast(H)
# result.forecast(horizon=H, reindex=False).mean.values[-1] / 100

lo, hi = m.forecast_intervals(H, level=0.95)
# var_fc = result.forecast(...).variance.values[-1] / 10000
# Con β̂≈0.8, la varianza condicional proyectada decae hacia sigma^2_incondicional
# conforme h crece: sigma_h^2 = sigma^2_incond*(1 - (α+β)^h) + sigma_T^2*(α+β)^h
```

---

## Exp 1.11 — GARCH(1,1) media cero

$$Y_t = \sigma_t\,z_t, \qquad \sigma_t^2 = 0.1 + 0.1\,Y_{t-1}^2 + 0.8\,\sigma_{t-1}^2$$

donde $z_t \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** Sin componente AR. La media condicional es cero en todo horizonte: $\mathbb{E}[Y_{T+h} | \mathcal{F}_T] = 0$ para todo $h \geq 1$. La única información relevante es la varianza condicional actual $\sigma_T^2$, que afecta el ancho de los intervalos pero no el pronóstico de nivel. Persistencia $\alpha+\beta=0.9$; $\bar\sigma^2=1$.

**Por qué es relevante:** Caso límite: ambos modelos deben pronosticar 0. La diferencia entre GARCH y Chronos es puramente en la calibración de intervalos. Mide si Chronos infla innecesariamente los intervalos cuando la media es cero conocida.

**DGP — `mectesis/dgp/garch.py` · `PureGARCH`**
```python
PureGARCH(seed=SEED).simulate(T=200, omega=0.1, alpha=0.1, beta=0.8)
# Inicialización: sigma2[0] = omega/(1-alpha-beta) = 1.0
# Bucle (burn_in=500):
#   sigma2[t] = omega + alpha*y[t-1]**2 + beta*sigma2[t-1]
#   y[t]      = sqrt(sigma2[t]) * z[t]
# Nota: usa y[t-1]^2 directamente (no hay eps separado)
```

**Modelo — `mectesis/models/garch_model.py` · `GARCHModel`**
```python
# Librería: arch
m = GARCHModel()   # p=1, q=1, mean='Zero'
m.fit(y_train)
# Internamente:
#   arch_model(y_train*100, mean='Zero',
#              vol='GARCH', p=1, q=1, rescale=False).fit(disp='off')
# mean='Zero' → no estima media; solo estima ω̂, α̂, β̂

y_hat = m.forecast(H)
# Pronóstico puntual: array de ceros para todo h (media cero exacta)

lo, hi = m.forecast_intervals(H, level=0.95)
# var_fc = result.forecast(...).variance.values[-1] / 10000
# Los intervalos se construyen en torno a 0:
#   lo = -z_{0.975} * sqrt(var_fc_h)
#   hi = +z_{0.975} * sqrt(var_fc_h)
# Ancho aumenta con sigma_T^2 actual y converge a sigma^2_incond conforme h crece
```

---

## Exp 1.12 — AR(1)–GJR–GARCH(1,1,1), efecto leverage

$$Y_t = 0.3\,Y_{t-1} + \varepsilon_t$$
$$\sigma_t^2 = 0.1 + 0.05\,\varepsilon_{t-1}^2 + 0.1\,\varepsilon_{t-1}^2\,\mathbf{1}\{\varepsilon_{t-1}<0\} + 0.8\,\sigma_{t-1}^2$$

donde $\varepsilon_t = \sigma_t z_t$, $z_t \sim \mathcal{N}(0,1)$ i.i.d.

**Propiedades:** Extiende GARCH con asimetría: los shocks negativos ($\varepsilon_{t-1}<0$) aumentan la volatilidad en $\alpha+\gamma=0.15$, mientras que los positivos solo en $\alpha=0.05$. Persistencia efectiva $\alpha + \gamma/2 + \beta = 0.9$. Varianza incondicional $\bar\sigma^2=1$. El efecto leverage imita lo observado en activos financieros (caídas $\to$ mayor volatilidad).

**Por qué es relevante:** Evalúa si el modelo GJR captura la asimetría y si eso se traduce en ventaja frente a Chronos. En RMSE la diferencia es mínima (la asimetría afecta la varianza, no la media); la ventaja aparece en calibración de intervalos.

**DGP — `mectesis/dgp/garch.py` · `AR1GJRGARCH`**
```python
AR1GJRGARCH(seed=SEED).simulate(T=200, phi=0.3, omega=0.1,
                                 alpha=0.05, gamma=0.1, beta=0.8)
# Inicialización: sigma2[0] = omega/(1 - alpha - gamma/2 - beta) = 1.0
# Bucle (burn_in=500):
#   ind       = 1.0 if eps[t-1] < 0.0 else 0.0
#   sigma2[t] = omega + alpha*eps[t-1]**2 + gamma*eps[t-1]**2*ind + beta*sigma2[t-1]
#   eps[t]    = sqrt(sigma2[t]) * z[t]
#   y[t]      = phi * y[t-1] + eps[t]
```

**Modelo — `mectesis/models/garch_model.py` · `ARGJRGARCHModel`**
```python
# Librería: arch
m = ARGJRGARCHModel()   # ar_lags=1, p=1, o=1, q=1
m.fit(y_train)
# Internamente:
#   arch_model(y_train*100, mean='AR', lags=1,
#              vol='GARCH', p=1, o=1, q=1, rescale=False).fit(disp='off')
# El parámetro o=1 añade el término asimétrico γ*eps^2*1{eps<0}
# Estima: φ̂, ω̂, α̂, γ̂, β̂

y_hat = m.forecast(H)
# result.forecast(horizon=H, reindex=False).mean.values[-1] / 100
# Pronóstico de nivel igual que GARCH simétrico (mismo componente AR)

lo, hi = m.forecast_intervals(H, level=0.95)
# var_fc proyectada usando la recursión GJR:
# sigma_h^2 | GJR incorpora el impacto asimétrico del shock más reciente en h=1
# Para h>1, converge a la varianza incondicional (como GARCH)
```

---

## Exp 1.13 — Nivel local (Local Level → ETS(A,N,N))

$$\ell_t = \ell_{t-1} + \eta_t,\quad \eta_t \sim \mathcal{N}(0,\sigma_\eta^2)$$
$$Y_t = \ell_t + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2)$$

**Propiedades:** Equivalente a ARIMA(0,1,1). El nivel evoluciona como un random walk; las observaciones son el nivel más ruido de observación. El cociente $q = \sigma_\eta^2/\sigma_\varepsilon^2$ determina el parámetro $\alpha$ óptimo del suavizado exponencial. El pronóstico óptimo es $\hat{Y}_{T+h} = \hat{\ell}_T$ para todo $h$ — constante a cualquier horizonte.

**Por qué es relevante:** ETS(A,N,N) es el modelo de suavizado exponencial simple. Su equivalencia exacta con ARIMA(0,1,1) hace que la estimación por MLE y el suavizado sean equivalentes. Evalúa si Chronos captura el nivel local sin modelo paramétrico.

**DGP — `mectesis/dgp/ets_dgps.py` · `LocalLevelDGP`**
```python
LocalLevelDGP(seed=SEED).simulate(T=50, sigma_eps=1.0, sigma_eta=0.3, l0=0.0)
# level[0] = l0
# Bucle:
#   level[t+1] = level[t] + eta[t],  eta[t] ~ N(0, sigma_eta^2)
#   Y[t]       = level[t+1] + eps[t], eps[t] ~ N(0, sigma_eps^2)
```

**Modelo — `mectesis/models/ets_model.py` · `ETSModel`**
```python
m = ETSModel(trend=None)  # ETS(A,N,N)
m.fit(y_train)
# Internamente:
#   ExponentialSmoothing(y_train, trend=None,
#       initialization_method='estimated').fit(optimized=True)
# Estima α y ℓ̂_0 por MLE

y_hat = m.forecast(H)   # self._result.forecast(H)

lo, hi = m.forecast_intervals(H, level=0.95)
# sims = result.simulate(nsimulations=H, anchor='end', repetitions=500)
# lo = quantile(sims, 0.025, axis=1)
# hi = quantile(sims, 0.975, axis=1)
```

---

## Exp 1.14 — Tendencia local (Local Trend → ETS(A,A,N))

$$\ell_t = \ell_{t-1} + b_{t-1} + \eta_t,\quad \eta_t \sim \mathcal{N}(0,\sigma_\eta^2)$$
$$b_t = b_{t-1} + \zeta_t,\quad \zeta_t \sim \mathcal{N}(0,\sigma_\zeta^2)$$
$$Y_t = \ell_t + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2)$$

**Propiedades:** Equivalente a ARIMA(0,2,2). Nivel y tendencia son ambos random walks independientes (doble integración estocástica). La incertidumbre predictiva crece cuadráticamente con $h$. El pronóstico óptimo es $\hat{Y}_{T+h} = \hat{\ell}_T + h\,\hat{b}_T$ pero con varianza que se acumula en dos fuentes.

**Por qué es relevante:** El DGP de tendencia local genera la mayor incertidumbre del bloque state-space. Evalúa si Chronos captura la acumulación de incertidumbre de nivel más tendencia y si ETS la calibra correctamente en muestras cortas.

**DGP — `LocalTrendDGP`**
```python
LocalTrendDGP(seed=SEED).simulate(T=50, sigma_eps=1.0, sigma_eta=0.2,
                                   sigma_zeta=0.1, l0=0.0, b0=0.1)
# level[0] = l0; b[0] = b0
# Bucle:
#   level[t+1] = level[t] + b[t] + eta[t]
#   b[t+1]     = b[t] + zeta[t]
#   Y[t]       = level[t+1] + eps[t]
```

**Modelo — `ETSModel`**
```python
m = ETSModel(trend='add')  # ETS(A,A,N)
m.fit(y_train)
# ExponentialSmoothing(y_train, trend='add',
#     initialization_method='estimated').fit(optimized=True)
# Estima α, β y estados iniciales

y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)  # simulación, igual que 1.13
```

---

## Exp 1.15 — Tendencia amortiguada (Damped Trend → ETS(A,Ad,N))

$$\ell_t = \ell_{t-1} + \phi\,b_{t-1} + \eta_t,\quad \eta_t \sim \mathcal{N}(0,\sigma_\eta^2)$$
$$b_t = \phi\,b_{t-1} + \zeta_t,\quad \zeta_t \sim \mathcal{N}(0,\sigma_\zeta^2),\quad \phi = 0.9$$

**Propiedades:** La tendencia decae geométricamente con tasa $\phi=0.9$. El pronóstico converge a un límite finito $\hat{\ell}_T + \hat{b}_T\,\phi/(1-\phi)$ en lugar de diverger linealmente. Esto lo hace más realista que la tendencia local para series económicas que no crecen indefinidamente.

**Por qué es relevante:** El parámetro de amortiguamiento $\phi$ requiere suficientes datos para ser bien estimado. Con muestras cortas la estimación de $\phi$ es imprecisa, haciendo que ETS sea menos fiable que en 1.13–1.14. Evalúa el umbral de $T$ a partir del cual la estimación de $\phi$ estabiliza el modelo.

**DGP — `DampedTrendDGP`**
```python
DampedTrendDGP(seed=SEED).simulate(T=50, phi=0.9, sigma_eps=1.0,
                                    sigma_eta=0.2, sigma_zeta=0.1,
                                    l0=0.0, b0=0.1)
# level[0] = l0; b[0] = b0
# Bucle:
#   level[t+1] = level[t] + phi*b[t] + eta[t]
#   b[t+1]     = phi*b[t] + zeta[t]
#   Y[t]       = level[t+1] + eps[t]
```

**Modelo — `ETSModel`**
```python
m = ETSModel(trend='add', damped_trend=True)  # ETS(A,Ad,N)
m.fit(y_train)
# ExponentialSmoothing(y_train, trend='add', damped_trend=True,
#     initialization_method='estimated').fit(optimized=True)
# Estima α, β, φ y estados iniciales

y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)
```

---

## Exp 1.16 — Estacionalidad determinística (s=12)

$$Y_t = \mu + s_{t \bmod 12} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2),\quad \sum_{j=0}^{11} s_j = 0$$

**Propiedades:** El patrón estacional es **fijo** — la misma forma se repite exactamente en cada ciclo. El estimador óptimo de $s_j$ es la media histórica de las observaciones del mes $j$ (OLS con dummies estacionales). Seasonal Naive usa solo el último ciclo observado: es ineficiente pero consistente.

**Por qué es relevante:** Contrasta con exp 1.17: la diferencia clave es la naturaleza de la estacionalidad. Con estacionalidad determinística, más contexto beneficia a cualquier estimador porque el patrón no varía. Con estacionalidad estocástica (1.17), más contexto puede perjudicar. Evalúa si Chronos promedia implícitamente sobre ciclos para explotar la invarianza del patrón.

**DGP — `mectesis/dgp/ets_dgps.py` · `DeterministicSeasonalDGP`**
```python
DeterministicSeasonalDGP(seed=SEED).simulate(T=50, mu=5.0, sigma_eps=1.0, s=12)
# Patrón: seno discreto sin(2π·j/s) para j=0,...,s-1, normalizado a suma-cero
# Y[t] = mu + s[t % s] + eps[t]
```

**Modelo — `mectesis/models/naive.py` · `SeasonalNaiveModel`**
```python
m = SeasonalNaiveModel(period=12)
m.fit(y_train)
# Almacena y_train; pronóstico = y_train[-s + ((h-1) % s)]

y_hat = m.forecast(H)
# No produce intervalos de predicción (supports_intervals=False)
# Las métricas probabilísticas (cov_80, cov_95, winkler, crps)
# solo están disponibles para Chronos en este experimento
```

---

## Exp 1.17 — Seasonal random walk (s=12)

$$Y_t = Y_{t-12} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma^2)$$

**Propiedades:** No estacionario; raíz unitaria estacional exacta. Equivalente a SARIMA(0,0,0)(0,1,0)₁₂. La varianza condicional crece con el horizonte: $\text{Var}(Y_{T+h}|\mathcal{F}_T) = \lfloor h/12 \rfloor \cdot \sigma^2$. El pronóstico óptimo es $\hat{Y}_{T+h} = Y_{T+h-12}$ — exactamente lo que produce Seasonal Naive.

**Por qué es relevante:** La no-estacionariedad estacional es el caso adverso clave para Chronos identificado en el bloque. Con más contexto, Chronos no converge al pronóstico óptimo; en cambio tiende a sobreajustar la estructura temporal de la serie no estacionaria.

**DGP — `SeasonalRandomWalkDGP`**
```python
SeasonalRandomWalkDGP(seed=SEED).simulate(T=50, s=12, sigma=1.0, burn_in=100)
# Inicializa Y[0]...Y[s-1] = 0 (burn_in de 100 períodos)
# Bucle: Y[s + t] = Y[t] + eps[t],  eps ~ N(0, sigma^2)
# Retorna los últimos T puntos
```

**Modelo — `SeasonalNaiveModel`**
```python
m = SeasonalNaiveModel(period=12)
m.fit(y_train)
y_hat = m.forecast(H)   # y_hat[h] = y_train[-12 + ((h-1) % 12)]
# No produce intervalos de predicción
```

---

## Exp 1.18 — Tendencia + estacionalidad estocástica (ETS(A,A,A))

$$\ell_t = \ell_{t-1} + b_{t-1} + \eta_t,\quad \eta_t \sim \mathcal{N}(0,\sigma_\eta^2)$$
$$b_t = b_{t-1} + \zeta_t,\quad \zeta_t \sim \mathcal{N}(0,\sigma_\zeta^2)$$
$$\gamma_t = \gamma_{t-12} + \omega_t,\quad \omega_t \sim \mathcal{N}(0,\sigma_\omega^2)$$
$$Y_t = \ell_t + \gamma_t + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2)$$

**Propiedades:** El DGP más complejo del bloque: nivel, tendencia y estacionalidad evolucionan estocásticamente e independientemente. La incertidumbre predictiva acumula contribuciones de los tres estados. El modelo ETS(A,A,A) es el correctamente especificado y el único que separa explícitamente los tres componentes.

**Por qué es relevante:** Evalúa si Chronos puede capturar simultáneamente tendencia y estacionalidad estocásticas sin model paramétrico. La suma de tres fuentes de incertidumbre genera el mayor spread de errores del bloque.

**DGP — `mectesis/dgp/ets_dgps.py` · `LocalLevelSeasonalDGP`**
```python
LocalLevelSeasonalDGP(seed=SEED).simulate(
    T=50, sigma_eps=0.5, sigma_eta=0.1, sigma_zeta=0.05,
    sigma_omega=0.1, l0=5.0, b0=0.1, s=12)
# Inicialización estacional: seno discreto normalizado a suma-cero
# Bucle:
#   level[t+1] = level[t] + b[t] + eta[t]
#   b[t+1]     = b[t] + zeta[t]
#   gamma[t+1] = gamma[t-s+1] + omega[t]
#   Y[t]       = level[t+1] + gamma[(t+1) % s] + eps[t]
```

**Modelo — `ETSModel`**
```python
m = ETSModel(trend='add', seasonal='add', seasonal_periods=12)  # ETS(A,A,A)
m.fit(y_train)
# ExponentialSmoothing(y_train, trend='add', seasonal='add',
#     seasonal_periods=12, initialization_method='estimated').fit(optimized=True)
# Estima α, β, γ y los 12 estados estacionales iniciales

y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)
# Simulación de 500 trayectorias (mismo mecanismo que 1.13–1.15)
```

---

## Exp 1.19 — Tendencia lineal pura (Theta)

$$Y_t = 0.1\,t + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**Propiedades:** Tendencia lineal exactamente determinista, sin componente AR residual. El modelo óptimo es regresión OLS sobre $t$. El proceso es I(0) con media creciente: $\mathbb{E}[Y_t] = 0.1\,t$. Theta (con $\theta=2$) genera dos líneas a partir de la serie: una réplica (línea 1) y una con pendiente ampliada (línea 2 del theta). El pronóstico Theta es la media de ambas, lo que introduce sesgo positivo sistemático respecto al slope real.

**Por qué es relevante:** Evalúa Theta fuera de su habitat natural (series M-Competition con complejidad mixta) en un DGP de estructura simple. El sesgo estructural de Theta en este DGP es un resultado analíticamente interesante. Con muestras cortas la varianza domina al sesgo y Theta supera a Chronos; con muestras largas Chronos aprende el slope y domina.

**DGP — `mectesis/dgp/ar_trend.py` · `AR1WithTrend`**
```python
AR1WithTrend(seed=SEED).simulate(T=50, intercept=0.0, trend_coeff=0.1, phi=0.0)
# phi=0.0 → tendencia lineal pura sin componente AR
# Bucle: y[t] = intercept + trend_coeff * t + phi * y[t-1] + eps[t]
#        con phi=0: y[t] = 0.1 * t + eps[t]
```

**Modelo — `mectesis/models/theta_model.py` · `ThetaModel`**
```python
from statsmodels.tsa.forecasting.theta import ThetaModel as _Theta

m = ThetaModel()
m.fit(y_train)
# Internamente: _Theta(y_train, deseasonalize=False).fit(disp=False)
# El método Theta descompone la serie en dos líneas theta;
# el pronóstico es la media de ambas proyecciones

y_hat = m.forecast(H)   # self._result.forecast(H)

lo, hi = m.forecast_intervals(H, level=0.95)
# pi = self._result.prediction_intervals(H, alpha=0.05)
# lo = pi['lower'], hi = pi['upper']
# CRPS: crps_gaussian(y_true, mu, sigma) con sigma derivado del IP 95%
```

---

## Notas metodológicas

### ¿Hay otras variantes ARMA para modelar estos DGPs?

Sí. Lo implementado son los modelos **correctamente especificados** (o el más parsimonioso dentro de la familia). Variantes razonables:

| Exp | Modelo adicional | Diferencia |
|-----|-----------------|-----------|
| 1.1/1.2 | ARMA(1,1) | Agrega componente MA; sobreajuste si el DGP es AR puro |
| 1.3/1.4 | ARIMA(1,1,0) | Incluye AR post-diferenciación; agrega parámetro innecesario |
| 1.5 | ARIMAX con $t$ como regresor explícito | Equivalente a `trend='ct'` pero más transparente |
| 1.6 | AR(5) puro | Captura rezagos 1 y 4 sin factorizar; pierde parsimonia |
| 1.6 | SARMA(1,1)(1,1)₄ | Sobreparametrizado para este DGP |
| 1.7 | SARIMA(1,1,0)(1,1,0)₁₂ | Correcto si hubiera AR residual; aquí innecesario |
| 1.8 | AR(1) sin dummy | Modelo "ingenuo" contaminado — útil como benchmark negativo |
| 1.8 | Markov-switching AR | Correcto pero no identificable sin estimar número de regímenes |
| 1.9 | GARCH(1,1) | Añade β innecesario; ARCH(1) es parsimonioso aquí |
| 1.10/1.12 | EGARCH | Alternativa al GJR para capturar asimetría |

### ¿Por qué la estacionalidad no es una onda sin/cos?

Hay dos paradigmas para modelar estacionalidad:

**Determinista (Fourier):** $Y_t = \alpha\,\sin(2\pi t/s) + \beta\,\cos(2\pi t/s) + \varepsilon_t$  
El patrón estacional es **fijo**: misma amplitud y fase en cada año. Apropiado si la estacionalidad es una constante estructural del fenómeno (ej: temperatura diaria). Usado en ARIMAX con términos de Fourier y en Prophet.

**Estocástica (SARIMA):** $(1 - \Phi L^s)Y_t = \varepsilon_t$  
El patrón estacional **evoluciona**: la amplitud y la fase derivan lentamente con los choques acumulados. Apropiado para series económicas donde el comportamiento estacional cambia año a año (ej: ventas navideñas que crecen o cambian de forma).

Los experimentos 1.6 y 1.7 usan estacionalidad **estocástica** porque:
1. Es el supuesto estándar en el paradigma Box-Jenkins, aplicable a series económicas
2. El modelo SARIMA es el correcto especificado bajo ese DGP — el test de poder es justo
3. Con estacionalidad determinista (Fourier), el DGP y el modelo correcto serían una regresión lineal, no SARIMA

---

## Infraestructura común

```
numpy.random.default_rng(seed)       ← generador PCG64; self.rng en BaseDGP
mectesis/dgp/base.py    BaseDGP      ← abstracta; expone self.rng y simulate()
mectesis/models/base.py BaseModel    ← abstracta; fit / forecast / forecast_intervals
mectesis/simulation/engine.py        ← MonteCarloEngine.run_monte_carlo()
```

El ciclo Monte Carlo en `engine.py`:
```python
for s in range(R):
    y = dgp.simulate(T=T, **dgp_params)       # genera serie completa
    y_train, y_test = y[:-H], y[-H:]          # split fijo
    for model in models:
        model.fit(y_train)                    # estima sobre y_train
        y_hat      = model.forecast(H)        # shape (H,)
        lo, hi     = model.forecast_intervals(H, level=0.80)
        lo95, hi95 = model.forecast_intervals(H, level=0.95)
        # acumula bias, varianza, cobertura, ancho por horizonte
```

### Escalado en modelos ARCH/GARCH

Los modelos `ARARCHModel`, `ARGARCHModel`, `GARCHModel`, `ARGJRGARCHModel` escalan la serie de entrenamiento por `×100` antes de pasarla a la librería `arch`. Esto mejora la estabilidad numérica del optimizador cuando la serie tiene std≈1. Los resultados se re-escalan al retornar:

```python
# Dentro de fit():
am = arch_model(y_train * 100, ...)
self._fitted = am.fit(disp='off')

# Dentro de forecast():
fc.mean.values[-1]     / 100      # ÷ escala    → pronóstico en unidades originales

# Dentro de forecast_intervals():
fc.variance.values[-1] / 10000    # ÷ escala²   → varianza en unidades originales
std_fc = sqrt(max(var_fc, 0))
lo, hi = mean_fc ∓ z * std_fc
```

El caché por horizonte (`self._fc_cache`) evita tres llamadas redundantes a `arch.forecast()` por réplica (punto + intervalo 80% + intervalo 95%) — una única llamada almacenada y reutilizada.

---

### Chronos-2 (todos los experimentos)

**Modelo — `mectesis/models/chronos.py` · `ChronosModel`**
```python
# Librería: chronos (Amazon), torch (bfloat16)
chronos = ChronosModel(device="cpu")
# Carga una sola vez:
#   Chronos2Pipeline.from_pretrained("amazon/chronos-2", dtype=torch.bfloat16)

# fit (zero-shot — sin entrenamiento):
chronos.fit(y_train)              # solo almacena y_train en self._context

# forecast:
y_hat = chronos.forecast(H)
# pipeline.predict_quantiles(
#     inputs=[{"target": y_train}],
#     prediction_length=H,
#     quantile_levels=[0.025, 0.10, 0.50, 0.90, 0.975]
# )
# Punto: cuantil 0.5 (mediana) → quantiles[0][0, :, 2]

# forecast_intervals:
lo, hi = chronos.forecast_intervals(H, level=0.95)
# 95%: cuantiles 0.025 / 0.975 → quantiles[0][0, :, 0] / [0, :, 4]
# 80%: cuantiles 0.10  / 0.90  → quantiles[0][0, :, 1] / [0, :, 3]
# Los cinco cuantiles se obtienen en una única llamada y se cachean
```
