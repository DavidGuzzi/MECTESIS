# DISEÑO EXPERIMENTAL – TESIS MEC

## 0. SETUP GENERAL DEL EXPERIMENTO

| Elemento | Detalle |
|---------|---------|
| Tamaños muestrales | $T \in \{200, 1000\}$ |
| Horizontes | $h \in \{1-12, 13-24\}$ |
| Repeticiones Monte Carlo | $R \in \{500, 1000\}$ |
| Semilla | 03649 |
| TSFM utilizados (todos los experimentos) | Chronos-2 (central), TimesFM-2.5, Moirai-2.0, TimeGPT-1 |
| Filosofía del trabajo | Comparar desempeño predictivo bajo **DGP controlados**, variando propiedades de la serie (persistencia, estacionalidad, tendencia, ruptura, dimensión y dinámica de volatilidad). Evaluar **sesgo**, **varianza**, **MSE/RMSE**, y **intervalos**, identificando bajo qué estructuras los TSFM dominan o son dominados por modelos clásicos. |
| Consideraciones adicionales | Resultados almacenados por experimento/horizonte/tamaño/repetición. Se usa la misma semilla para reproducibilidad. Los modelos clásicos pueden ser “core” o “adicionales”, según relevancia por experimento. |

---
## 0.1. SETUP GENERAL DEL EXPERIMENTO

| Elemento | Diseño propuesto |
|---|---|
| Tamaños muestrales | T ∈ {200, 1000} |
| Horizonte máximo de forecast | H = 24 |
| Bloques de evaluación | Corto plazo: h = 1,…,12; mediano plazo: h = 13,…,24 |
| Repeticiones Monte Carlo | R = 500 como base; R = 1000 para robustez |
| Semilla | 03649 |
| DGPs univariados | 19 |
| Series simuladas con R = 500 | 19 × 2 × 500 = 19.000 |
| Series simuladas con R = 1000 | 19 × 2 × 1000 = 38.000 |
| Modelos clásicos | Solo modelos core por experimento |
| TSFM principal | Chronos-2 |
| TSFM ampliados | TimesFM-2.5, Moirai-2.0, TimeGPT-1 |
| Estrategia computacional | Primero Chronos-2 vs clásicos; luego comparación ampliada con otros TSFM |
| Métricas puntuales | Bias, varianza, MSE, RMSE, intervalos |
| Almacenamiento | Resultados por DGP, T, repetición, modelo, horizonte y bloque |

---

## 1. BLOQUE UNIVARIADO – CUADRO COMPLETO

---

### Experimentos Univariados

| Exp | DGP | Modelos Clásicos (Core) | Modelos Clásicos (Adicionales) |
|-----|-----|-------------------------|---------------------------------|
| **1.1 AR(1) baja persistencia** | $Y_t = 0.3\,Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0) | ETS(A,N,N); Theta; Naive; Drift |
| **1.2 AR(1) alta persistencia** | $Y_t = 0.9\,Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0) | ETS(A,A,N); Theta; Naive |
| **1.3 RW I(1) sin drift** | $Y_t = Y_{t-1} + \varepsilon_t$ | ARIMA(0,1,0) | ETS(A,A,N); Theta; Drift |
| **1.4 RW I(1) con drift** | $Y_t = 0.5 + Y_{t-1} + \varepsilon_t$ | ARIMA(0,1,0) | ETS(A,A,N); Theta; Drift |
| **1.5 AR(1) + tendencia** | $Y_t = 5 + 0.1t + 0.6Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0)+trend | Holt–Winters; ETS; Theta |
| **1.6 SARIMA trimestral (s=4)** | $(1-\phi L)(1-\Phi L^4)Y_t=\varepsilon_t$ | SARIMA(1,0,0)(1,0,0)\_{4} | ETS(A,A,A); Seasonal Naive |
| **1.7 SARIMA mensual (s=12)** | $(1-L)(1-L^{12})Y_t=\varepsilon_t$ | SARIMA estacional | Holt–Winters multiplicativo; ETS; Seasonal Naive |
| **1.8 AR(1) con quiebre** | $Y_t=0.3Y_{t-1}+\varepsilon_t$ para $t\le T/2$; $Y_t=0.8Y_{t-1}+\varepsilon_t$ para $t>T/2$ | ARIMA(1,0,0) con dummy de quiebre | ARIMA(1,0,0) sin quiebre; ETS |
| **1.9 AR(1)–ARCH(1)** | Media: $Y_t = 0.3Y_{t-1} + \varepsilon_t$; $\varepsilon_t = \sigma_t z_t$, $z_t\sim N(0,1)$; Varianza cond.: $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2$ | ARIMA(1,0,0) + ARCH(1) | ARIMA(1,0,0) con varianza homocedástica (ignora volatilidad condicional) |
| **1.10 AR(1)–GARCH(1,1)** | Media: $Y_t = 0.3Y_{t-1} + \varepsilon_t$; Varianza cond.: $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$ con $\alpha,\beta>0$ y $\alpha+\beta<1$ | ARIMA(1,0,0) + GARCH(1,1) | ARIMA(1,0,0) con varianza homocedástica (ignora volatilidad condicional) |
| **1.11 GARCH(1,1) con media cero** | $Y_t = \sigma_t \varepsilon_t$, $\varepsilon_t\sim N(0,1)$; $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$ | GARCH(1,1) | ARIMA(0,0,0) con varianza constante (ignora heterocedasticidad) |
| **1.12 AR(1)–GJR–GARCH (asimetría)** | Media: $Y_t = 0.3Y_{t-1} + \varepsilon_t$; $\varepsilon_t = \sigma_t z_t$, $z_t\sim N(0,1)$; $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \gamma\,\varepsilon_{t-1}^2\mathbf{1}\{\varepsilon_{t-1}<0\} + \beta\,\sigma_{t-1}^2$ | ARIMA(1,0,0) + GJR–GARCH | ARIMA(1,0,0) + GARCH(1,1) (ignora asimetría) |
| **1.13 Nivel local (local level)** | $\ell_t = \ell_{t-1} + \eta_t$, $Y_t = \ell_t + \varepsilon_t$; $\eta_t\sim N(0,\sigma_\eta^2)$, $\varepsilon_t\sim N(0,\sigma_\varepsilon^2)$ | ETS(A,N,N) | ARIMA(0,1,1) (equivalente teórico); Naive |
| **1.14 Tendencia local (local trend / LLT)** | $\ell_t = \ell_{t-1} + b_{t-1} + \eta_t$, $b_t = b_{t-1} + \zeta_t$; $\eta_t\sim N(0,\sigma_\eta^2)$, $\zeta_t\sim N(0,\sigma_\zeta^2)$ | ETS(A,A,N) | Holt lineal; ARIMA(0,2,2) (equivalente teórico) |
| **1.15 Tendencia amortiguada (damped trend)** | $\ell_t = \ell_{t-1} + \phi b_{t-1} + \eta_t$, $b_t = \phi b_{t-1} + \zeta_t$, $\phi = 0.9$ | ETS(A,Ad,N) | ETS(A,A,N) (sin amortiguamiento) |
| **1.16 Estacionalidad determinística (s=12)** | $Y_t = \mu + s_{t \bmod 12} + \varepsilon_t$, $\sum_{j=0}^{11} s_j = 0$, $s_t = s_{t-12}$ | Seasonal Naive (s=12) | SARIMA(0,0,0)(0,0,0)\_12 con dummies estacionales |
| **1.17 Seasonal random walk (s=12)** | $Y_t = Y_{t-12} + \varepsilon_t$, $\varepsilon_t\sim N(0,\sigma^2)$ | Seasonal Naive (s=12) | SARIMA(0,0,0)(0,1,0)\_12 (equivalente teórico) |
| **1.18 Trend + estacionalidad – ETS(A,A,A)** | $\ell_t = \ell_{t-1} + b_{t-1} + \eta_t$, $b_t = b_{t-1} + \zeta_t$, $\gamma_t = \gamma_{t-12} + \omega_t$, $Y_t = \ell_t + \gamma_t + \varepsilon_t$ | ETS(A,A,A) | Holt–Winters aditivo; SARIMA con tendencia |
| **1.19 Tendencia lineal pura (Theta-type)** | $Y_t = 0.1t + \varepsilon_t$, $\varepsilon_t\sim N(0,1)$ | Theta | ETS(A,A,N); ARIMA(0,1,1)+drift; Drift |

---

## 2. BLOQUE MULTIVARIADO – CUADRO COMPLETO

---

### Experimentos VAR / VECM / VARMAX

| Exp | DGP | Modelos Clásicos (Core) |
|-----|-----|-------------------------|
| **2.1 VAR(1) bivariado – baja interdependencia** | $Y_t = A_1 Y_{t-1} + \varepsilon_t$, donde $A_1=\begin{pmatrix}0.5&0.1\\0.1&0.5\end{pmatrix}$ | VAR(1) |
| **2.2 VAR(1) bivariado – alta interdependencia** | $Y_t = A_1 Y_{t-1} + \varepsilon_t$, donde $A_1=\begin{pmatrix}0.5&0.5\\0.5&0.5\end{pmatrix}$ | VAR(1) |
| **2.3 VAR(2) bivariado** | $Y_t = A_1Y_{t-1} + A_2Y_{t-2} + \varepsilon_t$ | VAR(2) |
| **2.4 VAR(1) con 3 variables** | Matriz $3\times3$ conocida | VAR(1) |
| **2.5 VAR(1) con 5 variables** | Matriz $5\times5$ conocida | VAR(1) |
| **2.6 VAR(1) bivariado con volatilidad condicional (ARCH/GARCH diagonal)** | Media: $Y_t = A_1 Y_{t-1} + u_t$, con $A_1$ como en 2.1. Ruido: $u_t = (u_{1t},u_{2t})'$, donde cada componente sigue $u_{it} = \sigma_{it} z_{it}$, $z_{it}\sim N(0,1)$ y $\sigma_{it}^2 = \omega_i + \alpha_i u_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2$ (GARCH(1,1) por ecuación). | VAR(1) para la media, más GARCH(1,1) univariado por ecuación sobre los residuos (modelo tipo VAR + GARCH diagonal); comparación con VAR(1) estándar sin modelar GARCH para ver impacto de ignorar la volatilidad condicional. |
| **2.7 VECM bivariado – cointegración rango 1** | $\Delta Y_t = \alpha\,\beta'\,Y_{t-1} + \Gamma_1\,\Delta Y_{t-1} + \varepsilon_t$, $\varepsilon_t \sim N(0, I_2)$. Vector de cointegración $\beta = (1,-1)'$; velocidades de ajuste $\alpha = (-0.4, 0.2)'$; dinámica de corto plazo $\Gamma_1 = \text{diag}(0.3, 0.3)$. Cada serie individualmente es $I(1)$; la combinación $Y_{1t} - Y_{2t}$ es $I(0)$. | VECM (rango 1, estimado por Johansen); comparación con VAR(1) en niveles (ignora cointegración — inconsistente) y VAR(1) en diferencias (pierde la corrección de largo plazo). Librería: `statsmodels.tsa.vector_ar.vecm.VECM`. **Hipótesis**: Chronos-2 sin restricción de cointegración muestra sesgo creciente en $h \geq 6$ al no reproducir la relación de largo plazo. |

**Notas sobre Opcionales y Modelos Descartados:**

- **VECM rango 2** (Exp 2.8, opcional): sistema de 3 variables con rango $r=2$; permite testear estimación del rango via `coint_johansen`. Bajo prioridad.
- **MS-VAR**: no incorporar — `statsmodels` solo cubre MS univariado. Maduro en **R** (`MSwM::msmFit`, `MSBVAR`) y **Matlab/EViews**. El análogo univariado (Exp 1.20 MS-AR) ya captura el fenómeno de switching.
- **MGARCH (DCC / BEKK)**: no incorporar — sin librería Python activa y robusta. Maduro en **R** (`rmgarch` de Ghalanos, `ccgarch`, `MTS`) y **Matlab** (`Econometrics Toolbox`). El Exp 2.6 (GARCH diagonal) cubre el caso implementable.
- **VARMAX extendido**: no ampliar más allá de Exp 3.4 — `statsmodels.VARMAX` tiene problemas de estabilidad con $k>3$ o $p>1$. Maduro en **R** (`vars::VAR` con `exogen`, `MTS`).

---

## 3. BLOQUE CON COVARIABLES – CUADRO COMPLETO

---

### Experimentos ARIMAX / VARX

| Exp | DGP | Modelos Clásicos (Core) |
|-----|-----|-------------------------|
| **3.1 ARIMAX fuerte** | $Y_t = 0.6Y_{t-1} + 0.8X_t + \varepsilon_t$ | SARIMAX(1,0,0) con exógena |
| **3.2 ARIMAX débil** | $Y_t = 0.6Y_{t-1} + 0.2X_t + \varepsilon_t$ | SARIMAX(1,0,0) |
| **3.3 ARIMAX con dos covariables** | $Y_t = \phi Y_{t-1} + \beta_1X_{1t} + \beta_2X_{2t} + \varepsilon_t$ | SARIMAX(1,0,0) |
| **3.4 VARX bivariado** | $Y_t = A Y_{t-1} + \gamma X_t + \varepsilon_t$ | VARMAX(1) + exógena |
| **3.5 ARIMAX con volatilidad condicional (ARIMAX–GARCH)** | Media: $Y_t = 0.6Y_{t-1} + 0.5X_t + \varepsilon_t$. Ruido: $\varepsilon_t = \sigma_t z_t$, $z_t\sim N(0,1)$, con $\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$ (GARCH(1,1)). Opcional: permitir que $X_t$ también afecte la varianza vía un término $\delta X_t^2$ en la ecuación de $\sigma_t^2$. | ARIMAX (estimado como SARIMAX con exógenas) para la media, y GARCH(1,1) univariado sobre residuos para la varianza condicional; comparación con ARIMAX homocedástico para medir el efecto de modelar explícitamente la volatilidad. |
| **3.6 ADL-ECM – covariable cointegrada** | $X_t = X_{t-1} + u_t$, $u_t \sim N(0,1)$ (random walk). $Y_t = X_t + z_t$, $z_t = 0.7\,z_{t-1} + \eta_t$, $\eta_t \sim N(0,1)$. Equivalente ECM: $\Delta Y_t = -0.3(Y_{t-1} - X_{t-1}) + \Delta X_t + \eta_t$. Ambas series son $I(1)$; la relación $Y_t - X_t \sim I(0)$ es la cointegración de ecuación única (Engle–Granger). | ARDL/ECM ecuación única (core) — `statsmodels.tsa.ardl.ARDL` + transformación ECM; comparación con SARIMAX en diferencias (pierde largo plazo) y SARIMAX en niveles sin restricción (regresión espuria); Chronos-2 universal condicionando en $X_t$. **Nota**: $X_t$ debe ser provista como exógena futura — testea si Chronos-2 descubre la relación de largo plazo sin especificación explícita. |


---

## 4. VALIDACIÓN EMPÍRICA

---

| Elemento | Detalle |
|---------|---------|
| Series argentinas | PIB trimestral, IPC mensual, tipo de cambio, BADLAR, desempleo, exportaciones, importaciones, reservas |
| Modelos clásicos (core) | AutoARIMA, Seasonal Naive |
| Modelos clásicos adicionales | AutoETS, AutoTheta, ensemble estadístico |
| Modelos TSFM | Chronos-2, TimesFM-2.5, Moirai-2.0, TimeGPT-1 |
| Setup temporal | Train: hasta 2019 — Test: 2020–2024 |
| Esquemas | Expanding Window y Rolling Window |
| Horizontes | $h\in\{1,3,6,12\}$ |

---

## 5. MÉTRICAS

### Métricas principales

| Categoría | Métricas |
|-----------|----------|
| Sesgo | $\text{Bias} = \mathbb{E}[\hat{y} - y]$ |
| Varianza | $\text{Var}(\hat{y})$ |
| MSE | $\text{MSE} = \mathbb{E}[(\hat{y}-y)^2]$ |
| RMSE | $\text{RMSE} = \sqrt{\text{MSE}}$ |
| Intervalos | Cobertura y amplitud de intervalos (por ejemplo 90% y 95%) |

### Métricas adicionales (opcionales a revisar)

| Categoría | Métricas |
|-----------|----------|
| Puntuales | MAE, sMAPE |
| Probabilísticas | CRPS, Pinball Loss, Weighted Quantile Loss |
| Calibración extendida | Coverage 50%, 80%, 95% |
| Otros | MASE, interval score |