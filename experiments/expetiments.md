# DISEÑO EXPERIMENTAL – TESIS MEC

## 0. SETUP GENERAL DEL EXPERIMENTO

| Elemento | Detalle |
|---------|---------|
| Tamaños muestrales | $T \in \{200, 1000\}$ |
| Horizontes | $h \in \{12, 24\}$ |
| Repeticiones Monte Carlo | $R \in \{500, 1000\}$ |
| Semilla | 03649 |
| TSFM utilizados (todos los experimentos) | Chronos-2 (central), TimesFM-2.5, Moirai-2.0, TimeGPT-1 |
| Filosofía del trabajo | Comparar desempeño predictivo bajo **DGP controlados**, variando propiedades de la serie (persistencia, estacionalidad, tendencia, ruptura, dimensión). Evaluar **sesgo**, **varianza**, **MSE/RMSE**, y **intervalos**, identificando bajo qué estructuras los TSFM dominan o son dominados por modelos clásicos. |
| Consideraciones adicionales | Resultados almacenados por experimento/horizonte/tamaño/repetición. Se usa la misma semilla para reproducibilidad. Los modelos clásicos pueden ser “core” o “adicionales”, según relevancia por experimento. |

---

## 1. BLOQUE UNIVARIADO – CUADRO COMPLETO

---

### Experimentos Univariados

| Exp | DGP | Modelos Clásicos (Core) | Modelos Clásicos (Adicionales) |
|-----|-----|-------------------------|------------------------------|
| **1.1 AR(1) baja persistencia** | $Y_t = 0.3\,Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0); Naive; Drift | ETS(A,N,N); Theta |
| **1.2 AR(1) alta persistencia** | $Y_t = 0.9\,Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0); Naive | ETS(A,A,N); Theta |
| **1.3 RW I(1) sin drift** | $Y_t = Y_{t-1} + \varepsilon_t$ | ARIMA(0,1,0); Drift | ETS(A,A,N); Theta |
| **1.4 RW I(1) con drift** | $Y_t = 0.5 + Y_{t-1} + \varepsilon_t$ | ARIMA(0,1,0); Drift | ETS(A,A,N); Theta |
| **1.5 AR(1) + tendencia** | $Y_t = 5 + 0.1t + 0.6Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0)+trend | Holt-Winters; ETS; Theta |
| **1.6 SARIMA trimestral (s=4)** | $(1-\phi L)(1-\Phi L^4)Y_t=\varepsilon_t$ | SARIMA(1,0,0)(1,0,0)[4]; Seasonal Naive | ETS(A,A,A) |
| **1.7 SARIMA mensual (s=12)** | $(1-L)(1-L^{12})Y_t=\varepsilon_t$ | SARIMA estacional; Seasonal Naive | Holt-Winters multiplicativo; ETS |
| **1.8 AR(1) con quiebre** | $Y_t=0.3Y_{t-1}+\varepsilon_t$ para $t\le T/2$; $Y_t=0.8Y_{t-1}+\varepsilon_t$ para $t>T/2$ | ARIMA(1,0,0) sin quiebre | ARIMA con dummy; ETS |

---

## 2. BLOQUE MULTIVARIADO – CUADRO COMPLETO

---

### Experimentos VAR / VECM / VARMAX

| Exp | DGP | Modelos Clásicos (Core) |
|-----|-----|-------------------------|
| **2.1 VAR(1) bivariado – baja interdependencia** | $Y_t = A Y_{t-1} + \varepsilon_t$, donde $A_1=\begin{pmatrix}0.5&0.1\\0.1&0.5\end{pmatrix}$ | VAR(1) |
| **2.2 VAR(1) bivariado – alta interdependencia** | $Y_t = A_1 Y_{t-1} + \varepsilon_t$, donde $A_1=\begin{pmatrix}0.5&0.5\\0.5&0.5\end{pmatrix}$ | VAR(1) |
| **2.3 VAR(2) bivariado** | $Y_t = A_1Y_{t-1} + A_2Y_{t-2} + \varepsilon_t$ | VAR(2) |
| **2.4 VAR(1) con 3 variables** | Matriz $3\times3$ conocida | VAR(1) |
| **2.5 VAR(1) con 5 variables** | Matriz $5\times5$ conocida | VAR(1) |

Opcionales (solo si se decide ampliar análisis):

- Cointegración → `VECM`
- Exógenas → `VARMAX`

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
| RMSE | $\sqrt{\text{MSE}}$ |
| Intervalos | Cobertura y amplitud |

### Métricas adicionales (opcionales)

| Categoría | Métricas |
|-----------|----------|
| Puntuales | MAE, sMAPE |
| Probabilísticas | CRPS, Pinball Loss, Weighted Quantile Loss |
| Calibración extendida | Coverage 50%, 80%, 95% |
| Otros | MASE, interval score |