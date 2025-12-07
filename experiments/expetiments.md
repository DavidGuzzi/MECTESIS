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

## 1. BLOQUE UNIVARIADO – CUADRO COMPLETO

---

### Experimentos Univariados

| Exp | DGP | Modelos Clásicos (Core) | Modelos Clásicos (Adicionales) |
|-----|-----|-------------------------|---------------------------------|
| **1.1 AR(1) baja persistencia** | $Y_t = 0.3\,Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0); Naive; Drift | ETS(A,N,N); Theta |
| **1.2 AR(1) alta persistencia** | $Y_t = 0.9\,Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0); Naive | ETS(A,A,N); Theta |
| **1.3 RW I(1) sin drift** | $Y_t = Y_{t-1} + \varepsilon_t$ | ARIMA(0,1,0); Drift | ETS(A,A,N); Theta |
| **1.4 RW I(1) con drift** | $Y_t = 0.5 + Y_{t-1} + \varepsilon_t$ | ARIMA(0,1,0); Drift | ETS(A,A,N); Theta |
| **1.5 AR(1) + tendencia** | $Y_t = 5 + 0.1t + 0.6Y_{t-1} + \varepsilon_t$ | ARIMA(1,0,0)+trend | Holt–Winters; ETS; Theta |
| **1.6 SARIMA trimestral (s=4)** | $(1-\phi L)(1-\Phi L^4)Y_t=\varepsilon_t$ | SARIMA(1,0,0)(1,0,0)\_{4}; Seasonal Naive | ETS(A,A,A) |
| **1.7 SARIMA mensual (s=12)** | $(1-L)(1-L^{12})Y_t=\varepsilon_t$ | SARIMA estacional; Seasonal Naive | Holt–Winters multiplicativo; ETS |
| **1.8 AR(1) con quiebre** | $Y_t=0.3Y_{t-1}+\varepsilon_t$ para $t\le T/2$; $Y_t=0.8Y_{t-1}+\varepsilon_t$ para $t>T/2$ | ARIMA(1,0,0) con dummy de quiebre | ARIMA(1,0,0) sin quiebre; ETS |
| **1.9 AR(1)–GARCH(1,1)** | Media: $Y_t = 0.3Y_{t-1} + \varepsilon_t$; Varianza cond.: $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$ con $\alpha,\beta>0$ y $\alpha+\beta<1$ | ARIMA(1,0,0) + GARCH(1,1) (modelo de media y varianza conjunta) | ARIMA(1,0,0) con varianza homocedástica (modelo mal especificado para comparar impacto de ignorar GARCH) |
| **1.10 GARCH(1,1) con media cero** | $Y_t = \sigma_t \varepsilon_t$, $\varepsilon_t\sim N(0,1)$, $\sigma_t^2 = \omega + \alpha\,Y_{t-1}^2 + \beta\,\sigma_{t-1}^2$ | GARCH(1,1) (modelo puro de volatilidad); Naive sobre niveles para contraste | ARIMA(0,0,0) con varianza constante (para estudiar fallas de modelos clásicos al ignorar volatilidad condicional) |
| **1.11 AR(1)–GJR–GARCH (asimetría)** | Media: $Y_t = 0.3Y_{t-1} + \varepsilon_t$; $\varepsilon_t = \sigma_t z_t$, $z_t\sim N(0,1)$; $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \gamma\,\varepsilon_{t-1}^2\mathbf{1}\{\varepsilon_{t-1}<0\} + \beta\,\sigma_{t-1}^2$ | AR(1) + GJR–GARCH (para capturar efectos de “leverage”) | AR(1)+GARCH(1,1) estándar (sin término asimétrico) para ver pérdida de capacidad al ignorar asimetría |

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

Opcionales:

- Cointegración → VECM  
- Exógenas → VARMAX  
- Extensión de volatilidad multivariada → MGARCH (p.ej. BEKK o DCC).

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

### Métricas adicionales (opcionales)

| Categoría | Métricas |
|-----------|----------|
| Puntuales | MAE, sMAPE |
| Probabilísticas | CRPS, Pinball Loss, Weighted Quantile Loss |
| Calibración extendida | Coverage 50%, 80%, 95% |
| Otros | MASE, interval score |