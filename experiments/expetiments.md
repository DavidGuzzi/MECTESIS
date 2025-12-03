# DISEÑO EXPERIMENTAL – TESIS MEC  
Formato cuadro, listo para VS Code.  
Basado en `tesis_mec_experimentos.docx` (referencia interna).

---

# 0. SETUP GENERAL

| Elemento | Detalle |
|---------|---------|
| Tamaños muestrales | T_chico ∈ {100, 200} — T_grande ∈ {800, 1000} |
| Horizontes | h ∈ {1, 6, 12, 24} |
| Repeticiones Monte Carlo | R ∈ {200, 500, 1000} |
| Semilla | 03649 |
| TSFM (todos los experimentos) | Chronos-2 (central), TimesFM-2.5, Moirai-2.0, TimeGPT-1 |
| Filosofía | Comparación puntual, probabilística y calibración |

---

# 1. BLOQUE UNIVARIADO – CUADRO COMPLETO

---

## Experimentos Univariados

| Exp | DGP | Modelos Clásicos | Importaciones Python |
|-----|-----|------------------|----------------------|
| **1.1 AR(1) baja persistencia** | Y_t = 0.3·Y_{t-1} + ε_t | ARIMA(1,0,0); ETS(A,N,N); Theta; Naive; Drift | `from statsmodels.tsa.arima.model import ARIMA`<br>`from statsmodels.tsa.exponential_smoothing.ets import ETSModel`<br>`from sktime.forecasting.theta import ThetaForecaster`<br>`from sktime.forecasting.naive import NaiveForecaster` |
| **1.2 AR(1) alta persistencia** | Y_t = 0.9·Y_{t-1} + ε_t | ARIMA(1,0,0); ETS(A,A,N); Theta; Naive; Drift | (igual 1.1) |
| **1.3 RW I(1)** | Y_t = Y_{t-1} + ε_t | ARIMA(0,1,0); Drift; ETS(A,A,N); Theta | `ARIMA` etc. |
| **1.4 RW con drift** | Y_t = 0.5 + Y_{t-1} + ε_t | ARIMA(0,1,0) con constante; Drift; Theta; ETS(A,A,N) | igual |
| **1.5 AR(1) + tendencia** | Y_t = 5 + 0.1t + 0.6Y_{t-1} + ε_t | ARIMA(1,0,0)+trend; ETS(A,A,N); Holt-Winters; Theta | `from statsmodels.tsa.holtwinters import ExponentialSmoothing` |
| **1.6 SARIMA trimestral (s=4)** | (1 − φL)(1 − ΦL⁴)Y_t = ε_t | SARIMA(1,0,0)(1,0,0)[4]; ETS(A,A,A); Seasonal Naive | `from statsmodels.tsa.statespace.sarimax import SARIMAX`<br>`from sktime.forecasting.naive import NaiveForecaster` |
| **1.7 SARIMA mensual (s=12)** | (1 − L)(1 − L¹²)Y_t = ε_t | SARIMA; Holt-Winters (mult.); ETS; Seasonal Naive | igual 1.6 |
| **1.8 AR(1) con quiebre** | Y_t=0.3Y_{t-1}+ε_t (t≤T/2) — Y_t=0.8Y_{t-1}+ε_t (t>T/2) | ARIMA(1,0,0) sin quiebre; ARIMA con dummy; ETS | `ARIMA`, `ETSModel` |

---

# 2. BLOQUE MULTIVARIADO – CUADRO COMPLETO

---

## Experimentos VAR / VECM / VARMAX

| Exp | DGP | Modelos Clásicos | Importaciones Python |
|-----|-----|------------------|----------------------|
| **2.1 VAR(1) bivariado – baja interdependencia** | Y_t = A·Y_{t-1} + ε_t, A = [[0.5,0.1],[0.1,0.5]] | VAR(1) | `from statsmodels.tsa.api import VAR` |
| **2.2 VAR(1) bivariado – alta interdependencia** | A cruzados=0.5 | VAR(1) | VAR |
| **2.3 VAR(2) bivariado** | Y_t = A1Y_{t-1} + A2Y_{t-2} + ε_t | VAR(2) | VAR |
| **2.4 VAR(1) con 3 variables** | Matriz A 3×3 conocida | VAR(1) | VAR |
| **2.5 VAR(1) con 5 variables** | Matriz A 5×5 conocida | VAR(1) | VAR |

(Nota: si quisieras agregar cointegración → usar `VECM`)

Importaciones adicionales opcionales:
```
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.statespace.varmax import VARMAX
```

---

# 3. BLOQUE CON COVARIABLES – CUADRO COMPLETO

---

## Experimentos ARIMAX / VARX

| Exp | DGP | Modelos Clásicos | Importaciones Python |
|-----|-----|------------------|----------------------|
| **3.1 ARIMAX relación fuerte** | Y_t = 0.6Y_{t-1} + 0.8X_t + ε_t | SARIMAX(1,0,0) con exógena | `from statsmodels.tsa.statespace.sarimax import SARIMAX` |
| **3.2 ARIMAX relación débil** | Y_t = 0.6Y_{t-1} + 0.2X_t + ε_t | SARIMAX(1,0,0) | SARIMAX |
| **3.3 ARIMAX con 2 covariables** | Y_t = φY_{t-1} + β₁X₁_t + β₂X₂_t + ε_t | SARIMAX(1,0,0) | SARIMAX |
| **3.4 VARX bivariado con 1 covariable** | Y_t = A·Y_{t−1} + γX_t + ε_t | VARMAX(1) con exógena | `from statsmodels.tsa.statespace.varmax import VARMAX` |

---

# 4. VALIDACIÓN EMPÍRICA – CUADRO COMPLETO

---

| Elemento | Detalle |
|---------|---------|
| Series argentinas | PIB trimestral, IPC mensual, tipo de cambio, BADLAR, desempleo, export/import, reservas |
| Modelos clásicos | AutoARIMA, AutoETS, AutoTheta, Seasonal Naive, ensemble estadístico |
| TSFM | Chronos-2, TimesFM-2.5, Moirai-2.0, TimeGPT-1 |
| Setup | Train: hasta 2019 — Test: 2020–2024 |
| Estrategias | Expanding window y Rolling window |
| Horizontes | h ∈ {1, 3, 6, 12} |

Importaciones típicas:
```python
from pmdarima import auto_arima
from sktime.forecasting.theta import ThetaForecaster
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
```

---

# 5. MÉTRICAS – CUADRO COMPLETO

| Categoría | Métricas |
|-----------|----------|
| Puntuales | MAE, RMSE, MASE, sMAPE |
| Probabilísticas | CRPS, Pinball Loss, Weighted Quantile Loss |
| Calibración | Coverage (50%,90%,95%), Interval Width |
| Descomposición | Sesgo, Varianza, Bias–Variance Trade-off |

---

# 6. CONSIDERACIONES ADICIONALES

| Tema | Detalle |
|------|---------|
| Manejo de replicaciones | Cada experimento se corre para T chico y grande, todos los h, para R replicaciones |
| Comparación TSFM vs clásicos | Punto, intervalo, probabilidad y calibración |
| Robustez | Cambiar seed, cambiar variance de ε_t, introducir heteroscedasticidad opcional |
| Implementación | Guardar resultados por experimento en parquet/csv con estructura jerárquica |

---

# FIN DEL DOCUMENTO .md
