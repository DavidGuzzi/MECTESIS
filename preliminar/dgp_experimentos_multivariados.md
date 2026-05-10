# Procesos generadores de datos — Experimentos multivariados 2.1–2.7

**Experimentos 2.1–2.7:** $T \in \{50, 200\}$, $H=24$, $R=500$, semilla $=3649$.  
**Valores reportados:** promedios sobre las $k$ variables y los horizontes $h=1\ldots24$ (`avg_all`).  
**DGPs:** VAR estacionarios de distintas dimensiones y estructuras de dependencia, VAR+GARCH diagonal, VECM cointegrado.

---

## Exp 2.1 — VAR(1) bivariado, baja interdependencia

$$Y_t = A_1\,Y_{t-1} + \varepsilon_t, \quad
A_1 = \begin{pmatrix}0.5 & 0.1 \\ 0.1 & 0.5\end{pmatrix}, \quad
\Sigma = \begin{pmatrix}1 & 0.3 \\ 0.3 & 1\end{pmatrix}$$

**Propiedades:** VAR(1) bivariado estacionario. Los eigenvalores de $A_1$ son $0.6$ y $0.4$ — proceso de memoria corta con reversión rápida a la media. La dependencia cruzada es baja: los coeficientes off-diagonal ($0.1$) son pequeños frente a los diagonales ($0.5$). La correlación contemporánea entre innovaciones es $\rho=0.3$. El proceso tiene solución estacionaria única; varianza incondicional $\approx 1.5I$ por la componente AR más la innovación.

**Por qué es relevante:** Caso base multivariado. Evalúa si Chronos-joint (API multivariada) extrae valor del contexto cruzado frente a Chronos-ind (k pronósticos univariados independientes) y frente a VAR. Con baja interdependencia, la ventaja del modelado conjunto debería ser mínima.

**DGP — `mectesis/dgp/var_dgp.py` · `VARDGP`**
```python
from mectesis.dgp import VARDGP
dgp = VARDGP(
    seed=SEED,
    A_list=[[[0.5, 0.1], [0.1, 0.5]]],
    Sigma=[[1.0, 0.3], [0.3, 1.0]],
)
y = dgp.simulate(T=200)   # shape (200, 2)
# Bucle: Y[t] = A1 @ Y[t-1] + eps[t], eps ~ N(0, Sigma)
# burn_in=200 descartados; inicializa en cero
```

**Modelos**

```python
# Librería: statsmodels
from mectesis.models import VARModel
m = VARModel(lags=1)
m.fit(y_train)                        # VAR(y_train, maxlags=1, trend='c').fit()
y_hat = m.forecast(H)                 # shape (H, 2): forecast() analítico
lo, hi = m.forecast_intervals(H, level=0.95)   # forecast_interval() analítico
crps = m.compute_crps(y_test, H)      # crps_gaussian vía forecast_cov()

# Librería: chronos (API multivariada nativa)
from mectesis.models import ChronosMultivariateModel, ChronosPerVarModel
chronos_mv  = ChronosMultivariateModel()   # predict_quantiles sobre batch (k, T)
chronos_ind = ChronosPerVarModel()         # k llamadas univariadas independientes
```

---

## Exp 2.2 — VAR(1) bivariado, alta interdependencia

$$Y_t = A_1\,Y_{t-1} + \varepsilon_t, \quad
A_1 = \begin{pmatrix}0.4 & 0.4 \\ 0.4 & 0.4\end{pmatrix}, \quad
\Sigma = \begin{pmatrix}1 & 0.3 \\ 0.3 & 1\end{pmatrix}$$

**Propiedades:** Los eigenvalores de $A_1$ son $0.8$ y $0$ — un proceso muy cerca del límite de estacionariedad (la componente en la dirección $(1,1)$ tiene persistencia $0.8$; la componente ortogonal tiene coeficiente $0$). El coeficiente off-diagonal $0.4$ es del mismo orden que el diagonal: la información cruzada es esencial para el pronóstico óptimo. En muestras pequeñas, la estimación del VAR tiene mayor varianza porque los parámetros individuales son difíciles de identificar cuando la estructura de dependencia cruzada es fuerte.

**Por qué es relevante:** Contrasta con 2.1 manteniendo el DGP bivariado pero amplificando la interdependencia. Evalúa si la ventaja de modelar conjuntamente (VAR vs univariado) se amplía cuando la información cruzada es más valiosa.

**DGP — `VARDGP`**
```python
dgp = VARDGP(
    seed=SEED,
    A_list=[[[0.4, 0.4], [0.4, 0.4]]],
    Sigma=[[1.0, 0.3], [0.3, 1.0]],
)
```

**Modelos:** VAR(1), Chronos-2 (joint). Sin Chronos-ind (omitido por redundancia con 2.1).

---

## Exp 2.3 — VAR(2) bivariado

$$Y_t = A_1\,Y_{t-1} + A_2\,Y_{t-2} + \varepsilon_t$$

$$A_1 = \begin{pmatrix}0.5 & 0.2 \\ 0.1 & 0.4\end{pmatrix}, \quad
A_2 = \begin{pmatrix}0.1 & 0 \\ 0 & 0.1\end{pmatrix}, \quad
\Sigma = \begin{pmatrix}1 & 0.3 \\ 0.3 & 1\end{pmatrix}$$

**Propiedades:** Proceso VAR(2) estacionario. La dinámica de lag 2 es débil ($A_2 = 0.1\,I$) pero real: contribuye una memoria adicional pequeña sobre la estructura de lag 1. Las ecuaciones para $Y_1$ y $Y_2$ son asimétricas: $Y_1$ tiene mayor dependencia cruzada ($a_{12}=0.2$) que $Y_2$ ($a_{21}=0.1$).

**Por qué es relevante:** Evalúa el costo de misspecificación de orden. VAR(1) ignora el lag 2, pero como $A_2$ es débil, el impacto en RMSE puede ser pequeño. Contrasta con la hipótesis de que Chronos captura implícitamente dinámicas de orden superior via contexto largo.

**DGP — `VARDGP`**
```python
dgp = VARDGP(
    seed=SEED,
    A_list=[[[0.5, 0.2], [0.1, 0.4]],
            [[0.1, 0.0], [0.0, 0.1]]],
    Sigma=[[1.0, 0.3], [0.3, 1.0]],
)
```

**Modelos**
```python
# Modelo correcto:
VAR(2)  = VARModel(lags=2)

# Misspecificado (orden 1):
VAR(1)  = VARModel(lags=1)

# Zero-shot:
chronos_mv  = ChronosMultivariateModel()
```

---

## Exp 2.4 — VAR(1) trivariado ($k=3$)

$$Y_t = A_1\,Y_{t-1} + \varepsilon_t, \quad
A_1 = \begin{pmatrix}0.5 & 0.1 & 0 \\ 0.1 & 0.5 & 0.1 \\ 0 & 0.1 & 0.5\end{pmatrix}, \quad
\Sigma = \begin{pmatrix}1 & 0.2 & 0 \\ 0.2 & 1 & 0.2 \\ 0 & 0.2 & 1\end{pmatrix}$$

**Propiedades:** VAR(1) estacionario con $k=3$ variables. Matriz $A_1$ tridiagonal: la dependencia cruzada es solo entre variables adyacentes. Los eigenvalores de $A_1$ son $\approx \{0.64, 0.5, 0.36\}$. $\Sigma$ es tridiagonal con correlaciones contemporáneas entre variables adyacentes ($\rho=0.2$). Con $k=3$ y $T_{train}=T-H$, el VAR(1) estima $k^2=9$ coeficientes de $A_1$ más $k=3$ interceptos y $k(k+1)/2=6$ parámetros de varianza — total 18 parámetros.

**Por qué es relevante:** Primera incursión en dimensión mayor a 2. Evalúa la maldición de dimensionalidad: a $T=50$ el VAR tiene $T_{train}=26$ observaciones para estimar 18 parámetros. Contrasta si Chronos escala mejor con la dimensión.

**DGP — `VARDGP`**
```python
dgp = VARDGP(
    seed=SEED,
    A_list=[[[0.5, 0.1, 0.0],
             [0.1, 0.5, 0.1],
             [0.0, 0.1, 0.5]]],
    Sigma=[[1.0, 0.2, 0.0],
           [0.2, 1.0, 0.2],
           [0.0, 0.2, 1.0]],
)
```

**Modelos:** VAR(1), Chronos-2 (joint).

---

## Exp 2.5 — VAR(1) pentavariado ($k=5$)

$$Y_t = A_1\,Y_{t-1} + \varepsilon_t$$

Matrices $5\times5$ tridiagonales:

$$[A_1]_{ij} = \begin{cases}0.3 & i=j \\ 0.05 & |i-j|=1 \\ 0 & \text{otherwise}\end{cases}, \qquad
[\Sigma]_{ij} = \begin{cases}1.0 & i=j \\ 0.2 & |i-j|=1 \\ 0 & \text{otherwise}\end{cases}$$

**Propiedades:** VAR(1) con $k=5$ variables y estructura de dependencia local (solo vecinos inmediatos). Los eigenvalores de $A_1$ están en $[0.2, 0.4]$: proceso de memoria corta. Con $T_{train}=26$ ($T=50$) el VAR(1) estima $25+5+15=45$ parámetros — ratio parámetros/observaciones crítico. Los coeficientes off-diagonal ($0.05$) son muy pequeños: la información cruzada aportada por cada variable vecina es mínima.

**Por qué es relevante:** Caso extremo de dimensión con muestra fija. Evalúa el límite del VAR con alta dimensionalidad relativa a $T$ y si Chronos-joint ofrece ventaja por no estimar parámetros cruzados explícitamente.

**DGP — `VARDGP`**
```python
dgp = VARDGP(
    seed=SEED,
    A_list=[[[0.3, 0.05, 0.0, 0.0, 0.0],
             [0.05, 0.3, 0.05, 0.0, 0.0],
             [0.0, 0.05, 0.3, 0.05, 0.0],
             [0.0, 0.0, 0.05, 0.3, 0.05],
             [0.0, 0.0, 0.0, 0.05, 0.3]]],
    Sigma=[[1.0, 0.2, 0.0, 0.0, 0.0],
           [0.2, 1.0, 0.2, 0.0, 0.0],
           [0.0, 0.2, 1.0, 0.2, 0.0],
           [0.0, 0.0, 0.2, 1.0, 0.2],
           [0.0, 0.0, 0.0, 0.2, 1.0]],
)
```

**Modelos:** VAR(1), Chronos-2 (joint).

---

## Exp 2.6 — VAR(1) + GARCH(1,1) diagonal

**Media:** $Y_t = A_1\,Y_{t-1} + u_t$, con $A_1$ como en 2.1.

**Volatilidad:** cada ecuación $i$ sigue un GARCH(1,1) independiente:

$$u_{it} = \sigma_{it}\,z_{it}, \qquad z_{it} \sim \mathcal{N}(0,1) \text{ i.i.d.}$$
$$\sigma_{it}^2 = \omega_i + \alpha_i\,u_{i,t-1}^2 + \beta_i\,\sigma_{i,t-1}^2$$

Con parámetros $(\omega_1, \alpha_1, \beta_1) = (0.1, 0.1, 0.8)$ y $(\omega_2, \alpha_2, \beta_2) = (0.1, 0.15, 0.75)$.

**Propiedades:** La media condicional es idéntica a exp 2.1 (mismo $A_1$). La diferencia está en la estructura de la varianza: heteroscedástica, con clustering de volatilidad. La persistencia GARCH es $\alpha+\beta=0.9$ (ec. 1) y $\alpha+\beta=0.9$ (ec. 2). La varianza incondicional de $u_{1t}$ es $\omega/(1-\alpha-\beta)=1$ y $\omega/(1-\alpha-\beta)=1$ — igual para ambas ecuaciones. Las innovaciones $u_{1t}$ y $u_{2t}$ son independientes dado la historia: el GARCH es **diagonal** (no hay GARCH cruzado).

**Por qué es relevante:** Desacopla la dinámica de media (VAR) de la dinámica de varianza (GARCH). La pregunta clave es si el modelo VAR+GARCH-diag mejora la calibración de intervalos respecto al VAR estándar (que ignora la heteroscedasticidad) y si Chronos captura la volatilidad condicional implícitamente.

**DGP — `mectesis/dgp/var_dgp.py` · `VARGARCHDiagonalDGP`**
```python
from mectesis.dgp import VARGARCHDiagonalDGP
dgp = VARGARCHDiagonalDGP(
    seed=SEED,
    A1=[[0.5, 0.1], [0.1, 0.5]],
    omegas=[0.1, 0.1],
    alphas=[0.1, 0.15],
    betas=[0.8, 0.75],
    burn_in=500,
)
# Bucle:
#   sigma2[t] = omegas + alphas * u_prev^2 + betas * sigma2[t-1]
#   u[t]      = sqrt(sigma2[t]) * z[t],  z ~ N(0, I)
#   Y[t]      = A1 @ Y[t-1] + u[t]
```

**Modelos**
```python
from mectesis.models import VARGARCHDiagonalModel, VARModel

# Modelo correcto: VAR(1) para la media + GARCH(1,1) por ecuación
m_garch = VARGARCHDiagonalModel(n_sim=500, seed=SEED)
m_garch.fit(y_train)
# Internamente:
#   1. VAR(y_train, maxlags=1).fit()  → obtiene residuos
#   2. arch_model(resid[:,j], mean='Zero', vol='GARCH', p=1, q=1).fit()
#      para j=0,1 → parámetros (omega_j, alpha_j, beta_j)
# Intervalos: simulación de 500 paths con GARCH condicional

# Benchmark: VAR(1) que ignora la heteroscedasticidad
m_var = VARModel(lags=1)

# Zero-shot:
chronos_mv = ChronosMultivariateModel()
```

---

## Exp 2.7 — VECM bivariado, cointegración rango 1

$$\Delta Y_t = \alpha\,\beta^{\top}Y_{t-1} + \Gamma_1\,\Delta Y_{t-1} + \varepsilon_t$$

$$\beta = \begin{pmatrix}1 \\ -1\end{pmatrix}, \quad
\alpha = \begin{pmatrix}-0.4 \\ 0.2\end{pmatrix}, \quad
\Gamma_1 = \begin{pmatrix}0.3 & 0 \\ 0 & 0.3\end{pmatrix}, \quad
\Sigma = I_2$$

**Propiedades:** Cada serie individualmente es $I(1)$; la combinación lineal $Y_{1t} - Y_{2t}$ es $I(0)$ (relación de cointegración). Los parámetros de ajuste $\alpha_1=-0.4$ y $\alpha_2=0.2$ cuantifican la velocidad de corrección del desequilibrio: si $Y_1 - Y_2 > 0$, $Y_1$ disminuye y $Y_2$ aumenta, restaurando el equilibrio. $\Gamma_1 = 0.3\,I$ agrega dinámica de corto plazo. El VECM en niveles equivale a:

$$Y_t = (I + \alpha\beta^{\top})Y_{t-1} + \Gamma_1(Y_{t-1} - Y_{t-2}) + \varepsilon_t$$

con $\Pi = I + \alpha\beta^{\top}$ de rango reducido ($\text{rk}(\Pi)=1$). Los errores de pronóstico crecen con $h$ porque el proceso es $I(1)$ en niveles — el RMSE esperado crece como $O(\sqrt{h})$.

**Por qué es relevante:** El experimento más desafiante del bloque. Evalúa tres aspectos: (1) si el VECM correctamente especificado aprovecha la relación de cointegración; (2) si el VAR en niveles (que ignora la cointegración) produce pronósticos sesgados o inestables con muestras pequeñas; (3) si Chronos puede capturar la cointegración implícitamente desde el contexto de la serie.

**DGP — `mectesis/dgp/vecm_dgp.py` · `VECMBivariateDGP`**
```python
from mectesis.dgp import VECMBivariateDGP
dgp = VECMBivariateDGP(
    seed=SEED,
    # Parámetros default del DGP:
    alpha=[-0.4, 0.2],
    beta=[1.0, -1.0],
    Gamma1=[[0.3, 0.0], [0.0, 0.3]],
    Sigma=[[1.0, 0.0], [0.0, 1.0]],
    burn_in=200,
)
# Bucle:
#   ecm_term = alpha * (beta @ Y[t-1])        # corrección de desequilibrio
#   dyn_term = Gamma1 @ (Y[t-1] - Y[t-2])    # dinámica de corto plazo
#   Y[t]     = Y[t-1] + ecm_term + dyn_term + eps[t]
```

**Modelos**
```python
from mectesis.models import VECMModel, VARModel

# Modelo correcto: VECM con rango de cointegración 1
m_vecm = VECMModel(coint_rank=1, k_ar_diff=1, n_sim=500, seed=SEED)
m_vecm.fit(y_train)
# Internamente:
#   VECM(y_train, k_ar_diff=1, coint_rank=1, deterministic='ci').fit()
#   → estimación Johansen; extrae alpha, beta, Gamma1
# Pronóstico puntual: predict(steps=H)
# Intervalos/CRPS: bootstrap residual con ecuación VECM en niveles:
#   Pi = I + alpha @ beta.T   → (k, k)
#   Y_next = Pi @ Y_prev + alpha @ det_coef + eps_resampleado

# Benchmark: VAR(1) en niveles sin restricción de cointegración
m_var = VARModel(lags=1)
# Aplica VAR a datos I(1) → puede producir spurious regression con T pequeño

# Zero-shot joint y univariado:
chronos_mv  = ChronosMultivariateModel()
chronos_ind = ChronosPerVarModel()
```

---

## Infraestructura común

```
mectesis/dgp/var_dgp.py         VARDGP, VARGARCHDiagonalDGP
mectesis/dgp/vecm_dgp.py        VECMBivariateDGP
mectesis/models/var_model.py    VARModel, VECMModel, VARGARCHDiagonalModel
mectesis/models/chronos_multivariate.py  ChronosMultivariateModel, ChronosPerVarModel
mectesis/simulation/multivariate_engine.py  MultivariateMonteCarloEngine
```

El ciclo Monte Carlo en `multivariate_engine.py`:
```python
for s in range(R):
    y = dgp.simulate(T=T)              # shape (T, k)
    y_train, y_test = y[:-H], y[-H:]   # split fijo
    for model in models:
        model.fit(y_train)              # (T_train, k)
        y_hat = model.forecast(H)       # (H, k)
        lo, hi = model.forecast_intervals(H, level=0.80)  # (H, k) cada uno
        lo95, hi95 = model.forecast_intervals(H, level=0.95)
        crps = model.compute_crps(y_test, H)              # (H, k)
        # acumula métricas por variable j=0,...,k-1
```

### CRPS por modelo

| Modelo | Método CRPS |
|---|---|
| `VARModel` | Gaussiano analítico vía `forecast_cov(steps)` → $\text{CRPS}\big(\mathcal{N}(\hat\mu_h,\,[\Sigma_h]_{jj})\big)$ |
| `VECMModel` | Bootstrap residual: $N_{sim}=500$ paths del VECM en niveles; `crps_ensemble` |
| `VARGARCHDiagonalModel` | Ensemble de $N_{sim}=500$ paths con GARCH condicional; `crps_ensemble` |
| `ChronosMultivariateModel` | Cuantiles de la distribución predictiva Chronos; `crps_ensemble` |
