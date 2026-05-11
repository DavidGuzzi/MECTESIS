# Procesos generadores de datos — Experimentos GP (Gaussian Process / KernelSynth)

**Experimentos GP.1–GP.3:** $T \in \{50, 200\}$, $H=24$, $R=500$, semilla $=3649$.  
**Motivación:** Chronos-2 (Ansari et al. 2024) fue entrenado con datos sintéticos generados mediante *KernelSynth* — un procedimiento que muestrea series temporales desde Gaussian Processes con kernels compuestos seleccionados aleatoriamente. Estos experimentos buscan evaluar si Chronos obtiene ventaja competitiva en series generadas por el mismo mecanismo probabilístico que su training data.

---

## Fundamentos: Gaussian Process y Cholesky sampling

Un Gaussian Process (GP) es una distribución sobre funciones. Para el caso discreto con $T$ puntos de tiempo indexados $\{0, 1, \ldots, T-1\}$, el proceso se caracteriza completamente por su función de media $m(t) = 0$ y su función de covarianza (kernel) $k(t, t')$:

$$y = (y_0, y_1, \ldots, y_{T-1}) \sim \mathcal{GP}(0, K)$$

donde $K \in \mathbb{R}^{T \times T}$ es la **matriz de covarianza** o *Gram matrix*, con $K_{ij} = k(t_i, t_j)$.

**Muestreo exacto por descomposición de Cholesky:** dado que $K$ es semidefinida positiva (y definida positiva con ruido de observación), se puede factorizar como $K = L L^\top$ con $L$ triangular inferior. Entonces:

$$y = L z, \quad z \sim \mathcal{N}(0, I_T) \implies y \sim \mathcal{N}(0, K) = \mathcal{GP}(0, K)$$

Cada replicación Monte Carlo genera un nuevo $z$ con semilla diferente, produciendo una trayectoria distinta del mismo proceso estocástico.

**Propiedad clave:** La covarianza GP no tiene representación ARMA de orden finito bajo kernels como el RBF. El proceso tiene memoria de largo alcance — la autocorrelación $k(t, t+h)$ no decae exponencialmente sino como una Gaussiana en $h$ (para RBF) o como una función periódica amortiguada (para kernels compuestos). Cualquier modelo ARIMA de orden finito sub-especifica inevitablemente esta estructura.

---

## Kernels implementados

### Kernel RBF (Squared-Exponential)

$$k_{RBF}(t, t') = \sigma_{rbf}^2 \cdot \exp\!\left(-\frac{(t - t')^2}{2\,\ell_{rbf}^2}\right)$$

- **Parámetros:** $\sigma_{rbf}^2$ (varianza / amplitud), $\ell_{rbf}$ (longitud de escala)
- **Propiedades:** La función resultante es infinitamente diferenciable — genera trayectorias extremadamente suaves. La longitud de escala $\ell_{rbf}$ controla qué tan rápido varía la función: con $\ell_{rbf}=30$ (unidades de tiempo), valores separados por 30 períodos tienen covarianza $\exp(-0.5) \approx 0.61$.
- **No-estacionariedad aparente:** En muestras finitas, un GP-RBF genera tendencias no lineales que son localmente similares a una caminata aleatoria, pero globalmente acotadas.

### Kernel Periódico

$$k_{Per}(t, t') = \sigma_{per}^2 \cdot \exp\!\left(-\frac{2\,\sin^2\!\left(\pi\,|t - t'| / p\right)}{\ell_{per}^2}\right)$$

- **Parámetros:** $\sigma_{per}^2$ (amplitud), $p$ (período), $\ell_{per}$ (suavidad dentro del período)
- **Propiedades:** Función estrictamente periódica con período $p$. A diferencia de la estacionalidad determinística (senos y cosenos fijos), el GP periódico genera patrones estacionales que varían levemente de ciclo a ciclo según la función de covarianza.
- **$\ell_{per}$ controla la forma:** con $\ell_{per}=1$, la función es suave dentro del período (armónicos de orden bajo); con $\ell_{per} \to 0$, la función tiende a ser más rígida (solo el armónico fundamental).

### Kernel Compuesto (KernelSynth)

$$K = k_{RBF} + k_{Per} + (\sigma_{noise}^2 + \varepsilon)\,I_T$$

donde $\varepsilon = 10^{-6}$ es *jitter* numérico para garantizar que $K$ sea definida positiva. La suma de kernels produce un proceso que combina tendencia suave (RBF) y estacionalidad (Periódico), con ruido de observación $\sigma_{noise}^2$ sobre la diagonal.

La **varianza marginal teórica** (varianza de cada $y_t$ individualmente) es:

$$\mathrm{Var}(y_t) = \begin{cases} \sigma_{rbf}^2 + \sigma_{noise}^2 & \text{kernel RBF} \\ \sigma_{per}^2 + \sigma_{noise}^2 & \text{kernel Periódico} \\ \sigma_{rbf}^2 + \sigma_{per}^2 + \sigma_{noise}^2 & \text{RBF + Periódico} \end{cases}$$

La media teórica es cero en los tres casos.

---

## Implementación — `mectesis/dgp/gp_dgp.py` · `GPKernelSynthDGP`

```python
# Librería: numpy (descomposición de Cholesky)
from mectesis.dgp import GPKernelSynthDGP

dgp = GPKernelSynthDGP(seed=SEED)

# --- GP.1: RBF puro ---
y = dgp.simulate(T=200, kernel="rbf", lengthscale_rbf=30.0,
                 sigma_rbf=1.0, noise_std=0.3)

# --- GP.2: Periódico puro ---
y = dgp.simulate(T=200, kernel="periodic", period=12.0,
                 lengthscale_per=1.0, sigma_per=1.0, noise_std=0.3)

# --- GP.3: RBF + Periódico (KernelSynth) ---
y = dgp.simulate(T=200, kernel="rbf+periodic",
                 lengthscale_rbf=30.0, sigma_rbf=1.0,
                 period=12.0, lengthscale_per=1.0, sigma_per=0.8,
                 noise_std=0.3)
```

**Código interno de `simulate()`:**

```python
def simulate(self, T, kernel="rbf+periodic",
             lengthscale_rbf=30.0, sigma_rbf=1.0,
             period=12.0, lengthscale_per=1.0, sigma_per=0.8,
             noise_std=0.3):
    t = np.arange(T, dtype=float)
    diff = t[:, None] - t[None, :]          # (T, T) signed differences

    K = np.zeros((T, T))
    if "rbf" in kernel:
        K += sigma_rbf**2 * np.exp(-(diff**2) / (2.0 * lengthscale_rbf**2))
    if "periodic" in kernel:
        K += sigma_per**2 * np.exp(
            -2.0 * np.sin(np.pi * np.abs(diff) / period)**2
            / lengthscale_per**2)
    K += (noise_std**2 + 1e-6) * np.eye(T)  # ruido + jitter

    L = np.linalg.cholesky(K)
    return L @ self.rng.standard_normal(T)   # y ~ GP(0, K)
```

**Reproducibilidad:** `self.rng = np.random.default_rng(seed)` está definido en `BaseDGP`. Cada llamada a `simulate()` avanza el estado del RNG, por lo que la misma instancia produce trayectorias distintas y reproducibles en el bucle Monte Carlo.

---

## Tabla de parámetros por experimento

| Parámetro | GP.1 (RBF) | GP.2 (Periódico) | GP.3 (RBF+Per) |
|-----------|-----------|-----------------|----------------|
| `kernel` | `"rbf"` | `"periodic"` | `"rbf+periodic"` |
| `lengthscale_rbf` $\ell_{rbf}$ | 30.0 | — | 30.0 |
| `sigma_rbf` $\sigma_{rbf}$ | 1.0 | — | 1.0 |
| `period` $p$ | — | 12.0 | 12.0 |
| `lengthscale_per` $\ell_{per}$ | — | 1.0 | 1.0 |
| `sigma_per` $\sigma_{per}$ | — | 1.0 | 0.8 |
| `noise_std` $\sigma_{noise}$ | 0.3 | 0.3 | 0.3 |
| $\mathrm{Var}(y_t)$ teórica | 1.09 | 1.09 | 1.73 |
| $\mathrm{E}[y_t]$ | 0 | 0 | 0 |

---

## Modelos por experimento

### Exp GP.1 — RBF puro

**Benchmark:** `ARIMAModel`, `ETSModel`

```python
# Modelo 1 — ARIMA(1,1,1)
# Librería: statsmodels
m = ARIMAModel(order=(1, 1, 1))
m.fit(y_train)
# Justificación: d=1 aproxima la tendencia suave del GP-RBF como una
# caminata aleatoria integrada. Es el modelo I(1) más simple que puede
# adaptarse localmente a la tendencia cambiante.
y_hat = m.forecast(H)
lo, hi = m.forecast_intervals(H, level=0.95)

# Modelo 2 — ETS(A,A,N)
# Librería: statsmodels
m = ETSModel(error="add", trend="add", seasonal=None)
m.fit(y_train)
# Justificación: tendencia local aditiva, sin estacionalidad.
# Captura la deriva local de la función GP-RBF mediante suavizamiento
# exponencial de doble componente.
```

**Por qué el clásico puede competir:** La diferenciación $d=1$ en ARIMA captura el comportamiento de "casi caminata aleatoria" del GP-RBF en muestras cortas. A $T$ grande, Chronos tiene más contexto para identificar la curvatura de la tendencia, lo que puede darle ventaja.

### Exp GP.2 — Periódico puro

**Benchmark:** `SARIMAModel`, `ETSModel`

```python
# Modelo 1 — SARIMA(1,0,1)(1,0,1)_12
# Librería: statsmodels
m = SARIMAModel(order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
m.fit(y_train)
# Justificación: captura la periodicidad con período s=12 mediante
# componentes AR y MA estacionales. Sin diferenciación (d=0) porque
# el GP periódico es estacionario.
# ADVERTENCIA: inestable a T=50 (< 4.2 ciclos completos).

# Modelo 2 — ETS(A,N,A)
# Librería: statsmodels
m = ETSModel(error="add", trend=None, seasonal="add", seasonal_periods=12)
m.fit(y_train)
# Justificación: estacionalidad aditiva pura, sin tendencia.
# Apropiado para el GP periódico que tiene media cero y estacionalidad
# sin tendencia de largo plazo.
```

**Por qué ETS domina en T=50:** Con menos de 5 ciclos completos, ETS puede estimar los 12 coeficientes estacionales con razonable precisión, mientras que SARIMA necesita estimar parámetros adicionales ($\phi, \theta, \Phi, \Theta$) que resultan inestables. Chronos no tiene ventaja porque el patrón periódico regular es capturado eficientemente por ETS.

### Exp GP.3 — RBF + Periódico (KernelSynth completo)

**Benchmark:** `SARIMAModel`, `ETSModel`, `ThetaModel`

```python
# Modelo 1 — SARIMA(1,1,1)(1,0,1)_12
# Librería: statsmodels
m = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
m.fit(y_train)
# Justificación: intenta capturar tendencia (d=1) + estacionalidad (s=12).
# ADVERTENCIA: la diferenciación sobre un proceso GP estacionario
# introduce una raíz unitaria artificial, catastrófico a T=50.

# Modelo 2 — ETS(A,A,A)
# Librería: statsmodels
m = ETSModel(error="add", trend="add", seasonal="add", seasonal_periods=12)
m.fit(y_train)
# Justificación: captura tendencia local + estacionalidad. El modelo más
# completo del bloque ETS para este DGP.
# LIMITACIÓN: 3 componentes estocásticos requieren suficiente historia;
# subcovertura severa a T=50.

# Modelo 3 — Theta
# Librería: statsmodels
m = ThetaModel(seasonal_periods=12)
m.fit(y_train)
# Justificación: incluido como benchmark adicional. Combina tendencia
# (SES) con ajuste de drift. Peor que ETS en RMSE en ambos T.
```

**Por qué Chronos gana en GP.3:** El kernel compuesto RBF+Periódico produce series con estructura idéntica a la del entrenamiento KernelSynth de Chronos. El modelo de fundación puede reconocer simultáneamente la tendencia suave no lineal y la variación estacional sin estimar explícitamente ningún parámetro. Con T=200, tiene suficiente contexto para identificar ambas componentes con precisión superior a cualquier modelo paramétrico de orden finito.

---

## Justificación teórica: ¿por qué el GP es el DGP "natural" de Chronos?

### Memoria infinita del kernel RBF

La autocovarianza del proceso GP-RBF en lag $h$ es $k_{RBF}(h) = \sigma_{rbf}^2 \exp(-h^2 / 2\ell^2)$. Para aproximar esta función con un proceso ARMA$(p,q)$, necesitamos que la representación espectral del ARMA se ajuste a la densidad espectral del GP:

$$f_{GP}(\omega) \propto \ell \cdot \sigma_{rbf}^2 \cdot \exp\!\left(-\frac{\omega^2 \ell^2}{2}\right)$$

La densidad espectral Gaussiana del GP-RBF es **no racional** — no puede representarse como el cociente de dos polinomios en $e^{i\omega}$ que caracteriza a los procesos ARMA. Por tanto, ningún ARMA$(p,q)$ finito puede capturar exactamente la estructura de covarianza del GP-RBF: el modelo clásico siempre incurre en un error de misspecificación que no desaparece con más datos.

### Chronos y KernelSynth

El paper de Chronos (Ansari et al. 2024) describe que el modelo fue entrenado con un corpus que incluye datos sintéticos generados por *KernelSynth*: combinaciones aleatorias de kernels GP seleccionados de un conjunto que incluye RBF, periódico, lineal, periódico local (RBF × Periódico) y ruido. El resultado es que Chronos aprendió a reconocer y extrapolar patrones de covarianza GP implícitamente, sin formular el problema en términos de kernels.

Esta es la justificación teórica para la hipótesis del experimento: ante series generadas por el mismo mecanismo GP que el training data de Chronos, el modelo de fundación debería tener una ventaja estructural sobre cualquier modelo paramétrico de orden finito.

Los resultados de GP.3 T=200 confirman esta hipótesis de forma moderada: Chronos obtiene una ventaja de ~8% en RMSE sobre ETS(A,A,A). La ventaja no es dominante porque ETS también captura componentes tendencia+estacionalidad de manera razonablemente eficiente, y porque el kernel RBF+Periódico es solo uno de los muchos tipos de series en el training data de Chronos.

---

## Fidelidad de los DGPs al paper de Chronos (KernelSynth)

### KernelSynth según el paper: Algorithm 2 y Tabla 2

El paper de Chronos (Ansari et al. 2024, Apéndice D) describe el Algoritmo 2 con el siguiente **banco de kernels** (Tabla 2):

| Kernel | Fórmula | Hiperparámetros |
|--------|---------|----------------|
| Constant | $\kappa_{Const}(x,x') = C$ | $C = 1$ |
| White Noise | $\kappa_{White}(x,x') = \sigma_n \cdot \mathbf{1}_{x=x'}$ | $\sigma_n \in \{0.1, 1\}$ |
| Linear | $\kappa_{Lin}(x,x') = \sigma^2 + x \cdot x'$ | $\sigma \in \{0.1, 1, 10\}$ |
| RBF | $\kappa_{RBF}(x,x') = \exp(-\|x-x'\|^2 / 2\ell^2)$ | $\ell \in \{0.1, 1, 10\}$ |
| Rational Quadratic | $\kappa_{RQ}(x,x') = (1 + \|x-x'\|^2 / 2\alpha n)^{-\alpha}$ | $\alpha \in \{0.1,1,10\}$ |
| Periodic | $\kappa_{Per}(x,x') = \exp(-2\sin^2(\pi\|x-x'\|/p))$ | $p \in \{7,14,30,365,4,12,52,\ldots\}$ |

El procedimiento de composición (Algorithm 2) es:

```
j ~ U(1, 5)                            # número de kernels, entre 1 y 5
{κ_1, ..., κ_j} ~^{i.i.d.} K          # samplear kernels del banco
κ* = κ_1
for i in 2..j:
    ⋆ ~ {+, ×}                         # operador binario aleatorio
    κ* = κ* ⋆ κ_i                      # suma o producto de kernels
x_{1:1024} ~ GP(0, κ*)                 # serie de longitud 1024
```

**Proporciones de entrenamiento:** el paper reporta que ~10% de datos sintéticos KernelSynth sobre una base de datos reales produce la mejor performance (Figura 10b). El modelo no fue entrenado *exclusivamente* en GPs.

### Comparación: nuestra implementación vs. KernelSynth

| Aspecto | KernelSynth (paper) | Nuestra implementación | Impacto para la tesis |
|---------|--------------------|-----------------------|----------------------|
| Kernels disponibles | 6 tipos: Constant, White Noise, Linear, RBF, Rational Quadratic, Periodic | Solo RBF y Periodic | **Alto**: falta kernel Linear (tendencias lineales) y Rational Quadratic |
| Operadores de composición | `+` y `×` con igual probabilidad | Solo suma `+` | **Alto**: el producto `RBF × Periodic` genera "locally periodic" (patrón estacional que varía en amplitud) |
| Composición por serie | Aleatoria: $j \sim U(1,5)$ kernels distintos por serie | Fija: 1 o 2 kernels siempre iguales | **Medio**: nuestros experimentos son controlados, KernelSynth es estocástico |
| Lengthscale RBF | $\ell \in \{0.1, 1, 10\}$ | $\ell_{rbf} = 30$ | **Medio**: ver análisis de correlación abajo |
| Amplitud de kernels | Sin parámetro $\sigma$ explícito (implícitamente $\sigma=1$) | `sigma_rbf=1.0`, `sigma_per=0.8` | **Bajo**: escalar la covarianza no cambia la estructura matemática |
| Longitud de las series | $l_{syn} = 1024$ | $T \in \{50, 200\}$ | **Medio**: el paper entrena en series mucho más largas |

### Análisis de la diferencia en lengthscale RBF

El paper usa $\ell \in \{0.1, 1, 10\}$ para el kernel RBF. La correlación entre dos puntos separados por $h$ unidades de tiempo es $\exp(-h^2/2\ell^2)$:

| Lengthscale | Correlación en $h=30$ | Correlación en $h=5$ | Carácter |
|-------------|----------------------|---------------------|---------|
| $\ell=0.1$ | $\approx 0$ | $\approx 0$ | Casi ruido blanco |
| $\ell=1$ | $\approx 0$ | $\approx 0$ | Variaciones muy locales |
| $\ell=10$ | $\exp(-4.5) \approx 0.011$ | $\exp(-0.125) \approx 0.88$ | Suave en escala corta |
| $\ell=30$ (nuestro) | $\exp(-0.5) \approx 0.61$ | $\exp(-0.014) \approx 0.99$ | Tendencia muy suave |

Con los valores del paper ($\ell \leq 10$), el kernel RBF genera variaciones locales que en el horizonte $H=24$ se asemejan a ruido de alta frecuencia. Con nuestro $\ell=30$, la función GP tiene una tendencia suave y reconocible en el horizonte de pronóstico — lo que hace la comparación con ARIMA más ilustrativa para la tesis.

### Diferencia crítica: el operador producto (×)

El operador de multiplicación es responsable de los patrones más complejos del KernelSynth:

- $k_{RBF} \times k_{Per}$: **locally periodic** — patrón periódico con amplitud que varía suavemente en el tiempo. Más realista para muchas series económicas.
- $k_{Lin} \times k_{Per}$: **growing seasonal** — patrón estacional con amplitud que crece linealmente.
- $k_{RBF} \times k_{RBF}$: otro RBF (el producto de dos RBF sigue siendo RBF con $\ell$ más corto).

Nuestra implementación excluye este operador. El kernel compuesto de GP.3 ($K_{RBF} + K_{Per}$) genera tendencia y estacionalidad *independientes*, no una estacionalidad con amplitud variable.

### Justificación de las simplificaciones

Las diferencias entre nuestra implementación y KernelSynth son **decisiones de diseño experimental**, no errores:

1. **Kernels fijos con parámetros controlados**: Los experimentos Monte Carlo requieren condiciones reproducibles y comparables. Con $j \sim U(1,5)$ kernels aleatorios, cada replicación generaría un tipo de serie distinto — el experimento mediría la capacidad promedio sobre una mezcla heterogénea, no el comportamiento ante un proceso específico.

2. **Solo suma (+)**: El producto de kernels genera patrones más complejos que ameritan experimentos propios. La suma aditiva $k_{RBF} + k_{Per}$ produce la interpretación más directa: tendencia + estacionalidad independientes, el caso conceptualmente más limpio.

3. **$\ell_{rbf} = 30$**: Elegido para que la tendencia GP sea reconocible en el horizonte de pronóstico $H=24$. Con $\ell \leq 10$ (valores del paper), la función RBF sería prácticamente ruido blanco a ese horizonte, haciendo imposible detectar diferencias entre modelos.

4. **$T \in \{50, 200\}$**: El foco de la tesis es el comportamiento en muestras típicas de la práctica econométrica. El paper usa $T=1024$ para el entrenamiento de Chronos, no para sus experimentos de evaluación.

### Conclusión: fidelidad y validez de la hipótesis

Nuestros DGPs son **subconjuntos simplificados y controlados** de KernelSynth, no réplicas exactas. Los kernels RBF y Periodic del paper están incluidos; el muestreo GP con Cholesky es correcto; la composición aditiva es uno de los operadores válidos del paper.

La hipótesis de que **Chronos reconoce estos patrones porque fue entrenado en ellos** es **válida pero debe formularse con precisión**: Chronos fue entrenado en una mezcla de kernels mucho más diversa (incluyendo producto, lineal, y parámetros variables), no solo en las dos configuraciones específicas que evaluamos. La ventaja de Chronos en GP.3 T=200 (~8% sobre ETS) es real, pero la magnitud sería probablemente mayor si usáramos series más parecidas a las condiciones exactas de entrenamiento (más largas, con producto de kernels, con $\ell \in \{0.1,1,10\}$).
