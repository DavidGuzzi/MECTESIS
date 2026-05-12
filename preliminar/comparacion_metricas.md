# Comparación de métricas: protocolo propio vs. papers Chronos

## 1. Introducción

Este documento compara las métricas del protocolo de evaluación propio (`metricas.md`) con las usadas en los dos papers de referencia: **Chronos 1** (Ansari et al., 2024, *Chronos: Learning the Language of Time Series*) y **Chronos-2** (Ansari et al., 2025, *From Univariate to Universal Forecasting*). El objetivo es determinar si nuestras métricas son apropiadas y qué conviene agregar, mantener o eliminar.

**Contexto propio**: $R$ simulaciones Monte Carlo, horizontes $h = 1\text{–}24$, modelos ETS / Theta / Chronos / SeasonalNaive, bloques corto plazo ($h=1\text{–}12$) y mediano plazo ($h=13\text{–}24$).

---

## 2. Métricas de los papers Chronos

### 2.1 Chronos 1 — WQL y MASE

#### WQL (Weighted Quantile Loss)

La idea central: evaluar la distribución predictiva completa sin asumir ninguna forma paramétrica. En lugar de calcular un único error, se evalúa el modelo en $K = 9$ niveles de cuantil $\alpha \in \{0.1, 0.2, \ldots, 0.9\}$.

**Paso 1 — Quantile Loss asimétrica** para un cuantil $\alpha$, valor predicho $q$ y valor real $x$:

$$QL_\alpha(q,\, x) = \begin{cases} \alpha\,(x - q) & \text{si } x > q \quad \text{(subestimación)} \\ (1-\alpha)\,(q - x) & \text{si } x \leq q \quad \text{(sobreestimación)} \end{cases}$$

La asimetría es clave: para $\alpha = 0.9$ la penalidad por subestimar es 9 veces mayor que por sobreestimar, lo que induce al modelo a aprender el verdadero cuantil 90%.

**Paso 2 — WQL por cuantil** (normalizado por los valores reales):

$$WQL_\alpha = \frac{2\displaystyle\sum_{t,i} QL_\alpha\!\left(q_{t,i}^\alpha,\, x_{t,i}\right)}{\displaystyle\sum_{t,i} |x_{t,i}|}$$

La normalización por $\sum |x_{t,i}|$ hace la métrica **libre de escala**: permite comparar series de distintas magnitudes.

**Paso 3 — WQL agregado** sobre los $K$ cuantiles:

$$WQL = \frac{1}{K} \sum_{j=1}^K WQL_{\alpha_j}$$

> **Relación con CRPS**: El CRPS es la integral continua $\int_0^1 2\,QL_\alpha\,d\alpha$. El WQL con $K$ cuantiles uniformes es una aproximación de Riemann del CRPS sobre el intervalo $[0.1,\, 0.9]$; ignora las colas extremas ($\alpha < 0.1$ y $\alpha > 0.9$).

#### MASE (Mean Absolute Scaled Error)

Evalúa el error puntual escalándolo contra el error que cometería un **naive estacional** sobre la propia muestra de entrenamiento. El resultado es adimensional y comparable entre series.

**Denominador** — MAE del naive estacional sobre los datos de entrenamiento ($C$ observaciones, estacionalidad $S$):

$$\text{MAE}_{\text{naive}} = \frac{1}{C - S} \sum_{t=S+1}^{C} |y_t - y_{t-S}|$$

**MASE**:

$$MASE = \frac{\dfrac{1}{H}\displaystyle\sum_{h=1}^H \left|\hat{y}_{T+h} - y_{T+h}\right|}{\text{MAE}_{\text{naive}}}$$

- $MASE < 1$: el modelo supera al naive estacional.
- $MASE > 1$: el modelo es peor que el naive estacional.
- $MASE = 1$: empatan.

**Protocolo de agregación en Chronos 1**: cada score del modelo se divide por el score de Seasonal Naive y los ratios se agregan con **media geométrica** sobre todos los datasets. Esto evita que datasets con métricas absolutas grandes dominen el promedio.

---

### 2.2 Chronos-2 — SQL, WQL, MASE, WAPE, Win Rate, Skill Score

Chronos-2 introduce nuevas métricas de comparación y extiende WQL. Se explica cada una desde cero.

#### SQL (Scaled Quantile Loss)

Chronos-2 no tokeniza la serie como texto; en cambio, predice directamente cuantiles mediante regresión cuantílica. El objetivo de entrenamiento es:

$$\mathcal{L} = \sum_{q \in Q} \Big[ q \cdot \max(z - z^q,\, 0) + (1-q) \cdot \max(z^q - z,\, 0) \Big]$$

donde $z$ es el valor real (normalizado), $z^q$ es el cuantil predicho al nivel $q$, y $Q$ tiene **21 niveles**:

$$Q = \{0.01,\; 0.05,\; 0.1,\; 0.2,\; \ldots,\; 0.8,\; 0.9,\; 0.95,\; 0.99\}$$

Los extremos $0.01$ y $0.99$ se agregan (respecto a los 9 cuantiles de Chronos 1) para capturar mejor eventos raros y facilitar tareas de detección de anomalías.

**SQL vs. WQL**: SQL promedia las pérdidas cuantílicas sin normalizar por los valores absolutos del target (a diferencia del WQL). Esto lo hace más adecuado como **función de pérdida** durante el entrenamiento, mientras que WQL (normalizado) se usa como métrica de evaluación comparativa.

#### WQL en Chronos-2

Idéntico al de Chronos 1 (Paso 1–3 arriba). Se evalúa con $K=9$ cuantiles sobre los conjuntos de test.

#### MASE en Chronos-2

Idéntica fórmula que en Chronos 1. Se reporta como **Skill Score** (ver abajo) para facilitar la interpretación.

#### WAPE (Weighted Absolute Percentage Error)

$$WAPE = \frac{\displaystyle\sum_{h=1}^H \left|\hat{y}_{T+h} - y_{T+h}\right|}{\displaystyle\sum_{h=1}^H \left|y_{T+h}\right|}$$

Es la suma de errores absolutos normalizada por la suma de valores reales en el horizonte de pronóstico.

- **No confundir con MAPE**: MAPE promedia los errores porcentuales por observación ($\frac{1}{H}\sum|e_h/y_{T+h}|$), lo que puede explotar cuando $y_{T+h} \approx 0$. WAPE pondera por el volumen total, siendo más estable.
- Se usa principalmente en dominios de **retail y negocios** donde los valores son siempre positivos y la magnitud total importa (ej. ventas totales de una tienda).

#### Win Rate

Compara modelos de forma **por pares** sobre múltiples tareas (combinaciones dataset × horizonte).

**Construcción**:

1. Sea $\mathcal{T}$ el conjunto de tareas. Para cada tarea $\tau \in \mathcal{T}$, calcular la métrica (ej. WQL) de cada modelo.
2. Para cada par de modelos $(A, B)$ y cada tarea $\tau$:
   $$w_{A,B}^\tau = \mathbf{1}\!\left[\text{métrica}_A^\tau < \text{métrica}_B^\tau\right]$$
3. Win Rate de $A$ sobre $B$:
   $$WR_{A,B} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} w_{A,B}^\tau$$
4. Win Rate global de $A$:
   $$WR_A = \frac{1}{N-1} \sum_{B \neq A} WR_{A,B}$$

Los **intervalos de confianza al 95%** se obtienen por bootstrap sobre las tareas: se remuestrea $\mathcal{T}$ con reemplazo $B=1000$ veces y se calculan los percentiles 2.5 y 97.5 del Win Rate.

**Interpretación**: $WR_A = 60\%$ significa que, en promedio, modelo $A$ supera a sus competidores en el 60% de las tareas evaluadas.

#### Skill Score

Mide la **mejora porcentual** de un modelo sobre un baseline de referencia (Seasonal Naive):

$$\text{Skill}_{A} = \left(1 - \frac{\text{métrica}_A}{\text{métrica}_{\text{baseline}}}\right) \times 100\%$$

- **Positivo**: $A$ mejora sobre el baseline.
- **Negativo**: $A$ es peor que el baseline.
- **0%**: empatan exactamente.
- **100%**: el modelo tiene error cero.

Los IC 95% también se calculan por bootstrap.

**¿Por qué Skill Score es más útil que reportar la métrica cruda?** Permite comparar modelos en distintos datasets con distintas escalas de dificultad: un $MASE = 0.80$ en una serie fácil no equivale a $MASE = 0.80$ en una serie difícil. El Skill Score siempre responde "¿cuánto mejor que el naive?".

---

## 3. Tabla de correspondencia

| Dimensión evaluada | Chronos 1 | Chronos-2 | Nuestras métricas |
|---|---|---|---|
| Error puntual (escala-dependiente) | — | — | RMSE, MAE |
| Error puntual (escala-libre) | MASE | MASE | — |
| Error puntual relativo (%) | — | WAPE | — |
| Sesgo sistemático | — | — | Bias |
| Variabilidad del error | — | — | Varianza |
| Distribución predictiva completa | WQL | SQL, WQL | CRPS |
| Calibración de intervalos | implícita en WQL | implícita en SQL/WQL | cov_80, cov_95 |
| Amplitud de intervalos | implícita en WQL | implícita en SQL/WQL | width_80, width_95 |
| Score unificado (cobertura + amplitud) | — | — | Winkler_80, Winkler_95 |
| Comparación relativa al baseline | ratio + geom. mean | Win Rate, Skill Score | — |

---

## 4. Análisis crítico: ¿están bien nuestras métricas?

### 4.1 Lo que tenemos y es valioso (aunque los papers no lo usen)

**Bias y Varianza**. Los papers omiten estas métricas porque agregan resultados sobre decenas de datasets heterogéneos y necesitan resúmenes compactos. En nuestro contexto de simulaciones Monte Carlo sobre un DGP controlado, la descomposición $MSE = \text{Bias}^2 + \text{Varianza}$ es **diagnósticamente superior**: permite distinguir si un modelo falla por sesgo sistemático (ej. Theta subestima tendencias) o por variabilidad excesiva (ej. ETS explota en horizontes largos). Esto es imposible de detectar solo con MSE o MASE.

**cov_80, cov_95, width_80, width_95**. WQL y SQL capturan calibración e amplitud *implícitamente* (un cuantil mal calibrado tiene mayor QL). Pero reportarlos **explícitamente** aporta transparencia: permite afirmar "Chronos cubre el 94% de los valores reales con IP al 95% pero ETS solo el 86%", algo que no es legible del WQL sin desagregarlo por cuantil.

**Winkler Score**. Unifica cobertura y amplitud en un score único con una penalidad fuerte por las salidas (×10 para 80%, ×40 para 95%). No aparece en los papers de Chronos pero sí en la competencia M4 (Makridakis et al., 2020) y es estándar en la literatura de probabilistic forecasting. Es más interpretable que WQL para el lector no especializado.

**CRPS**. Conceptualmente más completo que WQL: integra sobre toda la distribución incluidas las colas (no solo $\alpha \in [0.1, 0.9]$). Para modelos ensemble (ETS, Chronos) calculado desde las trayectorias simuladas, el CRPS ensemble es exacto; para Theta se asume gaussianidad.

### 4.2 Lo que usan los papers y nos falta

**MASE** es la brecha más importante. Sin MASE no podemos comparar nuestros resultados directamente con las tablas del paper. Si el paper reporta que Chronos-T5 alcanza $MASE_{\text{rel}} = 0.935$ en Benchmark I, necesitamos calcular MASE con la misma fórmula para saber si nuestro DGP reproduce esa relación. RMSE y MAE informan sobre el error en unidades absolutas, pero no son comparables con los valores publicados.

**WQL normalizado** (opcional). Nuestro CRPS no está normalizado por los valores absolutos del target, a diferencia del WQL de los papers. Para reproducir exactamente la métrica del paper habría que calcular $WQL = CRPS\text{-ensemble} / (2\bar{y})$ aproximadamente. Dado que ya tenemos CRPS, la prioridad es baja salvo que se quiera reportar el ratio exacto.

### 4.3 ¿CRPS ≈ WQL? Cuándo difieren

La relación formal es:

$$\text{CRPS}(F, x) = \int_0^1 2\,QL_\alpha\!\left(F^{-1}(\alpha),\, x\right) d\alpha$$

WQL con $K=9$ cuantiles uniformes en $[0.1, 0.9]$ aproxima esta integral solo sobre esa región: **ignora las colas** ($\alpha < 0.1$ y $\alpha > 0.9$). Si la distribución predictiva tiene colas pesadas o asimétricas, el CRPS ensemble (calculado desde 500 trayectorias) captura mejor el comportamiento extremo y será mayor que WQL. En series estacionarias bien portadas la diferencia es mínima; en series con alta volatilidad o eventos extremos puede ser significativa.

**Conclusión**: CRPS es más general. No conviene reemplazarlo por WQL; lo que falta es agregar MASE para facilitar la comparación con benchmarks.

---

## 5. Recomendaciones

### Mantener sin cambios

| Métrica | Razón |
|---|---|
| Bias, Varianza | Diagnóstico único en simulaciones MC; imposible de derivar de MASE o CRPS |
| RMSE, MAE | Informan el error en unidades de la serie; interpretación sustantiva directa |
| cov_80, cov_95 | Calibración explícita y pedagógicamente clara |
| width_80, width_95 | Eficiencia de los intervalos separada de la calibración |
| CRPS | Métrica probabilística más completa (incluye colas); más general que WQL |
| Winkler_80, Winkler_95 | Score unificado para intervalos; estándar en competencias de forecasting |

### Agregar

| Métrica | Prioridad | Razón |
|---|---|---|
| **MASE** | **Alta** | Permite comparación directa con benchmarks publicados de Chronos; denominador = $\text{MAE}_{\text{naive estacional}}$ con $S=12$ para series mensuales |
| WQL normalizado | Baja | Solo si se quiere reproducir exactamente la métrica del paper; CRPS ya cubre la misma dimensión |

### No agregar

| Métrica | Razón |
|---|---|
| WAPE | Útil en retail/negocios; no aporta sobre MAE normalizado en series macroeconómicas/financieras |
| Win Rate | Métrica de agregación entre datasets; con un DGP por experimento no aplica directamente |
| Skill Score | Puede calcularse post-hoc como $(1 - \text{métrica}/\text{métrica}_{\text{Naive}}) \times 100\%$ si se desea; no requiere implementación separada |

---

## Apéndice: construcción paso a paso con ejemplos numéricos

### A. Quantile Loss y WQL

$R = 3$ réplicas, horizonte $h=1$. Valores reales: $y = [100,\; 102,\; 98]$.

**Cuantil $\alpha = 0.1$** predicho por el modelo: $q^{0.1} = [97,\; 99,\; 96]$.

| Réplica | $y$ | $q^{0.1}$ | $y > q$? | $QL_{0.1}$ |
|---|---|---|---|---|
| 1 | 100 | 97 | sí | $0.1 \times (100-97) = 0.30$ |
| 2 | 102 | 99 | sí | $0.1 \times (102-99) = 0.30$ |
| 3 | 98 | 96 | sí | $0.1 \times (98-96) = 0.20$ |

**Cuantil $\alpha = 0.9$** predicho: $q^{0.9} = [104,\; 106,\; 101]$.

| Réplica | $y$ | $q^{0.9}$ | $y > q$? | $QL_{0.9}$ |
|---|---|---|---|---|
| 1 | 100 | 104 | no | $(1-0.9)\times(104-100) = 0.40$ |
| 2 | 102 | 106 | no | $0.1\times(106-102) = 0.40$ |
| 3 | 98 | 101 | no | $0.1\times(101-98) = 0.30$ |

$$WQL_{0.1} = \frac{2\times(0.30+0.30+0.20)}{100+102+98} = \frac{1.60}{300} = 0.00533$$

$$WQL_{0.9} = \frac{2\times(0.40+0.40+0.30)}{300} = \frac{2.20}{300} = 0.00733$$

$$WQL = \frac{0.00533 + 0.00733}{2} = 0.00633 \quad \text{(promediado sobre 2 cuantiles; con }K=9\text{ igual)}$$

---

### B. MASE

Serie de entrenamiento $C = 24$ meses, estacionalidad $S = 12$.

**Denominador** (MAE del naive estacional in-sample):

$$\text{MAE}_{\text{naive}} = \frac{1}{24-12} \sum_{t=13}^{24} |y_t - y_{t-12}| = 5.0 \text{ unidades}$$

**MAE del modelo** sobre $h=1,\ldots,12$: $\hat{e} = 3.5$ unidades.

$$MASE = \frac{3.5}{5.0} = 0.70$$

El modelo supera al naive estacional en un 30%.

---

### C. Skill Score

$$\text{Skill}_{\text{ETS}} = \left(1 - \frac{MASE_{\text{ETS}}}{MASE_{\text{Naive}}}\right)\times 100 = \left(1 - \frac{0.70}{1.00}\right)\times 100 = 30\%$$

$$\text{Skill}_{\text{Chronos}} = \left(1 - \frac{0.85}{1.00}\right)\times 100 = 15\%$$

ETS mejora un 30% sobre el naive; Chronos un 15%.

---

### D. Win Rate (ejemplo simplificado)

3 modelos, 4 tareas. MASE por tarea (menor = mejor):

| Tarea | ETS | Chronos | Naive |
|---|---|---|---|
| 1 | 0.70 | 0.85 | 1.00 |
| 2 | 0.90 | 0.80 | 1.00 |
| 3 | 0.65 | 0.75 | 1.00 |
| 4 | 1.10 | 0.95 | 1.00 |

$WR_{\text{ETS vs Chronos}}$: ETS gana en tareas 1, 3 → $WR = 2/4 = 50\%$

$WR_{\text{ETS vs Naive}}$: ETS gana en tareas 1, 2, 3 → $WR = 3/4 = 75\%$

$WR_{\text{ETS global}} = (50 + 75)/2 = 62.5\%$

Los IC 95% se calculan remuestreando las 4 tareas con reemplazo 1000 veces y tomando los percentiles 2.5 y 97.5 del $WR$ resultante.
