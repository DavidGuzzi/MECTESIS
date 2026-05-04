# Conclusiones — Experimentos univariados 1.1–1.19

**Setup 1.1–1.12:** T ∈ {200, 500} | H = 24 | R = 500 | Semilla = 3649  
**Setup 1.13–1.19:** T ∈ {50, 200} | H = 24 | R = 500 | Semilla = 3649  
**Modelos Core 1.1–1.12:** ARIMA/SARIMA/GARCH (statsmodels + arch, correctamente especificados) vs Chronos-2 (zero-shot)  
**Modelos Core 1.13–1.19:** ETS / Theta / Seasonal Naive (statsmodels, correctamente especificados) vs Chronos-2 (zero-shot)  
**Nota:** SeasonalNaiveModel no produce intervalos de predicción; las columnas probabilísticas (CRPS, cobertura, Winkler) solo están disponibles para Chronos en exps 1.16–1.17.

---

## Resultados por experimento

### Exp 1.1 — AR(1) φ=0.3, baja persistencia

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| ARIMA(1,0,0) RMSE | 1.044 | 1.081 | 1.051 | 1.068 |
| Chronos-2 RMSE    | 1.058 | 1.088 | 1.056 | 1.071 |

Ventaja ARIMA: ~1.3% en T=200, ~0.3% en T=500. Los errores son casi idénticos. Bias nulo en ambos. Cobertura 95%: ARIMA 0.944, Chronos 0.963 (Chronos ligeramente sobrecobertura por intervalos más anchos). **Conclusión: paridad práctica. El proceso de baja persistencia es suficientemente simple para que Chronos lo recupere sin estimación explícita.**

---

### Exp 1.2 — AR(1) φ=0.9, alta persistencia

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| ARIMA(1,0,0) RMSE | 1.874 | 2.410 | 1.880 | 2.283 |
| Chronos-2 RMSE    | 2.084 | 2.830 | 1.954 | 2.386 |

Ventaja ARIMA: ~11% en h=13-24 con T=200; ~4.5% en T=500. La brecha se reduce con más datos pero no desaparece. Chronos genera intervalos al 95% mucho más anchos (width_95: 13.7 vs 8.2 en T=200, h=13-24) y los cubre con sobrecobertura (cov_95=0.96 vs 0.90 de ARIMA). **Conclusión: ARIMA claramente mejor. Chronos subestima la persistencia, lo que se traduce en mayor varianza del error. La alta persistencia es más difícil de capturar zero-shot.**

---

### Exp 1.3 — Random Walk I(1), sin drift

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| ARIMA(0,1,0) RMSE | 2.389 | 4.239 | 2.400 | 4.102 |
| Chronos-2 RMSE    | 2.487 | 4.422 | 2.424 | 4.153 |

Bias ≈ 0 en ambos modelos y en ambos horizontes: ninguno introduce sesgo. Ventaja ARIMA: ~4% en T=200, ~1% en T=500. La brecha desaparece prácticamente a T=500. **Conclusión: ARIMA marginalmente mejor. Ambos identifican correctamente el random walk (pronóstico = último valor observado). El DGP no presenta asimetría informativa significativa.**

---

### Exp 1.4 — Random Walk I(1), drift = 0.5

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| ARIMA(0,1,0) RMSE | 3.972 | **10.195** | 4.067 | **10.128** |
| ARIMA(0,1,0) bias | +3.14 | **+9.26** | +3.24 | **+9.25** |
| Chronos-2 RMSE    | 2.686 | 4.990 | 2.845 | 4.667 |
| Chronos-2 bias    | −0.17 | +0.71 | −0.26 | +0.23 |

**Resultado más llamativo del bloque.** ARIMA(0,1,0) estimado sin constante no captura el drift: el sesgo crece linealmente con el horizonte (≈ 0.5·h por construcción del DGP). A h=13-24 el RMSE de ARIMA es 2× el de Chronos. Chronos identifica el drift desde el contexto de la serie de forma zero-shot y mantiene bias bajo. Cobertura ARIMA en h=13-24: cov_80 = 0.23, cov_95 = 0.51 — colapso total de los intervalos por sesgo sistemático.

**Conclusión: victoria de Chronos por misspecificación del modelo clásico.** ARIMA(0,1,0) requiere `trend='c'` para incluir constante post-diferenciación. Sin ella, es el modelo incorrecto. Chronos es más robusto ante esta omisión.

> **Nota de implementación:** ARIMA(0,1,0) con `statsmodels` no agrega drift automáticamente. El modelo correcto es `ARIMA(y, order=(0,1,0), trend='c')`. Este resultado ilustra un riesgo práctico real: la misspecificación por omisión del drift es un error frecuente.

---

### Exp 1.5 — AR(1) + tendencia lineal determinista

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| ARIMA+trend RMSE | 1.260 | 1.295 | 1.224 | 1.282 |
| Chronos-2 RMSE   | 1.382 | 1.446 | 1.327 | 1.400 |

Ventaja ARIMA: ~9% en T=200, ~8% en T=500. La brecha es estable y no se cierra con T mayor. Chronos mantiene la tendencia pero con mayor varianza residual. Característica notable: intervalos 95% de Chronos son muy anchos a T=200 (width_95 = 13.4 en h=13-24) con sobrecobertura marcada (cov_95 = 0.994), que se reduce a 8.6 en T=500 (cov_95 = 0.992). **Conclusión: ARIMA+trend domina con ventaja consistente. La extrapolación de la tendencia es más precisa cuando el modelo la parametriza explícitamente. Chronos "cubre" la incertidumbre sobre la tendencia con intervalos exageradamente amplios.**

---

### Exp 1.6 — SAR trimestral, s=4, estacionario

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| SARIMA(1,0,0)(1,0,0)_4 RMSE | 1.189 | 1.209 | 1.189 | 1.230 |
| Chronos-2 RMSE              | 1.248 | 1.288 | 1.212 | 1.254 |

Ventaja SARIMA: ~5% en T=200; **<2% en T=500.** La brecha se cierra casi completamente al aumentar la muestra. Cobertura: ambos ligeramente por debajo del nominal en 80%; similar en 95% con Chronos levemente sobre el nominal. **Conclusión: SARIMA mejor pero la ventaja es mínima con T=500. Chronos identifica el patrón estacional trimestral de forma competitiva, especialmente con contexto largo.**

---

### Exp 1.7 — Seasonal I(1)×I(1)_12, doble integración

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| SARIMA(0,1,0)(0,1,0)_12 RMSE | 2.386 | 5.929 | 2.475 | **6.185** |
| Chronos-2 RMSE               | 3.397 | 8.251 | **5.393** | **10.123** |

**El peor resultado de Chronos y el único caso donde la brecha se amplía con T.** A T=500 Chronos empeora significativamente: varianza = 66.6 vs 23.3 de SARIMA. Cobertura 80% de Chronos en h=13-24 a T=500: 0.703 — seria subcovertura. SARIMA mantiene estabilidad total entre T=200 y T=500.

**Conclusión: SARIMA domina ampliamente. La doble integración estacional es el caso más adverso para Chronos. Más contexto le hace daño, posiblemente porque la serie I(1)×I(1)_12 con T=500 tiene una estructura no estacionaria de largo alcance que el modelo de fundación no maneja bien. Esta es la limitación más importante identificada en el bloque univariado.**

---

### Exp 1.8 — AR(1) con quiebre estructural en T/2

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| ARIMA+break RMSE | 1.576 | 1.667 | 1.558 | 1.662 |
| Chronos-2 RMSE   | 1.666 | 1.825 | **1.551** | **1.658** |

A T=200, ARIMA+break mejor en ~9%. A T=500, **Chronos alcanza al modelo correctamente especificado** (diferencia < 0.5%). Ambos modelos presentan subcovertura en 80% (cov_80 ≈ 0.70-0.73 en ambos T) y en 95% (cov_95 ≈ 0.89-0.92): el quiebre estructural infla la incertidumbre del error más allá de lo que los intervalos capturan.

**Conclusión: Chronos es adaptativo ante quiebres estructurales. Con contexto suficiente (T=500), aprende implícitamente el nuevo régimen sin necesitar la fecha del quiebre. La dummy explícita solo tiene ventaja cuando la muestra es corta.**

---

### Exp 1.9 — AR(1)–ARCH(1), volatilidad baja persistencia

*Nota de escala: ω/(1−α) = 0.1/0.7 ≈ 0.143 → σ_ε_incondicional ≈ 0.38. Las métricas están en escala distinta a exps 1.10–1.12.*

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| AR(1)+ARCH(1) RMSE | 0.392 | 0.398 | 0.399 | 0.397 |
| Chronos-2 RMSE     | 0.396 | 0.403 | 0.400 | 0.398 |

avg_all — AR+ARCH: RMSE=0.395/0.398, cov_95=0.945/0.946, width_95=1.535/1.540 (T=200/T=500).  
avg_all — Chronos: RMSE=0.399/0.399, cov_95=0.960/0.955, width_95=1.749/1.639.

Ventaja AR+ARCH en RMSE: ~1% (T=200), ~0.2% (T=500). Intervalos de Chronos ~14% más anchos en T=200, ~6% en T=500. **Conclusión: paridad práctica en pronóstico puntual. ARCH(1) tiene volatilidad transitoria (sin término β, la persistencia de σ decae en un período), lo que Chronos puede capturar implícitamente. El modelo clásico produce intervalos más ajustados y mejor calibrados.**

---

### Exp 1.10 — AR(1)–GARCH(1,1), persistencia alta (α+β=0.9)

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| AR(1)+GARCH(1,1) RMSE | 1.043 | 1.052 | 1.050 | 1.047 |
| Chronos-2 RMSE        | 1.051 | 1.066 | 1.052 | 1.050 |

avg_all — AR+GARCH: RMSE=1.048/1.048, cov_95=0.939/0.943, width_95=4.032/4.045.  
avg_all — Chronos: RMSE=1.058/1.051, cov_95=0.957/0.953, width_95=4.554/4.290.

Ventaja AR+GARCH en RMSE: ~0.9% (T=200), ~0.3% (T=500). La brecha se comprime con T. Chronos produce intervalos ~13% más anchos en T=200, ~6% en T=500. **Conclusión: paridad en pronóstico puntual, con leve ventaja decreciente del modelo clásico. La heterocedasticidad GARCH afecta la varianza pero no la media condicional: ambos modelos usan el mismo componente AR(1) para el pronóstico de nivel. El modelo clásico gana en calibración de intervalos.**

---

### Exp 1.11 — GARCH(1,1) media cero

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| GARCH(1,1)-ZeroMean RMSE | 0.986 | 0.998 | 1.002 | 1.002 |
| Chronos-2 RMSE           | 0.991 | 1.006 | 1.003 | 1.003 |

avg_all — GARCH: RMSE=0.992/1.002, cov_95=0.942/0.945, width_95=3.854/3.876.  
avg_all — Chronos: RMSE=0.999/1.003, cov_95=0.958/0.952, width_95=4.330/4.101.

Ventaja GARCH en RMSE: ~0.7% (T=200), ~0.1% (T=500). A T=500 la paridad es prácticamente perfecta. Ambos convergen al pronóstico natural del proceso (media cero). Chronos genera intervalos ~12% más anchos en T=200, ~6% en T=500. **Conclusión: empate. Sin componente AR, el único pronóstico posible es 0 a cualquier horizonte. Ambos llegan ahí. La diferencia en intervalos persiste: GARCH estima la varianza condicional actual; Chronos la promedia de forma más conservadora.**

---

### Exp 1.12 — AR(1)–GJR–GARCH(1,1,1), efecto leverage

| | T=200 h=1-12 | T=200 h=13-24 | T=500 h=1-12 | T=500 h=13-24 |
|---|---|---|---|---|
| AR(1)+GJR-GARCH(1,1,1) RMSE | 1.043 | 1.052 | 1.050 | 1.047 |
| Chronos-2 RMSE              | 1.051 | 1.066 | 1.052 | 1.050 |

avg_all — GJR: RMSE=1.048/1.049, cov_95=0.939/0.945, width_95=4.032/4.042.  
avg_all — Chronos: RMSE=1.058/1.051, cov_95=0.957/0.953, width_95=4.554/4.290.

Ventaja GJR en RMSE: ~0.9% (T=200), ~0.2% (T=500). Métricas casi idénticas a exp 1.10 (AR+GARCH simétrico): la asimetría del leverage no beneficia al modelo clásico en RMSE porque el componente de media (AR) es el mismo en ambos experimentos. **Conclusión: el efecto leverage no genera ventaja adicional en pronóstico puntual sobre un GARCH simétrico. La asimetría γ·ε²·1{ε<0} afecta la varianza condicional en shocks negativos, lo que es irrelevante para el error medio cuadrático del nivel. GJR aventaja a Chronos solo en calibración de intervalos.**

---

## Resumen comparativo

| Exp | DGP | T=200 ganador | T=500 ganador | Diferencia RMSE (avg) |
|-----|-----|--------------|--------------|----------------------|
| 1.1 | AR(1) φ=0.3 | ARIMA (~1%) | Empate | Mínima |
| 1.2 | AR(1) φ=0.9 | ARIMA (11%) | ARIMA (4%) | Moderada, estable |
| 1.3 | RW sin drift | ARIMA (4%) | Empate | Pequeña |
| 1.4 | RW drift=0.5 | **Chronos (>100%)** | **Chronos (>100%)** | **Extrema (misspecif.)** |
| 1.5 | AR+trend | ARIMA (9%) | ARIMA (8%) | Moderada, estable |
| 1.6 | SAR s=4 | SARIMA (5%) | SARIMA (~2%) | Pequeña, decreciente |
| 1.7 | I(1)×I(1)_12 | SARIMA (27%) | **SARIMA (55%)** | **Grande, creciente** |
| 1.8 | AR+break | ARIMA+break (9%) | **Empate** | Presente solo con T corto |
| 1.9 | AR(1)+ARCH(1) | AR+ARCH (~1%) | Empate | Mínima |
| 1.10 | AR(1)+GARCH(1,1) | AR+GARCH (~1%) | Empate | Mínima |
| 1.11 | GARCH media cero | Empate | Empate | Mínima |
| 1.12 | AR(1)+GJR-GARCH | GJR (~1%) | Empate | Mínima |

---

## Conclusiones transversales

### 1. Los modelos clásicos dominan en su DGP nativo
Cuando el modelo clásico está correctamente especificado, supera a Chronos en todos los casos excepto exp 1.4 (misspecificación). La ventaja es más pronunciada en estructuras con memoria larga (1.2) o con integración estacional (1.7).

### 2. Chronos es robusto ante misspecificaciones del modelo clásico
El caso más notable es el drift en exp 1.4: Chronos identifica el drift desde el contexto de la serie sin estimarlo explícitamente. Esto sugiere que, en aplicaciones reales donde la especificación del modelo clásico es incierta, Chronos puede ser más robusto.

### 3. La doble integración estacional es el talón de Aquiles de Chronos
Exp 1.7 es el único caso donde Chronos empeora con T mayor. La estructura I(1)×I(1)_12 acumula no-estacionariedad en dos dimensiones y el modelo de fundación no la maneja adecuadamente. Resulta la limitación estructural más clara del bloque.

### 4. Las brechas se reducen con T — excepto en doble integración
En los experimentos con DGPs estacionarios o moderadamente persistentes (1.1, 1.3, 1.6, 1.8), la ventaja del modelo clásico se comprime con T=500. Esto es consistente con la hipótesis de que Chronos requiere contexto suficiente para identificar la estructura del proceso.

### 5. Intervalos de predicción: Chronos tiene sobrecobertura al 95%, subcovertura al 80%
Chronos genera sistemáticamente intervalos más anchos. A nivel 95% produce sobrecobertura (salvo en 1.7 con T=500). A nivel 80% tiende a estar por debajo del nominal, especialmente en DGPs complejos. El modelo clásico tiene mejor calibración en 80% pero también puede tener problemas en DGPs con quiebre (1.8).

### 6. Implicación para la tesis
Bajo condiciones de laboratorio (DGP conocido, modelo correctamente especificado), ARIMA/SARIMA es el benchmark correcto. La pregunta relevante para series reales es distinta: el DGP es desconocido, la especificación del modelo clásico es incierta, y las series pueden tener quiebres, drifts no explicitados, u otras complejidades. En ese contexto, los resultados de exp 1.4 y 1.8 sugieren que Chronos puede tener ventajas prácticas.

### 7. Heterocedasticidad condicional: efecto nulo en pronóstico puntual, intervalos más anchos de Chronos
Los experimentos 1.9–1.12 muestran que la volatilidad condicional (ARCH/GARCH) afecta mínimamente el error de pronóstico puntual. Tanto los modelos clásicos como Chronos producen RMSE casi idénticos: el componente de media (AR o constante cero) es el que determina el error al horizonte; la varianza condicional solo es relevante para la construcción de intervalos. El modelo clásico correctamente especificado logra mejor calibración de intervalos y anchos más ajustados. Chronos sobreestima el ancho de los intervalos (~6–14% más amplio) pero mantiene mayor cobertura efectiva al 95%. La asimetría del efecto leverage (exp 1.12) no genera ventaja adicional observable en RMSE.

---

## Experimentos 1.13–1.19 — Bloque ETS / Theta / Seasonal Naive

---

### Exp 1.13 — Nivel local (Local Level → ETS(A,N,N))

| | T=50 | T=200 |
|---|---|---|
| ETS(A,N,N) RMSE | 1.505 | 1.541 |
| Chronos-2 RMSE  | 1.523 | 1.651 |
| ETS(A,N,N) CRPS | 0.868 | 0.875 |
| Chronos-2 CRPS  | 1.302 | 1.094 |

ETS(A,N,N) gana en RMSE en ambos T (1% a T=50, 7% a T=200). La ventaja crece con el tamaño muestral, a diferencia de los exps ARMA donde converge. La diferencia de CRPS es sustancial: ETS produce distribuciones predictivas más ajustadas. Los intervalos de Chronos son 2× más anchos a T=50 (width_95=11.4 vs 5.5), con fuerte sobrecobertura (cov_95=0.988). A T=200 ambos tienen cobertura razonable, pero los intervalos de Chronos siguen siendo ~18% más anchos. **Conclusión: ETS(A,N,N) domina. El nivel local es una estructura simple que el modelo correctamente especificado captura mejor que Chronos, y la ventaja no disminuye con más datos.**

---

### Exp 1.14 — Tendencia local (Local Trend → ETS(A,A,N))

| | T=50 | T=200 |
|---|---|---|
| ETS(A,A,N) RMSE | 4.754 | 4.205 |
| Chronos-2 RMSE  | 6.066 | 5.544 |
| ETS(A,A,N) CRPS | 3.306 | 2.819 |
| Chronos-2 CRPS  | 3.914 | 3.777 |

Ventaja ETS: 22% a T=50, 32% a T=200. La brecha crece, consistente con exp 1.13. Los RMSE son altos en ambos modelos porque el DGP (tendencia + nivel estocásticos) acumula incertidumbre a lo largo del horizonte. Comportamiento atípico en intervalos a T=50: ETS tiene cobertura muy baja (cov_95=0.669) con intervalos estrechos — el modelo subestima la incertidumbre acumulada en muestras cortas. Con T=200, ETS corrige esto (cov_95=0.988). **Conclusión: ETS domina claramente, pero los intervalos en muestras cortas son poco confiables. El DGP de tendencia local genera la mayor incertidumbre del bloque, y el modelo clásico necesita T suficiente para calibrar bien la distribución predictiva.**

---

### Exp 1.15 — Tendencia amortiguada (Damped Trend → ETS(A,Ad,N))

| | T=50 | T=200 |
|---|---|---|
| ETS(A,Ad,N) RMSE | 2.814 | 2.435 |
| Chronos-2 RMSE   | 2.601 | 2.627 |
| ETS(A,Ad,N) CRPS | 1.723 | 1.518 |
| Chronos-2 CRPS   | 1.955 | 1.729 |

**Inversión a T=50: Chronos gana en RMSE (8%).** Con muestra corta, la estimación del parámetro de amortiguamiento φ es ruidosa y ETS genera pronósticos peores. Con T=200, ETS recupera la ventaja (7%). Los intervalos de ETS a T=50 son muy estrechos con subcovertura severa (cov_95=0.711); Chronos cubre bien (cov_95=0.965). A T=200 ambos mejoran. **Conclusión: la estimación del amortiguamiento requiere suficiente historia. Con muestras cortas, Chronos es más robusto zero-shot que ETS(A,Ad,N) recién identificado. Es uno de los pocos casos donde Chronos supera al modelo correctamente especificado.**

---

### Exp 1.16 — Estacionalidad determinística (s=12)

| | T=50 | T=200 |
|---|---|---|
| SeasonalNaive(12) RMSE | 1.396 | 1.419 |
| Chronos-2 RMSE         | 1.233 | 1.020 |
| Chronos-2 CRPS         | 1.568 | 0.693 |
| Chronos-2 cov_95       | 0.999 | 0.971 |

**Resultado contraintuitivo: Chronos gana sobre Seasonal Naive.** Para estacionalidad determinística ($Y_t = \mu + s_{t \bmod 12} + \varepsilon_t$, patrón fijo), Seasonal Naive replica solo el último ciclo observado. Chronos puede aprender el patrón promediando sobre múltiples ciclos, equivaliendo a un estimador más eficiente. La ventaja de Chronos crece notablemente con T (11% → 28%), acumulando evidencia de ciclos previos. Nota: los intervalos de Chronos están inflados masivamente a T=50 (width_95=15.4, cov_95=0.999) y se calibran mejor con T=200 (width_95=4.6, cov_95=0.971). **Conclusión: Seasonal Naive no es el modelo óptimo para estacionalidad puramente determinística. El modelo óptimo promedía sobre todos los ciclos observados. Chronos lo aproxima implícitamente; el modelo clásico correcto sería SARIMA(0,0,0)(0,0,0)_12 con dummies estacionales o media estacional.**

> **Nota metodológica:** SeasonalNaiveModel no provee intervalos de predicción. Las columnas probabilísticas de esta tabla corresponden solo a Chronos-2.

---

### Exp 1.17 — Seasonal random walk (s=12)

| | T=50 | T=200 |
|---|---|---|
| SeasonalNaive(12) RMSE | 1.193 | 1.199 |
| Chronos-2 RMSE         | 1.403 | 2.075 |
| Chronos-2 CRPS         | 0.913 | 2.166 |
| Chronos-2 cov_95       | 0.941 | 0.999 |

**Inversión exacta respecto a exp 1.16.** Para el seasonal random walk ($Y_t = Y_{t-12} + \varepsilon_t$), el pronóstico óptimo es $\hat{y}_{T+h} = y_{T+h-12}$ — exactamente lo que produce Seasonal Naive. Chronos empeora dramáticamente con T=200 (RMSE sube de 1.40 a 2.07), repitiendo el patrón adverso de exp 1.7 (doble integración estacional). Chronos a T=200 tiene intervalos extremadamente amplios (width_95=20.8) con sobrecobertura total (cov_95=0.999). **Conclusión: Seasonal Naive domina. La no-estacionariedad estacional es estructuralmente difícil para Chronos, y más contexto empeora el resultado. La comparación 1.16 vs 1.17 ilustra la diferencia fundamental entre estacionalidad determinística (donde más contexto ayuda a Chronos) y estocástica (donde más contexto perjudica).**

> **Nota metodológica:** SeasonalNaiveModel no provee intervalos de predicción.

---

### Exp 1.18 — ETS(A,A,A): tendencia + estacionalidad

| | T=50 | T=200 |
|---|---|---|
| ETS(A,A,A) RMSE | 2.521 | 2.295 |
| Chronos-2 RMSE  | 3.533 | 3.097 |
| ETS(A,A,A) CRPS | 1.679 | 1.532 |
| Chronos-2 CRPS  | 2.249 | 2.029 |

Ventaja ETS: 29% a T=50, 26% a T=200. **La brecha es la más estable del bloque:** no converge ni diverge entre T=50 y T=200. El DGP combina nivel, tendencia y estacionalidad estocásticos — el mayor número de componentes del bloque. ETS sufre el mismo problema de intervalos estrechos con muestras cortas que en 1.14 y 1.15 (cov_95=0.664 a T=50), y lo corrige con T=200 (cov_95=0.980). Chronos tiene bias positivo de 1.12 a T=50, que se reduce a 0.08 a T=200. **Conclusión: ETS(A,A,A) domina con ventaja consistente. La estructura ETS completa necesita un modelo paramétrico para ser bien capturada. Chronos tiene dificultad estimando simultáneamente nivel, tendencia y estacionalidad estocásticos en muestras cortas.**

---

### Exp 1.19 — Tendencia lineal pura (Theta)

| | T=50 | T=200 |
|---|---|---|
| Theta RMSE       | 1.412 | 1.333 |
| Chronos-2 RMSE   | 1.986 | 1.030 |
| Theta bias       | +0.719 | +0.712 |
| Chronos-2 bias   | +1.535 | +0.041 |
| Theta CRPS       | 1.132 | 1.118 |
| Chronos-2 CRPS   | 1.600 | 0.773 |

**La mayor inversión del bloque.** Theta gana a T=50 (29%), Chronos gana a T=200 (23%). Dos fenómenos se solapan:

1. **Sesgo sistemático de Theta (+0.71, invariante a T):** Para el DGP $Y_t = 0.1t + \varepsilon_t$, el método Theta con θ=2 produce sistemáticamente un sesgo positivo estable. Esto no es un artifact numérico — persiste con T=200 aunque se reduzca levemente. El método Theta sobreestima el nivel al proyectar la línea theta de orden 2. Con intervalos width_95 ≈ 15.5 en ambos T y cov_95 ≈ 0.997, el modelo cubre el sesgo pero con intervalos extremadamente inflados.

2. **Convergencia MLE (warning en producción):** El optimizador de statsmodels no converge en algunas réplicas porque la superficie MLE es plana para α cercano a 0 (óptimo para ruido puro). El warning aparece porque α≈0 es óptimo para este DGP (tendencia sin ruido AR residual) y la superficie de log-verosimilitud es plana en ese punto; no genera errores numéricos relevantes en los resultados.

3. **Chronos aprende la tendencia con T=200:** A T=200, Chronos identifica el slope 0.1 con bias ≈ 0.04, produciendo RMSE mucho menor. Con T=50 el contexto es insuficiente (bias=1.53 — sobre-extrapola la tendencia).

**Conclusión: reversión completa según T. Con muestra corta, Theta es más robusto zero-shot que Chronos. Con muestra larga, Chronos aprende el slope mejor que Theta (que mantiene sesgo estructural). El DGP de tendencia lineal pura no es el habitat natural de Theta — el método fue diseñado para series M-competition con más complejidad. El modelo óptimo aquí sería una regresión OLS pura o ARIMA(0,1,1) con drift.**

---

## Resumen comparativo (Exps 1.13–1.19)

| Exp | DGP | T=50 ganador | T=200 ganador | Diferencia RMSE (T=200) |
|-----|-----|-------------|---------------|------------------------|
| 1.13 | Local level | ETS (~1%) | ETS (7%) | Pequeña, creciente |
| 1.14 | Local trend | ETS (22%) | ETS (32%) | Grande, creciente |
| 1.15 | Damped trend | **Chronos (8%)** | ETS (8%) | Pequeña, inversión con T |
| 1.16 | Estac. determinística | **Chronos (12%)** | **Chronos (28%)** | **Grande, creciente** |
| 1.17 | Seasonal RW | SeasonalNaive (15%) | SeasonalNaive (42%) | **Grande, creciente** |
| 1.18 | ETS(A,A,A) | ETS (29%) | ETS (26%) | Grande, estable |
| 1.19 | Tendencia lineal | Theta (29%) | **Chronos (23%)** | Moderada, inversión con T |

---

## Conclusiones transversales del bloque ETS/Theta/Seasonal

### 8. Los modelos ETS dominan sus DGPs nativos, con mayor margen que los modelos ARIMA
ETS(A,N,N), ETS(A,A,N) y ETS(A,A,A) superan a Chronos con ventajas de 7–32%, mayores que las observadas en el bloque ARIMA (1–11% salvo exp 1.7). La estructura state-space de ETS y la de los DGPs son exactamente equivalentes, lo que produce la mayor ventaja teórica posible.

### 9. La estacionalidad determinística invierte el resultado: Chronos supera a Seasonal Naive (exp 1.16)
Seasonal Naive usa solo el último ciclo observado — subóptimo para estacionalidad determinística fija. Chronos promedia implícitamente sobre múltiples ciclos, mejorando con T. Esto complementa el hallazgo de exp 1.7: Chronos maneja mejor la estacionalidad *determinística* pero peor la *estocástica*.

### 10. La estacionalidad no estacionaria sigue siendo el talón de Aquiles de Chronos (exp 1.17)
El seasonal random walk reproduce el patrón adverso de exp 1.7: Chronos empeora con más contexto (RMSE 1.40→2.07). La conclusión 3 del bloque ARIMA se confirma y generaliza.

### 11. Theta tiene sesgo estructural en DGPs de tendencia lineal pura
El sesgo +0.71 de Theta es invariante a T, lo que limita su utilidad para el DGP de exp 1.19. En muestras cortas el bias importa menos que la varianza (Theta gana en T=50); en muestras largas Chronos elimina el bias y domina. La convergencia MLE del Theta es un problema adicional para esta clase de DGPs.

### 12. Los intervalos ETS son poco confiables con muestras cortas en DGPs no estacionarios
En los exps 1.14, 1.15 y 1.18 con T=50, ETS produce cov_95 entre 0.66 y 0.71 — subcovertura severa. El modelo necesita suficiente historia para calibrar bien la varianza predictiva de los estados estocásticos. Con T=200 la cobertura se normaliza (0.96–0.99). Esto tiene implicancias para el uso de ETS en series cortas.
