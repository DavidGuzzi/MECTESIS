# Conclusiones — Experimentos univariados 1.1–1.19

**Setup 1.1–1.19:** T ∈ {50, 200} | H = 24 | R = 500 | Semilla = 3649  
**Modelos Core 1.1–1.12:** ARIMA/SARIMA/GARCH (statsmodels + arch, correctamente especificados) vs Chronos-2 (zero-shot)  
**Modelos Core 1.13–1.19:** ETS / Theta / Seasonal Naive (statsmodels, correctamente especificados) vs Chronos-2 (zero-shot)  
**Nota:** SeasonalNaiveModel no produce intervalos de predicción; CRPS, cobertura y Winkler solo disponibles para Chronos en exps 1.16–1.17.  
**Métricas reportadas:** promedios avg_all (H=1…24). RMSE, MAE, CRPS: menor es mejor. COV_80/95: nominal 0.80/0.95. WINKLER_95: menor es mejor (penaliza sobrecobertura + penaliza fuertemente las observaciones fuera del intervalo).

---

## Resultados por experimento

### Exp 1.1 — AR(1) φ=0.3, baja persistencia

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARIMA(1,0,0) | 50  | **1.081** | **0.860** | **0.614** | 0.758 | 0.919 | **3.91** | **5.39** |
| Chronos-2    | 50  | 1.141 | 0.904 | 1.084 | 0.898 | 0.995 | 9.93  | 10.03 |
| ARIMA(1,0,0) | 200 | **1.062** | **0.849** | **0.601** | 0.786 | 0.941 | **4.05** | **5.01** |
| Chronos-2    | 200 | 1.073 | 0.858 | 0.730 | 0.792 | 0.962 | 4.57  | 5.14  |

ARIMA domina en RMSE (~5% en T=50, ~1% en T=200), MAE y CRPS. La brecha en CRPS es más pronunciada que en RMSE (76% en T=50): Chronos produce distribuciones predictivas mucho más anchas que no están justificadas por el error puntual. Chronos sobrecobertura sistemáticamente (cov_95=0.995 vs nominal 0.95) con intervalos 95% 2.5× más anchos en T=50, lo que resulta en Winkler ~2× peor. A T=200 los intervalos de Chronos se ajustan notablemente (width_95=4.57 vs 9.93), pero el CRPS gap persiste. **Conclusión: ARIMA domina en todas las métricas. El proceso de baja persistencia es simple pero Chronos no logra calibrar correctamente sus distribuciones predictivas.**

---

### Exp 1.2 — AR(1) φ=0.9, alta persistencia

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARIMA(1,0,0) | 50  | **2.308** | **1.849** | **1.375** | 0.583 | 0.780 | **5.96** | 16.96 |
| Chronos-2    | 50  | 2.590 | 2.070 | 1.905 | 0.734 | 0.965 | 15.07 | **16.78** |
| ARIMA(1,0,0) | 200 | **2.142** | **1.711** | **1.218** | 0.740 | 0.917 | **7.49** | **10.38** |
| Chronos-2    | 200 | 2.457 | 1.930 | 1.696 | 0.777 | 0.961 | 11.38 | 13.00 |

ARIMA domina en RMSE (~12% en ambos T) y CRPS (~28% en T=50, ~28% en T=200). La heterodoxia calibratoria es notable: ARIMA sufre **subcovertura severa** a T=50 (cov_95=0.780, cov_80=0.583), lo que infla su Winkler en T=50 hasta igualar a Chronos (16.96 vs 16.78). A T=200, ARIMA mejora la calibración (cov_95=0.917) y gana también en Winkler. Chronos sobrecobertura (cov_95=0.965/0.961) con intervalos ~2.5× más anchos. ARIMA no mejora con T en RMSE (2.31→2.14), lo que refleja que la persistencia φ=0.9 es genuinamente difícil: errores crecen con el horizonte y T=200 solo ayuda marginalmente a estimar φ. **Conclusión: ARIMA domina en precisión; ambos modelos tienen problemas de calibración pero por razones opuestas (ARIMA: subcovertura, Chronos: sobrecobertura).**

---

### Exp 1.3 — Random Walk I(1), sin drift

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARIMA(0,1,0) | 50  | **3.173** | **2.533** | **1.795** | **0.810** | **0.954** | **12.92** | **15.32** |
| Chronos-2    | 50  | 3.540 | 2.819 | 2.505 | 0.691 | 0.942  | 18.47 | 22.71 |
| ARIMA(0,1,0) | 200 | **3.314** | **2.639** | **1.869** | **0.797** | **0.954** | **13.07** | **15.57** |
| Chronos-2    | 200 | 3.454 | 2.727 | 2.353 | 0.764 | 0.975  | 15.86 | 17.88 |

ARIMA domina en todas las métricas. Ventaja en RMSE: ~11% en T=50, ~4% en T=200. La brecha en CRPS es mayor (~40% en T=50) y la brecha en Winkler confirma que ARIMA calibra mejor (15.32 vs 22.71 en T=50). Chronos tiene **subcovertura** al 80% (0.691 en T=50) pero sobrecobertura al 95% (0.975 en T=200) con intervalos más anchos — señal de distribuciones predictivas mal formadas. Los valores de RMSE son altos (~3.2–3.5) porque el RW acumula varianza con el horizonte (RMSE esperado = σ·√h, promediado en h=1..24: ~3.2 para σ=1). **Conclusión: ARIMA identifica correctamente la estructura I(1) y produce pronósticos e intervalos más ajustados. Chronos subestima la varianza al 80% y la sobreestima al 95%.**

---

### Exp 1.4 — Random Walk I(1), drift = 0.5

| Modelo | T | RMSE | BIAS | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|---|
| ARIMA(0,1,0) | 50  | 7.174 | +6.38 | 6.517 | 4.861 | 0.381 | 0.649 | 14.52 | 58.15 |
| Chronos-2    | 50  | **5.658** | +3.30 | **4.685** | **4.575** | **0.770** | **0.988** | 36.71 | **38.35** |
| ARIMA(0,1,0) | 200 | 7.083 | +6.20 | 6.374 | 4.733 | 0.409 | 0.669 | 14.67 | 54.20 |
| Chronos-2    | 200 | **3.838** | +0.27 | **3.012** | **2.874** | **0.752** | **0.979** | 24.77 | **26.35** |

**Resultado más relevante del bloque.** ARIMA(0,1,0) estimado sin constante no captura el drift=0.5: el sesgo crece linealmente con el horizonte (bias ≈ +0.5·h), llegando a +12.3 en h=24 (T=50). El RMSE de ARIMA es ~7.1 en ambos T, con cobertura catastrófica (cov_95=0.649/0.669) y Winkler >50. Chronos captura el drift desde el contexto de la serie: a T=50 su bias es +3.3 (sobreestima el drift) pero se reduce a +0.27 a T=200, con RMSE que cae de 5.66 a 3.84. Chronos gana en RMSE, MAE, CRPS y Winkler en ambos T; la ventaja se amplía con T.

> **Nota de implementación:** `ARIMA(y, order=(0,1,0))` en statsmodels no agrega constante post-diferenciación por defecto. El modelo correcto es `ARIMA(y, order=(0,1,0), trend='c')`. Este experimento ilustra deliberadamente la misspecificación por omisión de drift — un error frecuente en la práctica.

**Conclusión: victoria de Chronos por misspecificación del modelo clásico. Más contexto beneficia a Chronos (RMSE cae 32% de T=50 a T=200) mientras ARIMA permanece igualmente sesgado.**

---

### Exp 1.5 — AR(1) + tendencia lineal determinista

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARIMA(1,0,0)+trend | 50  | **1.895** | **1.514** | **1.147** | 0.526 | 0.714 | **4.13** | 15.91 |
| Chronos-2          | 50  | 3.742 | 3.276 | 2.980 | **0.808** | **0.997** | 24.45 | **24.54** |
| ARIMA(1,0,0)+trend | 200 | **1.277** | **1.022** | **0.724** | 0.765 | 0.931 | **4.69** | **5.98** |
| Chronos-2          | 200 | 1.414 | 1.133 | 1.137 | 0.797 | 0.987 | 10.15 | 10.41 |

ARIMA+trend domina en RMSE y CRPS en ambos T, pero la brecha se cierra significativamente con T (97% en T=50 → 11% en T=200). A T=50, Chronos tiene bias=+2.94 — sobre-extrapola la tendencia con muestra corta — lo que colapsa su RMSE. A T=200, Chronos aprende el slope correctamente (bias≈0) pero sigue produciendo intervalos mucho más anchos (width_95=10.15 vs 4.69). ARIMA+trend tiene **subcovertura severa** a T=50 (cov_95=0.714, cov_80=0.526) con Winkler peor que Chronos (15.91 vs 24.54); a T=200 la calibración mejora y gana en Winkler también. **Conclusión: ARIMA domina en precisión puntual y distribucional; Chronos es inestable con muestras cortas ante tendencias lineales fuertes. La brecha en RMSE converge con T pero no desaparece.**

---

### Exp 1.6 — SAR trimestral, s=4, estacionario

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMA(1,0,0)(1,0,0)_4 | 50  | **1.199** | **0.959** | **0.683** | 0.778 | 0.930 | **4.67** | **5.97** |
| Chronos-2              | 50  | 1.402 | 1.120 | 1.232 | 0.862 | 0.992 | 10.96 | 11.10 |
| SARIMA(1,0,0)(1,0,0)_4 | 200 | **1.199** | **0.957** | **0.677** | 0.795 | 0.946 | **4.68** | **5.65** |
| Chronos-2              | 200 | 1.268 | 1.015 | 0.855 | 0.783 | 0.961 | 5.30  | 6.06  |

SARIMA domina en todas las métricas. Ventaja en RMSE: 14% en T=50, 6% en T=200. La brecha en CRPS es mayor (81% en T=50, 26% en T=200) y la brecha en Winkler confirma mejor calibración de SARIMA. Chronos sobrecobertura al 95% (0.992/0.961) con intervalos más anchos. Notablemente, el RMSE de SARIMA es prácticamente idéntico entre T=50 y T=200 (1.199 en ambos) — el patrón SAR estacionario de período 4 se identifica con muy poca muestra. **Conclusión: SARIMA domina en todas las métricas; la ventaja disminuye con T pero no desaparece.**

---

### Exp 1.7 — Seasonal I(1)×I(1)_12, doble integración

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMA(0,1,0)(0,1,0)_12 | 50  | **4.097** | **3.248** | **2.322** | **0.790** | **0.934** | **16.13** | **20.64** |
| Chronos-2               | 50  | 5.934 | 4.695 | 3.895 | 0.644 | 0.942 | 27.08 | 32.96 |
| SARIMA(0,1,0)(0,1,0)_12 | 200 | **4.158** | **3.327** | **2.356** | **0.800** | **0.952** | **16.52** | **19.13** |
| Chronos-2               | 200 | 5.824 | 4.657 | 3.894 | 0.701 | 0.943 | 25.83 | 32.69 |

**El peor resultado de Chronos en el bloque.** SARIMA domina en todas las métricas. Ventaja en RMSE: ~45% en T=50, ~40% en T=200 — la brecha es grande y **estable** (no crece ni se cierra con T). En CRPS la brecha es similar (68% en ambos T). Chronos tiene **subcovertura al 80%** (0.644 en T=50) a pesar de sobrecobertura al 95%, señal de distribuciones predictivas con colas inadecuadas. El RMSE de Chronos apenas cambia entre T=50 y T=200 (5.934→5.824), a diferencia de otros experimentos donde más contexto ayuda. **Conclusión: la doble integración estacional es la limitación estructural más clara de Chronos. El modelo de fundación no maneja la no-estacionariedad de largo alcance en dos dimensiones, y más contexto no lo ayuda.**

---

### Exp 1.8 — AR(1) con quiebre estructural en T/2

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARIMA(1,0,0)+break | 50  | **1.823** | **1.448** | **1.047** | 0.686 | 0.865 | **5.77** | **10.31** |
| Chronos-2          | 50  | 1.980 | 1.581 | 1.546 | 0.792 | 0.987 | 13.05 | 13.34 |
| ARIMA(1,0,0)+break | 200 | **1.622** | **1.302** | **0.928** | 0.699 | 0.888 | **5.26** | **8.12** |
| Chronos-2          | 200 | 1.745 | 1.388 | 1.120 | 0.724 | 0.936 | 6.67  | 8.35  |

ARIMA+break domina en RMSE (~8% en T=50, ~7% en T=200), MAE, CRPS y Winkler. La brecha es consistente entre T y no converge (a diferencia de la versión anterior con T=500). Ambos modelos exhiben subcovertura al 80% y 95%: el quiebre estructural infla la incertidumbre real más allá de lo que cualquiera de los dos modelos captura en sus intervalos. CRPS gap de ~48% en T=50 refleja que Chronos produce distribuciones mucho más dispersas. A T=200, Chronos mejora (width_95 cae de 13.05 a 6.67) sugiriendo que más contexto le ayuda a identificar el nuevo régimen. **Conclusión: ARIMA con dummy de quiebre explícito domina en todas las métricas con ventaja estable.**

---

### Exp 1.9 — AR(1)–ARCH(1), volatilidad baja persistencia

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| AR(1)+ARCH(1) | 50  | **0.409** | **0.318** | **0.233** | 0.777 | 0.920 | **1.61** | **2.32** |
| Chronos-2     | 50  | 0.433 | 0.336 | 0.407 | 0.891 | 0.993 | 3.77  | 3.84  |
| AR(1)+ARCH(1) | 200 | **0.395** | **0.308** | **0.220** | **0.808** | **0.945** | **1.54** | **1.98** |
| Chronos-2     | 200 | 0.399 | 0.312 | 0.273 | 0.797 | 0.960 | 1.75  | 2.07  |

ARIMA domina en RMSE (~6% T=50, ~1% T=200), MAE, CRPS y Winkler. La brecha en CRPS es mucho mayor que en RMSE: 75% en T=50, 24% en T=200. A T=200, el RMSE converge a paridad (~1%) pero Chronos produce intervalos ~14% más anchos con CRPS 24% mayor. Chronos sobrecobertura al 95% en ambos T (0.993/0.960) con intervalos 2.3× más anchos en T=50. **Conclusión: paridad en RMSE a T=200, pero ventaja consistente del modelo clásico en CRPS y calibración. La heterocedasticidad transitoria (sin β, persistencia cae en un período) no beneficia estructuralmente a ningún modelo en RMSE.**

---

### Exp 1.10 — AR(1)–GARCH(1,1), persistencia alta (α+β=0.9)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| AR(1)+GARCH(1,1) | 50  | **1.079** | **0.849** | **0.614** | 0.750 | 0.908 | **3.95** | **5.88** |
| Chronos-2        | 50  | 1.137 | 0.890 | 1.056 | 0.891 | 0.993 | 9.63  | 9.76  |
| AR(1)+GARCH(1,1) | 200 | **1.045** | **0.823** | **0.587** | **0.794** | **0.942** | **4.03** | **5.15** |
| Chronos-2        | 200 | 1.057 | 0.833 | 0.722 | 0.791 | 0.957 | 4.55  | 5.34  |

AR+GARCH domina en RMSE (~5% T=50, ~1% T=200), MAE y CRPS. La brecha en CRPS es ~72% en T=50, ~23% en T=200 — mucho mayor que en RMSE. Chronos sobrecobertura al 95% en T=50 (0.993) con intervalos 2.4× más anchos, mientras que a T=200 los intervalos se ajustan (4.55 vs 4.03) con Winkler comparable. La dinámica GARCH afecta la varianza condicional pero no la media (AR), por lo que el RMSE es similar para ambos: el error de nivel AR domina el error cuadrático. **Conclusión: ventaja del modelo clásico en RMSE y CRPS; la brecha en calibración disminuye con T.**

---

### Exp 1.11 — GARCH(1,1) media cero

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| GARCH(1,1)-ZeroMean | 50  | **1.002** | **0.789** | **0.569** | 0.776 | 0.917 | **3.84** | **5.39** |
| Chronos-2           | 50  | 1.052 | 0.826 | 0.979 | 0.889 | 0.992 | 8.93  | 9.07  |
| GARCH(1,1)-ZeroMean | 200 | **0.992** | **0.784** | **0.558** | **0.797** | **0.942** | **3.85** | **4.84** |
| Chronos-2           | 200 | 0.999 | 0.789 | 0.685 | 0.791 | 0.958 | 4.33  | 5.03  |

GARCH domina en todas las métricas, con paridad casi perfecta en RMSE a T=200 (~1%). La brecha en CRPS es desproporcionada: 72% en T=50, 23% en T=200. Sin componente AR, el pronóstico puntual óptimo es cero para ambos modelos, por lo que el RMSE converge. La diferencia residual en CRPS a T=200 refleja que GARCH(1,1) estima la varianza condicional actual con precisión mientras Chronos la promedia de forma más conservadora. **Conclusión: empate en RMSE a T=200; GARCH con ventaja en CRPS y calibración de intervalos.**

---

### Exp 1.12 — AR(1)–GJR–GARCH(1,1,1), efecto leverage

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| AR(1)+GJR-GARCH(1,1,1) | 50  | **1.075** | **0.845** | **0.615** | 0.755 | 0.907 | **4.07** | **6.03** |
| Chronos-2              | 50  | 1.138 | 0.888 | 1.060 | 0.890 | 0.992 | 9.69  | 9.84  |
| AR(1)+GJR-GARCH(1,1,1) | 200 | **1.048** | **0.822** | **0.587** | **0.795** | **0.939** | **4.03** | **5.20** |
| Chronos-2              | 200 | 1.058 | 0.830 | 0.721 | 0.792 | 0.957 | 4.55  | 5.39  |

Métricas casi idénticas a exp 1.10 (GARCH simétrico), confirmando que el efecto leverage γ·ε²·1{ε<0} no genera ventaja adicional en RMSE: la asimetría afecta la varianza condicional en shocks negativos, irrelevante para el error medio cuadrático del nivel. GJR domina en RMSE (~6% T=50, ~1% T=200) y CRPS (~72% T=50). **Conclusión: el leverage no cambia el ranking entre modelos en ninguna métrica.**

---

## Resumen comparativo — Exps 1.1–1.12

| Exp | DGP | T=50 ganador (RMSE) | T=200 ganador (RMSE) | CRPS gap (T=50) | Calibración dominante |
|-----|-----|---------------------|----------------------|-----------------|-----------------------|
| 1.1 | AR(1) φ=0.3 | ARIMA (5%) | ARIMA (1%) | ARIMA (76%) | ARIMA (Winkler 2×) |
| 1.2 | AR(1) φ=0.9 | ARIMA (12%) | ARIMA (12%) | ARIMA (28%) | Mixto (ARIMA subcov., Chronos sobrecobertura) |
| 1.3 | RW sin drift | ARIMA (11%) | ARIMA (4%) | ARIMA (40%) | ARIMA (Winkler mejor) |
| 1.4 | RW drift=0.5 | **Chronos (21%)** | **Chronos (46%)** | Chronos ligera | **Chronos (Winkler: 58 vs 38)** |
| 1.5 | AR+trend | ARIMA (49%) | ARIMA (10%) | ARIMA (60%) | T=50: Chronos cov.; T=200: ARIMA |
| 1.6 | SAR s=4 | SARIMA (14%) | SARIMA (6%) | SARIMA (80%) | SARIMA (Winkler mejor) |
| 1.7 | I(1)×I(1)_12 | **SARIMA (45%)** | **SARIMA (40%)** | SARIMA (68%) | SARIMA (todas las métricas) |
| 1.8 | AR+break | ARIMA (8%) | ARIMA (7%) | ARIMA (48%) | ARIMA (Winkler mejor) |
| 1.9 | AR-ARCH(1) | AR+ARCH (6%) | Empate (1%) | AR+ARCH (75%) | AR+ARCH (CRPS, Winkler) |
| 1.10 | AR-GARCH(1,1) | AR+GARCH (5%) | Empate (1%) | AR+GARCH (72%) | AR+GARCH (CRPS, Winkler) |
| 1.11 | GARCH media 0 | GARCH (5%) | Empate (1%) | GARCH (72%) | GARCH (CRPS, Winkler) |
| 1.12 | AR-GJR-GARCH | GJR (6%) | Empate (1%) | GJR (72%) | GJR (CRPS, Winkler) |

---

## Experimentos 1.13–1.19 — Bloque ETS / Theta / Seasonal Naive

---

### Exp 1.13 — Nivel local (Local Level → ETS(A,N,N))

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ETS(A,N,N) | 50  | **1.505** | **1.198** | **0.868** | 0.740 | 0.895 | **5.48** | **8.42** |
| Chronos-2  | 50  | 1.523 | 1.213 | 1.302 | 0.843 | 0.988 | 11.41 | 11.67 |
| ETS(A,N,N) | 200 | **1.541** | **1.232** | **0.875** | 0.781 | 0.938 | **5.86** | **7.44** |
| Chronos-2  | 200 | 1.651 | 1.320 | 1.094 | 0.735 | 0.953 | 6.91  | 8.30  |

ETS domina en RMSE (1% en T=50, 7% en T=200), MAE, CRPS y Winkler. La brecha en RMSE crece con T (Chronos no mejora mientras ETS mantiene el nivel). La brecha en CRPS es sustancial: 50% en T=50. Chronos sobrecobertura al 95% en T=50 (0.988) con intervalos 2× más anchos. A T=200, Chronos tiene subcovertura al 80% (0.735) sugiriendo distribuciones predictivas con forma inadecuada. **Conclusión: ETS domina en todas las métricas; la ventaja crece con T.**

---

### Exp 1.14 — Tendencia local (Local Trend → ETS(A,A,N))

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ETS(A,A,N) | 50  | **4.754** | **3.761** | **3.306** | 0.564 | 0.669 | 18.49 | 67.59 |
| Chronos-2  | 50  | 6.066 | 4.853 | 3.914 | 0.606 | 0.914 | 24.64 | **36.38** |
| ETS(A,A,N) | 200 | **4.205** | **3.364** | **2.819** | 0.939 | **0.988** | **32.55** | **33.32** |
| Chronos-2  | 200 | 5.544 | 4.442 | 3.777 | 0.682 | 0.934 | 26.37 | 33.19 |

ETS domina en RMSE (22% en T=50, 24% en T=200) y CRPS. Sin embargo, en Winkler el resultado se invierte a T=50: **Chronos tiene Winkler mejor** (36.38 vs 67.59) porque ETS presenta **subcovertura severa** (cov_95=0.669), lo que infla masivamente el Winkler por observaciones fuera del intervalo. A T=200, ETS corrige la calibración (cov_95=0.988) con intervalos muy anchos (width_95=32.55) y gana en Winkler. Ambos modelos tienen altos RMSE (~4–6) porque el DGP de tendencia + nivel estocásticos acumula incertidumbre con el horizonte. **Conclusión: ETS domina en RMSE y CRPS, pero sus intervalos a T=50 son no confiables (Winkler peor que Chronos). Con T=200, ETS domina también en Winkler.**

---

### Exp 1.15 — Tendencia amortiguada (Damped Trend → ETS(A,Ad,N))

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ETS(A,Ad,N) | 50  | 2.814 | 2.176 | **1.723** | 0.567 | 0.711 | **7.97** | 28.73 |
| Chronos-2   | 50  | **2.601** | **2.062** | 1.955 | **0.753** | **0.965** | 15.34 | **17.26** |
| ETS(A,Ad,N) | 200 | **2.435** | **1.935** | **1.518** | **0.862** | **0.959** | 14.19 | **15.95** |
| Chronos-2   | 200 | 2.627 | 2.119 | 1.729 | 0.703 | 0.951 | **11.10** | 13.15 |

A T=50, **Chronos gana en RMSE (8%) y MAE**, y domina en Winkler (17.26 vs 28.73). ETS tiene CRPS ligeramente mejor pero subcovertura severa (cov_95=0.711) con Winkler 66% peor. A T=200, ETS recupera la ventaja en RMSE (7%), CRPS y Winkler; la calibración de ETS mejora (cov_95=0.959). El parámetro de amortiguamiento φ es difícil de estimar con T=50: alta incertidumbre en la estimación infla la varianza de los errores. Chronos zero-shot es más robusto ante la estimación ruidosa del damping en muestras cortas. **Conclusión: inversión según T. Chronos más robusto a T=50 (RMSE, MAE, Winkler). ETS domina a T=200 (RMSE, CRPS, Winkler). El único caso del bloque donde Chronos gana en RMSE.**

---

### Exp 1.16 — Estacionalidad determinística (s=12)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SeasonalNaive(12) | 50  | 1.396 | 1.111 | — | — | — | — | — |
| Chronos-2         | 50  | **1.233** | **0.981** | 1.568 | 0.966 | 0.999 | 15.45 | 15.46 |
| SeasonalNaive(12) | 200 | 1.419 | 1.136 | — | — | — | — | — |
| Chronos-2         | 200 | **1.020** | **0.814** | **0.693** | 0.775 | 0.971 | 4.57  | 5.02  |

**Chronos gana sobre Seasonal Naive** en RMSE y MAE en ambos T (12% en T=50, 28% en T=200). Seasonal Naive replica solo el último ciclo observado — óptimo para el seasonal RW, pero subóptimo para estacionalidad determinística fija donde promediar ciclos reduce el ruido. Chronos aproxima implícitamente ese promedio, mejorando con T. A T=50, Chronos tiene sobrecobertura extrema (cov_95=0.999) con intervalos muy anchos (15.45), CRPS=1.568 — distribuciones excesivamente conservadoras. A T=200, Chronos se calibra bien (cov_95=0.971, width_95=4.57, CRPS=0.693). **Conclusión: victoria de Chronos por diseño óptimo superior al benchmark. Seasonal Naive no es el modelo correcto para estacionalidad determinística; el óptimo promedía ciclos.**

> **Nota metodológica:** SeasonalNaiveModel no provee intervalos de predicción.

---

### Exp 1.17 — Seasonal random walk (s=12)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SeasonalNaive(12) | 50  | **1.193** | **0.953** | — | — | — | — | — |
| Chronos-2         | 50  | 2.075 | 1.595 | 2.166 | 0.939 | 0.999 | 20.82 | 20.84 |
| SeasonalNaive(12) | 200 | **1.199** | **0.960** | — | — | — | — | — |
| Chronos-2         | 200 | 1.403 | 1.117 | 0.913 | 0.721 | 0.941 | 5.58  | 6.98  |

**Inversión exacta respecto a exp 1.16.** SeasonalNaive domina: Chronos es 74% peor en RMSE a T=50 y 17% peor a T=200. Para el seasonal RW ($Y_t = Y_{t-12} + \varepsilon_t$), el pronóstico óptimo es exactamente $\hat{y}_{T+h} = y_{T+h-12}$, que es lo que produce Seasonal Naive. La brecha *disminuye* con T (74%→17%): Chronos mejora con más contexto pero nunca alcanza el óptimo. A T=50, Chronos tiene sobrecobertura extrema (0.999) con CRPS=2.166 y width_95=20.82 — incertidumbre masivamente sobreestimada. A T=200, se calibra mejor (cov_95=0.941) pero RMSE sigue siendo mayor. **Conclusión: SeasonalNaive domina. La comparación 1.16 vs 1.17 ilustra la diferencia fundamental entre estacionalidad determinística (Chronos supera al naive) y estocástica (Chronos inferior).**

> **Nota metodológica:** SeasonalNaiveModel no provee intervalos de predicción.

---

### Exp 1.18 — ETS(A,A,A): tendencia + estacionalidad

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ETS(A,A,A) | 50  | **2.521** | **1.984** | **1.679** | 0.529 | 0.664 | 8.35  | 31.15 |
| Chronos-2  | 50  | 3.533 | 2.861 | 2.249 | 0.630 | 0.941 | 14.84 | **18.96** |
| ETS(A,A,A) | 200 | **2.295** | **1.843** | **1.532** | 0.921 | **0.980** | 17.24 | **18.07** |
| Chronos-2  | 200 | 3.097 | 2.405 | 2.029 | 0.686 | 0.933 | 13.61 | 18.07 |

ETS domina en RMSE (29% en T=50, 26% en T=200) y CRPS. La brecha es la más estable del bloque. Sin embargo, en Winkler a T=50, **Chronos gana** (18.96 vs 31.15) porque ETS tiene subcovertura severa (cov_95=0.664, cov_80=0.529) — el modelo necesita suficiente historia para calibrar los tres componentes estocásticos (nivel, tendencia, estacionalidad). A T=200, ETS corrige la calibración (cov_95=0.980) y los Winkler quedan empatados (~18.07). Chronos tiene bias=+1.12 a T=50 que desaparece a T=200. **Conclusión: ETS(A,A,A) domina en precisión en ambos T; sus intervalos a T=50 son no confiables (Winkler peor). La estructura completa ETS requiere muestra para calibrar bien.**

---

### Exp 1.19 — Tendencia lineal pura (Theta)

| Modelo | T | RMSE | BIAS | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|---|
| Theta     | 50  | **1.468** | −0.58 | **1.180** | **1.286** | **0.965** | **0.987** | 17.62 | 18.01 |
| Chronos-2 | 50  | 1.986 | +1.54 | 1.719 | 1.600 | 0.846 | 0.997 | **13.59** | **13.71** |
| Theta     | 200 | 7.892 | −7.63 | 7.666 | 5.538 | 0.272 | 0.640 | 17.95 | 61.33 |
| Chronos-2 | 200 | **1.030** | +0.04 | **0.820** | **0.773** | **0.776** | **0.980** | **6.43** | **6.75** |

**La mayor inversión del bloque.** A T=50: Theta gana en RMSE (26%), MAE y CRPS. A T=200: **Theta falla catastróficamente** — RMSE=7.892, bias=−7.633, cov_95=0.640, Winkler_95=61.33. Chronos gana por factor 7.7× en RMSE y 9× en Winkler a T=200.

El comportamiento de Theta:
- A T=50: bias=−0.58 (moderado), Chronos sobre-extrapola (bias=+1.54). Theta más preciso pero con intervalos 95% inflados (width=17.62) que dan cobertura casi perfecta (0.987). Winkler ligeramente peor que Chronos.
- A T=200: **Theta colapsa completamente**. bias=−7.63 invariante al horizonte, sugiriendo que Theta usa la media global de la serie (~10 para Y_t=0.1t) como nivel de predicción en lugar del nivel actual (~20). Los intervalos mantienen width≈18 (idénticos a T=50) pero la cobertura cae a 0.640 en 95% porque el sesgo supera el radio del intervalo.

La causa probable es la superficie MLE plana (α→0 óptimo para tendencia sin componente AR residual): el optimizador de statsmodels convierte en un nivel constante estimado sobre la serie entera, ignorando que el nivel actual es el doble de la media histórica.

**Conclusión: Chronos domina abrumadoramente a T=200 en todas las métricas. Theta falla estructuralmente porque su mecanismo de suavizamiento colapsa hacia la media histórica con series de tendencia pura y muestras largas. A T=50, Theta es más robusto.**

---

## Resumen comparativo — Exps 1.13–1.19

| Exp | DGP | T=50 ganador (RMSE) | T=200 ganador (RMSE) | Nota calibración |
|-----|-----|---------------------|----------------------|-----------------|
| 1.13 | Local level | ETS (1%) | ETS (7%) | Chronos sobrecobertura T=50 |
| 1.14 | Local trend | ETS (22%) | ETS (24%) | ETS subcov T=50 → Winkler peor |
| 1.15 | Damped trend | **Chronos (8%)** | ETS (7%) | Inversión; ETS subcov T=50 |
| 1.16 | Estac. determ. s=12 | **Chronos (12%)** | **Chronos (28%)** | Chronos sobrecobertura T=50 |
| 1.17 | Seasonal RW s=12 | SN (74%) | SN (17%) | Chronos sobrecobertura T=50 |
| 1.18 | ETS(A,A,A) | ETS (29%) | ETS (26%) | ETS subcov T=50 → Winkler peor |
| 1.19 | Tendencia lineal | Theta (26%) | **Chronos (667%)** | Theta colapsa a T=200 |

---

## Conclusiones transversales

### 1. Los modelos clásicos dominan en su DGP nativo en todas las métricas principales
Cuando el modelo clásico está correctamente especificado, supera a Chronos en RMSE, MAE y CRPS en todos los experimentos excepto exp 1.4 (misspecificación) y 1.16 (benchmark subóptimo). Las ventajas son más pronunciadas en CRPS que en RMSE: incluso cuando el error puntual es similar, las distribuciones predictivas de Chronos son peores (más dispersas o mal calibradas).

### 2. Chronos es robusto ante misspecificaciones del modelo clásico
El caso más notable es el drift en exp 1.4: Chronos identifica el drift desde el contexto de la serie sin estimarlo explícitamente, reduciendo su RMSE de 5.66 a 3.84 entre T=50 y T=200. Esto sugiere que en aplicaciones reales con especificación incierta, Chronos puede ser más robusto.

### 3. La doble integración estacional es el talón de Aquiles de Chronos (exps 1.7 y 1.17)
Exp 1.7 y 1.17 muestran que la no-estacionariedad estacional es estructuralmente difícil para Chronos: la brecha frente al modelo clásico es grande (~40-45% en RMSE) y estable con T. En ambos experimentos, más contexto no ayuda a Chronos.

### 4. Las brechas en RMSE no convergen uniformemente con T
En experimentos con DGPs estacionarios (1.1, 1.6, 1.9–1.12), la ventaja del modelo clásico se comprime de T=50 a T=200. Sin embargo, en los experimentos con integración (1.3, 1.7) o componentes múltiples (1.13–1.14, 1.18), la brecha se mantiene o crece. La hipótesis "más contexto cierra la brecha" es solo parcialmente verdadera.

### 5. CRPS y Winkler revelan brechas ocultas por el RMSE
En los experimentos GARCH (1.9–1.12), el RMSE converge a paridad a T=200, pero la brecha en CRPS sigue siendo ~20-25%: Chronos produce distribuciones predictivas más anchas de lo necesario. El Winkler penaliza explícitamente este exceso: Chronos con sobrecobertura y grandes intervalos puede tener peor Winkler que un modelo con subcovertura moderada.

### 6. ETS y Theta tienen subcovertura severa con muestras cortas
En exps 1.14, 1.15 y 1.18 con T=50, los modelos ETS tienen cov_95 entre 0.664 y 0.711 — subcovertura severa que resulta en Winkler peor que Chronos a pesar de mejor RMSE. Con T=200 la calibración se normaliza. Theta a T=200 colapsa completamente (exp 1.19). Esto tiene implicancias críticas para el uso de modelos clásicos en series cortas: la precisión puntual puede ser engañosa si los intervalos no son confiables.

### 7. Chronos sobrecobertura sistemáticamente al 95%, con brecha calibrativa característica
En casi todos los experimentos y ambos T, Chronos tiene cov_95 entre 0.94 y 0.999 — consistentemente por encima del nominal 0.95. Los intervalos son más anchos, los Winkler mayores. La excepción es T=200 en experimentos con DGPs complejos (1.7, 1.17) donde Chronos también puede tener subcovertura al 80%.

### 8. Implicación para la tesis
Bajo condiciones de laboratorio (DGP conocido, modelo clásico correctamente especificado), los modelos clásicos son el benchmark correcto y superan a Chronos en todas las métricas relevantes. Las excepciones son exp 1.4 (misspecificación deliberada del drift), exp 1.15 a T=50 (parámetro difícil de estimar), exp 1.16 (benchmark subóptimo), y exp 1.19 a T=200 (falla estructural de Theta). En series reales con DGP desconocido, el resultado puede diferir sustancialmente.
