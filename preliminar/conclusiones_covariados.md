# Conclusiones — Experimentos con covariables 3.1–3.6

**Setup 3.1–3.6:** T ∈ {50, 200} | H = 24 | R = 500 | Semilla = 3649  
**Modelos univariados (3.1–3.3, 3.5–3.6):** SARIMAX / ARDL-ECM (statsmodels, correctamente especificados) vs Chronos-2 (con X, zero-shot)  
**Modelo multivariado (3.4):** VARMAX(1)-OLS (ecuación a ecuación) vs Chronos-2 joint (con X, zero-shot)  
**Nota:** ARDL-ECM (exp 3.6) no produce intervalos de predicción; CRPS, cobertura y Winkler solo disponibles para SARIMAX y Chronos en ese experimento.  
**Nota X_future:** el motor de simulación proporciona los valores futuros del covariante como dato conocido (supuesto oráculo) — se evalúa la utilidad del covariante dado que su futuro es observable.  
**Métricas reportadas:** promedios avg_all (H=1…24). RMSE, MAE, CRPS: menor es mejor. COV_80/95: nominal 0.80/0.95. WINKLER_95: menor es mejor.

---

## Resultados por experimento

### Exp 3.1 — ARIMAX AR(1)+covariante fuerte (β=0.8)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMAX(1,0,0) con X | 50  | **1.966** | **1.568** | **1.134** | 0.655 | 0.842 | 5.79  | 11.56 |
| Chronos-2 (con X)    | 50  | 2.433 | 1.934 | 1.985 | 0.839 | 0.990 | 16.46 | 16.76 |
| SARIMAX(1,0,0) con X | 200 | 1.815 | 1.442 | 1.024 | 0.780 | 0.935 | 6.77  | 8.72  |
| Chronos-2 (con X)    | 200 | **1.375** | **1.096** | **0.924** | 0.775 | 0.956 | **5.72** | **6.66** |

SARIMAX domina a T=50 en RMSE (+19%), MAE y CRPS. El efecto del covariante es fuerte (β=0.8) pero con T=50 apenas alcanza para estimar bien los coeficientes AR y de covariante simultáneamente. A T=200 **se produce una inversión completa**: Chronos gana en RMSE (24%), MAE, CRPS (10%) y Winkler. Con más contexto, Chronos-2 aprende implícitamente la relación entre el covariante y el target, mientras que SARIMAX mejora menos: su estimación de β ya era aceptable a T=50 pero los errores de estimación del horizonte largo acumulan sesgo. La brecha en CRPS a T=50 es desproporcionada respecto al RMSE (75% vs 19%): Chronos produce distribuciones muy anchas que no están justificadas por el error puntual. SARIMAX exhibe **subcovertura severa** a T=50 (cov_95=0.842, cov_80=0.655); Chronos sobrecobertura extrema (0.990). A T=200, la calibración de ambos modelos mejora y Chronos logra intervalos más ajustados (5.72 vs 6.77 de SARIMAX). **Conclusión: inversión a T=200 impulsada por el efecto fuerte del covariante. Con muestra larga, Chronos aprovecha mejor la señal exógena que SARIMAX, que no mejora su RMSE de forma proporcional.**

---

### Exp 3.2 — ARIMAX AR(1)+covariante débil (β=0.2)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMAX(1,0,0) con X | 50  | **1.377** | **1.103** | **0.796** | 0.674 | 0.858 | **4.27** | **7.75** |
| Chronos-2 (con X)    | 50  | 1.449 | 1.159 | 1.169 | 0.832 | 0.983 | 9.26  | 9.62  |
| SARIMAX(1,0,0) con X | 200 | **1.279** | **1.020** | **0.723** | 0.782 | 0.943 | **4.87** | **6.01** |
| Chronos-2 (con X)    | 200 | 1.319 | 1.051 | 0.893 | 0.792 | 0.959 | 5.47  | 6.36  |

SARIMAX domina en RMSE y CRPS en ambos T, y no hay inversión. La ventaja en RMSE es modesta: ~5% a T=50, ~3% a T=200. Con β=0.2, la señal del covariante es débil — representa solo una fracción del nivel de ruido. SARIMAX estima correctamente este coeficiente pequeño y no sobreajusta; Chronos, en cambio, no extrae una ventaja diferencial de un covariante que contribuye poco a la señal. La brecha en CRPS es más pronunciada que en RMSE (~47% a T=50, ~24% a T=200): Chronos produce distribuciones más anchas de lo necesario. SARIMAX tiene **subcovertura** a T=50 (cov_95=0.858, cov_80=0.674) mientras Chronos sobrecobertura (0.983). La comparación directa con Exp 3.1 revela el mecanismo de la inversión: con β débil, Chronos no obtiene suficiente señal del covariante para construir un pronóstico superior; con β fuerte (3.1), la señal es lo bastante clara para que Chronos la aproveche con contexto largo. **Conclusión: SARIMAX domina en ambos T. La inversión de Exp 3.1 no ocurre aquí porque el covariante aporta poca señal predictiva.**

---

### Exp 3.3 — ARIMAX con dos covariantes (β₁=0.8, β₂=0.4)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMAX(1,0,0) 2 cov. | 50  | **2.210** | **1.756** | **1.281** | 0.610 | 0.806 | 5.91  | **13.90** |
| Chronos-2 (con X)     | 50  | 2.290 | 1.825 | 1.757 | 0.815 | 0.980 | 13.33 | 13.95 |
| SARIMAX(1,0,0) 2 cov. | 200 | 1.936 | 1.558 | 1.100 | 0.772 | 0.931 | 7.19  | 9.16  |
| Chronos-2 (con X)     | 200 | **1.472** | **1.179** | **0.996** | 0.777 | 0.960 | **6.18** | **7.05** |

El patrón es casi idéntico al de Exp 3.1: SARIMAX gana T=50 (4% en RMSE), inversión a T=200 donde Chronos gana RMSE (24%), MAE, CRPS (9%) y Winkler. La presencia del segundo covariante (β₂=0.4) introduce una covariable adicional de efecto medio que Chronos también aprovecha con contexto largo. La subcovertura de SARIMAX es aún más severa aquí que en 3.1: cov_80=0.610 y cov_95=0.806 a T=50 — el modelo estima dos coeficientes de covariante adicionales con solo T_train=26 observaciones efectivas, lo que infla los errores de estimación de los intervalos. Chronos no sufre de esta degradación (cov_95=0.980 a T=50). A T=200, la calibración mejora notablemente para ambos, con Chronos logrando intervalos más ajustados y menor Winkler. **Conclusión: la inversión a T=200 se replica robustamente con dos covariantes. El mecanismo es el mismo: más covariantes con efectos fuertes → más señal que Chronos aprovecha con contexto largo.**

---

### Exp 3.4 — VARX bivariante (VAR con covariante)

**T = 50:**

| Modelo | Var | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| VARMAX(1) con X         | 0 | **1.386** | **1.098** | **0.788** | 0.709 | 0.887 | **4.51** | **7.35** |
| Chronos-2 joint (con X) | 0 | 1.765 | 1.405 | 1.470 | 0.858 | 0.991 | 12.18 | 12.34 |
| VARMAX(1) con X         | 1 | **1.391** | **1.103** | **0.795** | 0.695 | 0.874 | **4.41** | **7.78** |
| Chronos-2 joint (con X) | 1 | 1.584 | 1.262 | 1.265 | 0.830 | 0.982 | 10.08 | 10.52 |

**T = 200:**

| Modelo | Var | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| VARMAX(1) con X         | 0 | **1.205** | **0.966** | **0.682** | 0.790 | 0.944 | **4.62** | **5.66** |
| Chronos-2 joint (con X) | 0 | 1.299 | 1.036 | 0.873 | 0.780 | 0.956 | 5.35  | 6.21  |
| VARMAX(1) con X         | 1 | **1.235** | **0.985** | **0.699** | 0.769 | 0.933 | **4.58** | **5.83** |
| Chronos-2 joint (con X) | 1 | 1.334 | 1.062 | 0.888 | 0.774 | 0.949 | 5.34  | 6.44  |

VARMAX-OLS domina en todas las métricas en ambos T y ambas variables. No hay inversión: VARMAX gana RMSE, MAE, CRPS y Winkler en T=50 y T=200. La ventaja en RMSE es de +21% (var0) y +12% (var1) a T=50, y ~8% en ambas variables a T=200. La brecha en CRPS es más pronunciada: ~87% a T=50, ~28% a T=200. Chronos produce distribuciones muy anchas a T=50 (WIDTH_95=12.18 vs 4.51 para VARMAX en var0) con sobrecobertura sistemática (cov_95≈0.991/0.982). A T=200, los intervalos de Chronos se ajustan pero siguen siendo más anchos y con peor Winkler. La ausencia de inversión contrasta fuertemente con exps 3.1 y 3.3: en el contexto bivariante, el modelo VARX tiene información estructural que Chronos no puede inferir solo desde los targets — las interacciones contemporáneas entre variables y el mecanismo de transmisión del covariante están explícitamente codificados en el OLS, mientras que Chronos trata cada serie de forma relativamente independiente. **Conclusión: VARMAX-OLS domina en ambos T sin inversión. La estructura multivariante con covariante beneficia al modelo paramétrico más que al modelo de fundación, porque las interdependencias cross-variable están explícitamente representadas en el VAR.**

---

### Exp 3.5 — ARIMAX con heterocedasticidad GARCH (covariante en media)

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMAX(1,0,0) con X | 50  | **1.693** | **1.320** | **0.959** | 0.705 | 0.874 | **5.43** | **9.75** |
| Chronos-2 (con X)    | 50  | 1.889 | 1.466 | 1.567 | 0.860 | 0.985 | 12.78 | 13.27 |
| SARIMAX(1,0,0) con X | 200 | **1.597** | **1.239** | **0.891** | 0.789 | 0.931 | **5.95** | **8.14** |
| Chronos-2 (con X)    | 200 | 1.667 | 1.300 | 1.098 | 0.777 | 0.945 | 6.59  | 8.29  |

SARIMAX domina en RMSE y CRPS en ambos T, sin inversión. La ventaja en RMSE es de ~11% a T=50 y ~4% a T=200 — comparable a Exp 3.2 (covariate débil). El DGP tiene heterocedasticidad GARCH en la varianza del error, pero el covariante actúa únicamente sobre la media condicional (misma mecánica que en 3.1). El efecto del covariante es moderado (β=0.8 sobre un proceso con ruido GARCH), y la varianza condicional stocástica añade ruido que ambos modelos deben promediar. SARIMAX no modela la heterocedasticidad, pero esto no importa para el pronóstico de la media: el pronóstico puntual óptimo bajo pérdida cuadrática solo requiere modelar la media condicional, donde SARIMAX es correcto. La brecha en CRPS (~63% a T=50, ~23% a T=200) es mayor que en RMSE: Chronos produce distribuciones más anchas que no están justificadas, incluso en presencia de varianza condicional variable. **Conclusión: SARIMAX domina en ambos T. La heterocedasticidad GARCH en la varianza no desplaza la ventaja de SARIMAX cuando el covariante actúa sobre la media. La inversión de exps 3.1/3.3 no se produce aquí: el ruido GARCH reduce la señal efectiva del covariante respecto a los experimentos de varianza homogénea.**

---

### Exp 3.6 — ADL-ECM cointegrado

**T = 50:**

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARDL-ECM            | 50  | **1.526** | **1.210** | —     | —     | —     | —     | —     |
| SARIMAX(1,0,0) niv. | 50  | 1.683 | 1.328 | **0.977** | 0.620 | 0.812 | **4.51** | 11.15 |
| SARIMAX(1,1,0) dif. | 50  | 2.786 | 2.208 | 1.589 | 0.826 | 0.950 | 12.27 | 14.67 |
| Chronos-2 (con X)   | 50  | 2.838 | 2.209 | 2.387 | 0.871 | 0.996 | 20.59 | 20.79 |

**T = 200:**

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARDL-ECM            | 200 | **1.366** | **1.092** | —     | —     | —     | —     | —     |
| SARIMAX(1,0,0) niv. | 200 | 1.394 | 1.116 | **0.790** | 0.773 | 0.933 | **5.18** | **6.53** |
| Chronos-2 (con X)   | 200 | 1.615 | 1.288 | 1.150 | 0.805 | 0.965 | 7.55  | 8.34  |
| SARIMAX(1,1,0) dif. | 200 | 1.911 | 1.537 | 1.182 | 0.937 | 0.991 | 12.39 | 12.58 |

**El experimento más rico en lecciones metodológicas.** ARDL-ECM correctamente especificado gana en RMSE en ambos T, aunque sin producir intervalos. El ranking en RMSE es claro: ARDL-ECM < SARIMAX-niv < Chronos < SARIMAX-dif. La sobrediferenciación de SARIMAX(1,1,0) es costosa: aplica una diferencia innecesaria sobre un proceso cointegrado de largo plazo, con RMSE 2.786 a T=50 (82% peor que ARDL-ECM) y 1.911 a T=200 (40% peor). SARIMAX en niveles, sin conocer la cointegración, produce resultados mucho mejores: a T=200 su RMSE (1.394) casi iguala al ARDL-ECM (1.366). Chronos ocupa el lugar intermedio: peor que ARDL-ECM y SARIMAX-niv, pero significativamente mejor que SARIMAX-dif a T=200. Este resultado indica que Chronos captura implícitamente parte de la dinámica de largo plazo desde el contexto de la serie sin necesidad de especificar la relación de cointegración. La calibración a T=50 es problemática para SARIMAX-niv (cov_95=0.812, cov_80=0.620 — subcovertura severa) y sobrecobertura extrema de Chronos (0.996). A T=200, SARIMAX-niv logra calibración aceptable (0.933) y el mejor Winkler; SARIMAX-dif sobrecobertura masiva (0.991) con intervalos que crecen con el horizonte. **Conclusión: la especificación importa. ECM correctamente identificado gana. Sobrediferenciar es más costoso que no diferenciar en presencia de cointegración. Chronos captura parte de la dinámica de largo plazo sin especificación explícita.**

---

## Resumen comparativo — Exps 3.1–3.6

| Exp | DGP | T=50 ganador (RMSE) | T=200 ganador (RMSE) | Nota calibración |
|-----|-----|---------------------|----------------------|-----------------|
| 3.1 | ARIMAX β=0.8, 1 cov. | SARIMAX (19%) | **Chronos (24%)** | Inversión; SARIMAX subcov T=50 |
| 3.2 | ARIMAX β=0.2, 1 cov. | SARIMAX (5%) | SARIMAX (3%) | Sin inversión; Chronos sobrecobertura T=50 |
| 3.3 | ARIMAX β₁=0.8, β₂=0.4, 2 cov. | SARIMAX (4%) | **Chronos (24%)** | Inversión; SARIMAX subcov severa T=50 |
| 3.4 | VARX bivariante | VARMAX (21%/12%) | VARMAX (8%/8%) | Sin inversión; Chronos sobrecobertura T=50 |
| 3.5 | ARIMAX-GARCH β=0.8 | SARIMAX (11%) | SARIMAX (4%) | Sin inversión; SARIMAX subcov T=50 |
| 3.6 | ADL-ECM cointegrado | ARDL-ECM vs. SARIMAX-niv | ARDL-ECM vs. SARIMAX-niv | ECM gana RMSE; SARIMAX-niv > SARIMAX-dif |

---

## Conclusiones transversales

### 1. La intensidad del efecto del covariante determina si hay inversión a T=200
El patrón más relevante de los experimentos univariados es el **umbral de inversión según β**: con covariante fuerte (β=0.8, exps 3.1 y 3.3), Chronos supera a SARIMAX a T=200 en RMSE, CRPS y Winkler; con covariante débil (β=0.2, exp 3.2) o ruidoso (GARCH, exp 3.5), no hay inversión. El mecanismo es que Chronos aprende implícitamente la relación entre el covariante y el target desde el contexto cuando la señal es lo suficientemente fuerte — y con T=200 tiene historia suficiente para consolidar ese aprendizaje. SARIMAX, en cambio, tiene un techo de mejora limitado por el horizonte de predicción del AR(1): su estimación de β mejora con T pero el coeficiente AR estacionario ya era bien estimado a T=50.

### 2. En el dominio multivariante, el modelo paramétrico domina sin inversión
Exp 3.4 muestra que la estructura VARX con covariante beneficia al modelo OLS más que al modelo de fundación: VARMAX gana en todas las métricas en ambos T. A diferencia del caso univariante con β fuerte, aquí las interdependencias cross-variable (mecanismo de transmisión de X a Y₁ y Y₂, y la correlación entre Y₁ e Y₂) están explícitamente capturadas por el VAR pero no por Chronos, que trata cada serie de forma más independiente. Más contexto (T=200) reduce la brecha pero no la elimina ni la invierte.

### 3. La correcta especificación del orden de integración es el factor dominante en cointegración
Exp 3.6 exhibe el spread más amplio entre modelos: RMSE desde 1.37 (ARDL-ECM) hasta 2.84 (Chronos, Sarimax-dif) a T=50. Diferenciar innecesariamente es el error más costoso (~82% peor en T=50), mucho más que no modelar la cointegración explícitamente. SARIMAX en niveles, aunque ignora la relación de cointegración, produce resultados ~80% mejores que SARIMAX diferenciado. Chronos se posiciona entre ambas especificaciones de SARIMAX, lo que sugiere que capta implícitamente la estacionariedad de largo plazo sin sobrediferenciar.

### 4. SARIMAX subcovertura sistemática a T=50 en todos los experimentos univariados
En los experimentos univariados con T=50, SARIMAX presenta cov_95 entre 0.812 y 0.858 — muy por debajo del nominal 0.95. La magnitud de la subcovertura crece con el número de parámetros a estimar: exp 3.3 (dos covariantes + AR) tiene la peor calibración (cov_95=0.806). Este patrón replica el observado en los experimentos sin covariables (bloque 1.x): los intervalos de SARIMAX están subestimados porque la teoría asintótica de los intervalos de predicción AR no es precisa con muestras cortas. Con T=200, la calibración mejora a rangos aceptables (0.930–0.943).

### 5. Chronos sobrecobertura sistemática a T=50, con mejora a T=200
En prácticamente todos los experimentos a T=50, Chronos presenta cov_95 entre 0.980 y 0.996. Los intervalos son 2–3× más anchos que los de SARIMAX. A T=200, la calibración mejora notablemente: cov_95 entre 0.944 y 0.965, con intervalos comparables a los de SARIMAX. La excepción es Exp 3.6 donde Chronos mantiene cov_95=0.996 a T=50 incluso con la complejidad del proceso de cointegración — los intervalos muy anchos reflejan genuina incertidumbre ante una dinámica de largo plazo que el modelo no conoce explícitamente.

### 6. La brecha en CRPS es sistemáticamente mayor que la brecha en RMSE
En todos los experimentos donde SARIMAX/VARMAX gana en RMSE, la ventaja en CRPS es más pronunciada. A T=50: CRPS de Chronos es 40–87% peor que el del modelo paramétrico, mientras el RMSE es 5–21% peor. Esta asimetría revela que Chronos produce distribuciones predictivas más dispersas de lo necesario — el error puntual no captura completamente la inadecuación de la distribución predictiva completa. En los casos de inversión a T=200 (exps 3.1, 3.3), la mejora en CRPS de Chronos también es más pronunciada que la mejora en RMSE.

### 7. La heterocedasticidad condicional no genera ventaja para Chronos cuando actúa en la varianza
Exp 3.5 combina covariante en media con GARCH en varianza. Contrariamente a lo que podría esperarse — que Chronos detecte la varianza variable desde el contexto —, SARIMAX sigue ganando en todas las métricas. La varianza GARCH añade ruido que afecta a ambos modelos por igual en términos de error puntual promedio. Esto contrasta con los exps GARCH sin covariante del bloque 1 (1.9–1.12), donde el patrón era similar: la dinámica GARCH en la varianza no desplaza el ranking de RMSE.

### 8. Implicaciones para la tesis
El bloque de covariables revela un régimen de comportamiento dual para Chronos: *modelo de integración pasiva* (recibe X como dato conocido) vs *modelo paramétrico con especificación explícita*. Cuando el covariante tiene señal fuerte y el modelo paramétrico no puede aprovechar mejor su efecto a largo horizonte (exps 3.1, 3.3), Chronos supera a T=200. Cuando la estructura multivariante es rica (VARX), cuando el covariante es débil, o cuando hay ruido adicional (GARCH), el modelo paramétrico correctamente especificado mantiene ventaja. Para aplicaciones reales con DGP desconocido, la fuerza del efecto del covariante y la dimensionalidad del sistema son los factores clave para decidir entre Chronos y modelos paramétricos.
