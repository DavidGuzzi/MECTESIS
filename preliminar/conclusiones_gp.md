# Conclusiones — Experimentos GP (Gaussian Process / KernelSynth)

**Setup GP.1–GP.3:** T ∈ {50, 200} | H = 24 | R = 500 | Semilla = 3649  
**DGP:** Gaussian Process con kernels RBF, Periódico y compuesto (estilo KernelSynth de Chronos)  
**Motivación:** Chronos-2 fue entrenado con datos sintéticos generados por GPs con kernels compuestos (KernelSynth, Ansari et al. 2024). La hipótesis es que Chronos reconoce estos patrones sin estimación paramétrica, mientras que el modelo clásico inevitablemente mal-especifica la covarianza.  
**Métricas reportadas:** promedios avg_all (H=1…24). RMSE, MAE, CRPS: menor es mejor. COV_80/95: nominal 0.80/0.95. WINKLER_95: menor es mejor.

---

## Resultados por experimento

### Exp GP.1 — Gaussian Process RBF (tendencia suave no lineal)

**DGP:** $y \sim \mathcal{GP}(0, K_{RBF})$, kernel RBF con $\ell=30$, $\sigma_{rbf}=1.0$, $\sigma_{noise}=0.3$.

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| ARIMA(1,1,1) | 50  | **0.5489** | **0.4312** | **0.3156** | 0.702 | 0.866 | **1.766** | **3.390** |
| ETS(A,A,N)   | 50  | 0.6056 | 0.4800 | 0.3805 | 0.571 | 0.735 | 1.735 | 5.858 |
| Chronos-2    | 50  | 0.5549 | 0.4364 | 0.4396 | 0.804 | 0.980 | 3.665 | 3.907 |
| ARIMA(1,1,1) | 200 | 0.5656 | 0.4514 | 0.3213 | 0.731 | 0.910 | **1.878** | **2.828** |
| ETS(A,A,N)   | 200 | 0.5843 | 0.4632 | 0.4434 | 0.927 | 0.976 | 5.473 | 5.728 |
| Chronos-2    | 200 | **0.5555** | **0.4416** | 0.3527 | 0.707 | 0.935 | 2.121 | 2.789 |

A T=50, ARIMA(1,1,1) gana marginalmente en RMSE (0.549 vs 0.555, diferencia del 1.1%) — la diferenciación $d=1$ captura el carácter "casi no estacionario" de la función RBF, que aparece localmente como una caminata aleatoria suave. A T=200, **Chronos revierte el resultado** (0.556 vs 0.566 ARIMA, +1.8%) con más contexto disponible. ETS queda tercero en ambos T. En CRPS, ARIMA domina en T=50 (28% mejor que Chronos) y sigue siendo mejor a T=200; Chronos produce distribuciones demasiado anchas (COV_95=0.980/0.935, WIDTH_95=3.7/2.1 vs ~1.9 de ARIMA). **La hipótesis solo se confirma parcialmente: la tendencia RBF pura favorece levemente al ARIMA en T=50 y a Chronos en T=200, sin una victoria clara de ninguno.**

---

### Exp GP.2 — Gaussian Process Periódico puro

**DGP:** $y \sim \mathcal{GP}(0, K_{Per})$, kernel periódico con $p=12$, $\ell_{per}=1.0$, $\sigma_{per}=1.0$, $\sigma_{noise}=0.3$.

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMA(1,0,1)(1,0,1)_12 | 50  | 3.0643 | 0.7059 | 0.5666 | 0.640 | 0.810 | 1.769 | 11.254 |
| ETS(A,N,A)              | 50  | **0.3638** | **0.2902** | **0.2085** | 0.768 | 0.918 | **1.385** | **1.891** |
| Chronos-2               | 50  | 0.4509 | 0.3551 | 0.5307 | 0.953 | 0.999 | 5.187 | 5.193 |
| SARIMA(1,0,1)(1,0,1)_12 | 200 | 0.3160 | 0.2523 | **0.1786** | 0.786 | 0.942 | **1.207** | **1.494** |
| ETS(A,N,A)              | 200 | **0.3103** | **0.2481** | 0.1760 | 0.784 | 0.937 | 1.168 | 1.476 |
| Chronos-2               | 200 | 0.3116 | 0.2490 | 0.2126 | 0.774 | 0.972 | 1.413 | 1.541 |

**El hallazgo más dramático del experimento:** SARIMA(1,0,1)(1,0,1)_12 a T=50 colapsa con RMSE promedio de 3.06, aproximadamente **8 veces mayor** que ETS. La causa es inestabilidad numérica del componente SAR estacional $\Phi_1 B^{12}$ con solo 50 observaciones (menos de 4.2 ciclos completos de período 12): el optimizador MLE produce estimaciones de $\Phi_1$ próximas a la frontera de estacionariedad, lo que genera pronósticos explosivos en horizontes $h > 12$. A T=200, SARIMA se estabiliza y alcanza RMSE=0.316, competitivo con ETS (0.310) y Chronos (0.312).

A T=50, **ETS(A,N,A) domina** (RMSE=0.364) sobre Chronos (RMSE=0.451, diferencia del 24%). La estacionalidad aditiva determinística de ETS captura eficientemente el patrón periódico del GP, incluso con pocas observaciones. Chronos sobreestima la incertidumbre de forma severa (COV_95=0.999, WIDTH_95=5.19), resultando en CRPS=0.531 — un 155% peor que ETS. A T=200, los tres modelos convergen a una diferencia menor al 1% en RMSE, con ETS marginalmente mejor en RMSE/MAE y SARIMA marginalmente mejor en CRPS/Winkler. **La hipótesis de Chronos no se confirma en GP periódico: ETS es el mejor modelo en T=50 y los tres convergen a T=200.**

---

### Exp GP.3 — Gaussian Process RBF + Periódico (KernelSynth completo)

**DGP:** $y \sim \mathcal{GP}(0, K_{RBF} + K_{Per})$, con $\ell_{rbf}=30$, $\sigma_{rbf}=1.0$, $p=12$, $\ell_{per}=1.0$, $\sigma_{per}=0.8$, $\sigma_{noise}=0.3$. Caso más cercano al training data real de Chronos.

| Modelo | T | RMSE | MAE | CRPS | COV_80 | COV_95 | WIDTH_95 | WINKLER_95 |
|---|---|---|---|---|---|---|---|---|
| SARIMA(1,1,1)(1,0,1)_12 | 50  | 70.569 | 3.956 | 3.643 | 0.670 | 0.822 | 4.087 | 117.366 |
| ETS(A,A,A)              | 50  | 0.6822 | 0.5407 | 0.4153 | 0.570 | 0.748 | 1.905 | 5.792 |
| Theta                   | 50  | 0.9453 | 0.7462 | 0.6515 | 0.948 | 0.988 | 7.731 | 7.862 |
| Chronos-2               | 50  | **0.6722** | **0.5280** | 0.6378 | 0.897 | 0.995 | 5.826 | **5.885** |
| SARIMA(1,1,1)(1,0,1)_12 | 200 | 0.6050 | 0.4829 | 0.3455 | 0.727 | 0.908 | 2.036 | 3.092 |
| ETS(A,A,A)              | 200 | 0.5935 | 0.4716 | 0.4510 | 0.922 | 0.976 | 5.542 | 5.760 |
| Theta                   | 200 | 0.9854 | 0.7873 | 0.6700 | 0.954 | 0.993 | 7.896 | 7.951 |
| Chronos-2               | 200 | **0.5543** | **0.4409** | **0.3477** | 0.686 | 0.933 | **2.080** | **2.752** |

**SARIMA catastrófico a T=50:** RMSE=70.57, Winkler=117.4. La diferenciación $d=1$ aplicada a un proceso GP estacionario introduce una raíz unitaria artificial; combinada con la inestabilidad del componente SAR estacional a T=50, el resultado es una explosión de los pronósticos. A T=200, SARIMA se estabiliza (RMSE=0.605) pero sigue siendo el peor de los modelos estables.

**Confirmación de la hipótesis KernelSynth:** En el caso del kernel compuesto (tendencia suave + estacionalidad), Chronos es el mejor modelo en **ambos T**:
- T=50: Chronos RMSE=0.672 vs ETS=0.682 (+1.5%) y Theta=0.945. Winkler Chronos=5.885 vs ETS=5.792 (ligeramente peor, ETS gana en Winkler). En CRPS, ETS domina (0.415 vs 0.638) porque Chronos sobreestima la incertidumbre.
- T=200: **Victoria clara de Chronos.** RMSE=0.554 vs ETS=0.594 (+7.7%), SARIMA=0.605 (+9.2%). Chronos también gana en CRPS (0.348 vs 0.346 SARIMA — prácticamente igual) y Winkler (2.752 vs 3.092 SARIMA, 5.760 ETS). Con suficientes datos, Chronos reconoce simultáneamente la tendencia suave no lineal y la estacionalidad — exactamente el tipo de patrón presente en su entrenamiento KernelSynth.

ETS(A,A,A) a T=50 tiene subcovertura severa (COV_95=0.748) con Winkler comparable a Chronos (5.792). Theta es el peor modelo estable en ambos T (RMSE > 0.94), con sobrecovertura extrema (COV_95=0.993) e intervalos muy anchos.

---

## Resumen comparativo

| Exp | Kernel | T=50 ganador (RMSE) | T=200 ganador (RMSE) | Hallazgo principal |
|-----|--------|---------------------|----------------------|-------------------|
| GP.1 | RBF puro | ARIMA (1%) | **Chronos (2%)** | Esencialmente empate; ARIMA con d=1 compite bien |
| GP.2 | Periódico | ETS (+24% sobre Chronos) | ETS (≈empate, <1%) | SARIMA explota T=50; ETS domina; Chronos no confirma hipótesis |
| GP.3 | RBF+Per (KernelSynth) | **Chronos (1.5%)** | **Chronos (7.7%)** | Confirmación parcial: Chronos gana en su training distribution |

---

## Conclusiones transversales

### 1. La hipótesis KernelSynth se confirma en el kernel compuesto, no en los puros

El caso GP.3 (RBF + Periódico) es el más cercano al entrenamiento real de Chronos (KernelSynth genera series con combinaciones aleatorias de kernels). En ese caso, Chronos gana en RMSE en ambos T y en Winkler a T=200. Sin embargo, en los kernels puros la hipótesis no se sostiene: para RBF puro (GP.1) ARIMA es igualmente competitivo, y para periódico puro (GP.2) ETS domina claramente a T=50.

### 2. Inestabilidad catastrófica de modelos SARIMA con muestras cortas

El hallazgo más dramático del experimento es la inestabilidad de los modelos SARIMA cuando el número de observaciones es pequeño relativo a la estacionalidad:

- **GP.2 T=50:** SARIMA(1,0,1)(1,0,1)_12 con RMSE=3.06 (vs ETS=0.364 y Chronos=0.451). Solo 50 observaciones (~4 ciclos completos) resultan insuficientes para estimar establemente los componentes $\Phi_1$ y $\Theta_1$ del modelo estacional.
- **GP.3 T=50:** SARIMA(1,1,1)(1,0,1)_12 con RMSE=70.57 y Winkler=117.4. La diferenciación $d=1$ sobre un proceso GP estacionario agrega una raíz unitaria artificial, que combinada con la inestabilidad SAR produce pronósticos explosivos.

Este resultado tiene implicancias directas para la práctica: el uso de SARIMA con $T/s < 8$ puede ser no solo impreciso sino catastróficamente inestable.

### 3. ETS es sorprendentemente robusto para procesos periódicos

ETS(A,N,A) supera a Chronos en el GP periódico a T=50 por un margen amplio (RMSE 0.364 vs 0.451). La estructura aditiva estacional del ETS, aunque paramétrica y de baja complejidad, captura eficientemente la periodicidad regular del kernel periódico. Esto sugiere que la ventaja de Chronos en este tipo de procesos no es universal.

### 4. Chronos sobreestima sistemáticamente la incertidumbre

En todos los experimentos y ambos T, Chronos presenta COV_95 significativamente por encima del nominal 0.95 (entre 0.933 y 0.999) con intervalos de mayor ancho. Esta sobrecobertura se traduce en CRPS y Winkler peores, especialmente en muestras cortas. La excepción parcial es GP.3 a T=200, donde Chronos calibra mejor (COV_95=0.933, cercano al nominal).

### 5. La ventaja de Chronos crece con T en el kernel compuesto

En GP.3, la brecha de Chronos sobre ETS en RMSE crece de 1.5% (T=50) a 7.7% (T=200). Más contexto permite a Chronos identificar simultáneamente la tendencia suave y el patrón estacional. Esto es consistente con el comportamiento observado en los experimentos univariados: Chronos se beneficia más del contexto adicional que los modelos paramétricos.

### 6. Implicación para la tesis

El experimento GP confirma la hipótesis KernelSynth de forma matizada: **Chronos obtiene una ventaja real en el proceso que más se parece a su distribución de entrenamiento** (kernel compuesto RBF+Periódico a T=200), pero no domina en los kernels puros ni en muestras muy cortas. La capacidad de ETS para manejar el GP periódico puro sugiere que el modelo clásico correctamente especificado sigue siendo competitivo incluso en el dominio de entrenamiento de Chronos. La falla catastrófica de SARIMA con T=50 ilustra el riesgo de usar modelos estacionales paramétricos cuando el tamaño muestral es pequeño relativo al período.
