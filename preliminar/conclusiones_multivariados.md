# Conclusiones — Experimentos multivariados 2.1–2.7

**Setup 2.1–2.7:** $T \in \{50, 200\}$ | $H=24$ | $R=500$ | Semilla $=3649$  
**Valores reportados:** promedios sobre las $k$ variables y los 24 horizontes (`avg_all`).  
**Métricas:** RMSE, MAE, CRPS: menor es mejor. COV\_80/95: nominal 0.80/0.95. WINKLER\_95: menor es mejor.  
**Modelos:** VAR(p) (statsmodels), VECM (statsmodels, Johansen), VAR+GARCH-diag (arch), Chronos-2 (joint/ind.).

---

## Resultados por experimento

### Exp 2.1 — VAR(1) bivariado, baja interdependencia ($k=2$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VAR(1)            | 50  | **1.234** | **0.984** | **0.702** | 0.765 | 0.921 | **4.59**  | **6.15**  |
| Chronos-2 (joint) | 50  | 1.325     | 1.055     | 1.216     | 0.890 | 0.996 | 10.83     | 10.89     |
| Chronos-2 (ind.)  | 50  | 1.333     | 1.063     | 1.238     | 0.891 | 0.996 | 11.25     | 11.31     |
| VAR(1)            | 200 | **1.181** | **0.942** | **0.667** | **0.799** | **0.948** | **4.61** | **5.62** |
| Chronos-2 (joint) | 200 | 1.193     | 0.952     | 0.820     | 0.809 | 0.964 | 5.11      | 5.75      |
| Chronos-2 (ind.)  | 200 | 1.208     | 0.963     | 0.836     | 0.808 | 0.966 | 5.30      | 5.90      |

VAR(1) domina en todas las métricas. La ventaja en RMSE es modesta (+7% a $T=50$, +1% a $T=200$) pero la brecha en CRPS es desproporcionada: +73% a $T=50$, +23% a $T=200$. Chronos-joint y Chronos-ind son prácticamente iguales — la ventaja del modelado conjunto es mínima cuando la interdependencia es baja (coeficientes off-diagonal $0.1$), confirmando la hipótesis. Chronos sobrecobertura sistemáticamente (cov\_95$\approx0.996$) con intervalos 2.4× más anchos a $T=50$, lo que resulta en Winkler\_95 casi el doble. A $T=200$, los intervalos de Chronos se ajustan pero siguen siendo más anchos. VAR tiene calibración cercana al nivel nominal en ambos $T$ (cov\_95$=0.921/0.948$). **Conclusión: VAR domina en precisión y calibración. Con baja interdependencia, la API joint de Chronos no aporta frente a pronósticos independientes.**

---

### Exp 2.2 — VAR(1) bivariado, alta interdependencia ($k=2$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VAR(1)            | 50  | **1.561** | **1.234** | **0.886** | 0.715 | 0.888 | **5.16**  | **8.27**  |
| Chronos-2 (joint) | 50  | 1.658     | 1.321     | 1.416     | 0.857 | 0.991 | 12.36     | 12.57     |
| VAR(1)            | 200 | **1.425** | **1.133** | **0.805** | 0.791 | **0.940** | **5.47** | **6.87** |
| Chronos-2 (joint) | 200 | 1.500     | 1.192     | 1.025     | 0.798 | 0.960 | 6.45      | 7.31      |

VAR domina en todas las métricas (+6% RMSE a $T=50$, +5% a $T=200$). El RMSE es mayor que en 2.1 porque el eigenvalor dominante de $A_1$ es $0.8$ (mayor persistencia). La brecha en CRPS es +60% a $T=50$ y +27% a $T=200$. Importante: VAR tiene **subcovertura** a $T=50$ (cov\_80$=0.715$, cov\_95$=0.888$) — la alta interdependencia hace la estimación más difícil con muestra corta, pero el Winkler de VAR sigue siendo mejor que el de Chronos. La ventaja de Chronos-joint sobre Chronos-ind esperada bajo alta interdependencia no puede verificarse aquí (Chronos-ind no se incluyó), pero la brecha VAR–Chronos-joint se mantiene en ~6% independientemente de $T$. **Conclusión: VAR domina, pero la alta interdependencia aumenta la dificultad de estimación y reduce la calibración del VAR a muestras cortas.**

---

### Exp 2.3 — VAR(2) bivariado ($k=2$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VAR(2)            | 50  | 1.386 | 1.053 | 0.770 | 0.758 | 0.916 | 5.12  | 7.01  |
| VAR(1)            | 50  | **1.311** | **1.044** | **0.747** | 0.746 | 0.910 | **4.88** | **6.72** |
| Chronos-2 (joint) | 50  | 1.410 | 1.123 | 1.255 | 0.878 | 0.995 | 11.09 | 11.19 |
| VAR(2)            | 200 | **1.243** | **0.991** | **0.702** | **0.794** | **0.946** | 4.83 | **5.95** |
| VAR(1)            | 200 | **1.243** | 0.990     | 0.702     | **0.794** | 0.945 | **4.82** | 5.95 |
| Chronos-2 (joint) | 200 | 1.266     | 1.009     | 0.870     | 0.806 | 0.962 | 5.44  | 6.14  |

**Resultado llamativo:** a $T=50$, VAR(1) supera a VAR(2) en RMSE (1.311 vs 1.386). El modelo misspecificado gana porque con solo $T_{train}=26$ observaciones, la estimación de los 2 parámetros adicionales de $A_2$ introduce más varianza de estimación que el sesgo por omisión de lag 2 ($A_2 = 0.1\,I$ es débil). A $T=200$ los dos modelos son prácticamente idénticos en todas las métricas — la contribución de $A_2$ al pronóstico es negligible. Chronos-joint no mejora sobre VAR(1) en RMSE a $T=50$ y es peor en CRPS (+68%). Chronos tampoco captura mejor la dinámica de lag 2 que VAR(1). **Conclusión: el costo de misspecificación del orden VAR es mínimo cuando $A_2$ es débil. Con muestras cortas, el modelo más parsimonioso (VAR(1)) domina al correctamente especificado (VAR(2)). Chronos no aprende el orden 2 implícitamente.**

---

### Exp 2.4 — VAR(1) trivariado ($k=3$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VAR(1)            | 50  | **1.276** | **1.014** | **0.723** | 0.755 | 0.918 | **4.61** | **6.42** |
| Chronos-2 (joint) | 50  | 1.323     | 1.057     | 1.125     | 0.871 | 0.992 | 9.37     | 9.52     |
| VAR(1)            | 200 | **1.203** | **0.962** | **0.680** | **0.790** | **0.944** | **4.63** | **5.70** |
| Chronos-2 (joint) | 200 | 1.208     | 0.965     | 0.822     | 0.798 | 0.960 | 5.05     | 5.79     |

VAR domina en todas las métricas. La ventaja en RMSE cae de +4% a $T=50$ a **+0.4% a $T=200$**: con 3 variables y muestra grande, Chronos-joint alcanza paridad en RMSE con el modelo correctamente especificado. La brecha en CRPS persiste: +56% a $T=50$, +21% a $T=200$. Esto refleja que aunque Chronos logra precisión puntual comparable, sus distribuciones predictivas siguen siendo más anchas de lo necesario (cov\_95$=0.992$ vs nominal $0.95$ a $T=50$). A $T=200$, Winkler de VAR y Chronos son comparables (5.70 vs 5.79). **Conclusión: la brecha en RMSE se comprime aceleradamente con $T$ al aumentar la dimensión — a $k=3$ ya hay cuasi-paridad puntual a $T=200$. El exceso distribucional de Chronos persiste en CRPS.**

---

### Exp 2.5 — VAR(1) pentavariado ($k=5$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VAR(1)            | 50  | **1.099** | **0.877** | **0.625** | **0.803** | **0.947** | **4.47** | **5.39** |
| Chronos-2 (joint) | 50  | 1.102     | 0.882     | 0.874     | 0.857 | 0.985 | 6.38     | 6.71     |
| VAR(1)            | 200 | **1.058** | **0.844** | **0.598** | **0.806** | **0.948** | **4.18** | **5.01** |
| Chronos-2 (joint) | 200 | 1.063     | 0.849     | 0.722     | 0.808 | 0.959 | 4.45     | 5.07     |

**Resultado más sorprendente del bloque:** con $k=5$ variables, VAR(1) y Chronos-joint tienen RMSE prácticamente idéntico a **ambos** $T$ (+0.3% a $T=50$, +0.5% a $T=200$). La maldición de dimensionalidad no se manifiesta en RMSE: con coeficientes cruzados débiles ($0.05$) y diagonal moderada ($0.3$), la estimación VAR es suficientemente precisa incluso con $T_{train}=26$ y 45 parámetros. La brecha en CRPS sigue siendo sustancial (+40% a $T=50$, +21% a $T=200$). VAR tiene calibración near-nominal en ambos $T$ (cov\_95$\approx0.947$) — mejor calibración que en $k=2$ con alta interdependencia. **Conclusión: empate en RMSE a toda $T$. VAR con ventaja sistemática en CRPS y calibración. La estructura dispersa de $A_1$ mitiga la maldición de dimensionalidad del VAR. Chronos no supera al VAR en puntos aunque tenga mayor capacidad representativa.**

---

### Exp 2.6 — VAR(1)+GARCH diagonal ($k=2$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VAR(1)+GARCH-diag | 50  | 1.240 | 0.966 | 0.735 | 0.591 | 0.749 | **3.38** | 10.95 |
| VAR(1)            | 50  | **1.239** | **0.964** | **0.696** | 0.749 | 0.908 | 4.41  | **6.58** |
| Chronos-2 (joint) | 50  | 1.307 | 1.020 | 1.155 | 0.873 | 0.993 | 10.07 | 10.21 |
| VAR(1)+GARCH-diag | 200 | 1.171 | 0.913 | 0.659 | 0.755 | 0.912 | **4.27** | 6.51 |
| VAR(1)            | 200 | **1.170** | **0.911** | **0.654** | **0.798** | **0.938** | 4.47  | **5.91** |
| Chronos-2 (joint) | 200 | 1.178     | 0.919     | 0.801     | 0.805 | 0.957 | 4.97  | 5.96  |

**El resultado más contraintuitivo del bloque.** El VAR+GARCH-diag tiene RMSE idéntico al VAR(1) estándar — esperado, ya que la dinámica de media es la misma. La sorpresa está en los **intervalos**: el VAR+GARCH-diag produce **peor calibración que el VAR simple**, con subcovertura severa a $T=50$ (cov\_95$=0.749$, winkler\_95$=10.95$ vs $6.58$) y subcovertura moderada a $T=200$ (cov\_95$=0.912$). El CRPS es también peor: $0.735$ vs $0.696$ a $T=50$, $0.659$ vs $0.654$ a $T=200$.

La causa es la **estimación ruidosa de los parámetros GARCH** con muestras cortas: el GARCH estima $(\hat\omega, \hat\alpha, \hat\beta)$ sobre $\approx 26$ residuos a $T=50$. La estimación MLE de GARCH es conocida por ser poco estable con muestras pequeñas; en particular, tiende a subestimar la varianza incondicional, produciendo intervalos demasiado estrechos. A $T=200$, la estimación mejora pero el VAR simple sigue siendo mejor en cov\_95 (0.938 vs 0.912). Chronos supera al VAR+GARCH en calibración a $T=50$ (cov\_95$=0.993$) pero sobrecobertura y tiene Winkler comparable al VAR+GARCH. A $T=200$, los tres modelos convergen en RMSE. **Conclusión: el GARCH diagonal no mejora la calibración de intervalos en la práctica — al contrario, la empeora a muestras pequeñas. La estimación MLE de GARCH con $T_{train}\approx26$ es insuficiente.**

---

### Exp 2.7 — VECM bivariado, cointegración rango 1 ($k=2$)

| Modelo | $T$ | RMSE | MAE | CRPS | COV\_80 | COV\_95 | WIDTH\_95 | WINKLER\_95 |
|---|---|---|---|---|---|---|---|---|
| VECM($r=1$)       | 50  | **3.450** | **2.757** | **2.217** | 0.600 | 0.777 | 9.66  | 27.09 |
| VAR(1)            | 50  | 6.084 | 3.648 | 2.878 | 0.487 | 0.661 | 7.99  | 50.50 |
| Chronos-2 (joint) | 50  | 3.802 | 3.024 | 2.724 | 0.721 | **0.952** | 20.49 | **24.06** |
| Chronos-2 (ind.)  | 50  | 3.821 | 3.041 | 2.749 | 0.713 | **0.952** | 20.84 | 24.52 |
| VECM($r=1$)       | 200 | **3.596** | **2.865** | **2.113** | 0.643 | 0.835 | 10.03 | 21.79 |
| VAR(1)            | 200 | 3.814 | 3.034 | 2.224 | 0.633 | 0.825 | 10.11 | 24.41 |
| Chronos-2 (joint) | 200 | 3.784 | 2.979 | 2.576 | **0.758** | **0.959** | 16.97 | **19.96** |
| Chronos-2 (ind.)  | 200 | 3.831 | 3.014 | 2.616 | 0.754 | 0.957 | 17.33 | 20.49 |

**El experimento más dramático del bloque.** Los resultados revelan cuatro fenómenos:

**1. VAR(1) colapsa a $T=50$:** RMSE$=6.084$ frente a VECM$=3.450$ (+76%) y Chronos$=3.802$ (+60%). Aplicar VAR a datos $I(1)$ cointegrados sin restricción de rango genera regresión espuria: el VAR en niveles estima ecuaciones que no capturan la corrección de desequilibrio, produciendo pronósticos que divergen del equilibrio de largo plazo. La subcovertura es catastrófica (cov\_95$=0.661$, Winkler$=50.50$).

**2. VAR(1) se recupera parcialmente a $T=200$:** RMSE cae de $6.084$ a $3.814$ — la brecha frente a VECM se reduce de $+76\%$ a $+6\%$. Con suficiente historia, el VAR en niveles puede aproximar parcialmente la relación de cointegración vía los coeficientes estimados, pero nunca alcanza la eficiencia del VECM correctamente especificado.

**3. Chronos-joint supera a VAR(1) en RMSE a $T=200$:** $3.784$ vs $3.814$ ($-0.8\%$). Es el único experimento del bloque donde Chronos gana en RMSE sobre el modelo clásico, y ocurre porque el modelo clásico (VAR) está **misspecificado** — no es el correcto para datos cointegrados. Chronos también domina en Winkler a $T=200$ ($19.96$ vs $24.41$) gracias a su mejor calibración (cov\_95$=0.959$). Sin embargo, en **CRPS**, el VECM sigue ganando ($2.113$ vs $2.576$).

**4. Todos los modelos tienen subcovertura:** incluso el VECM tiene cov\_95$<0.95$ en ambos $T$ (0.777 a $T=50$, 0.835 a $T=200$). La no-estacionariedad $I(1)$ hace que la varianza del error de pronóstico crezca con $h$: los intervalos correctamente calibrados deben ser muy anchos, y los modelos paramétricos los subestiman. Chronos produce intervalos más anchos (width\_95$\approx20$ vs $10$) y por eso tiene mejor cobertura, aunque a costa de mayor Winkler a $T=50$.

**Conclusión: en datos cointegrados, el VECM correctamente especificado domina en RMSE y CRPS en ambos $T$. Chronos supera a VAR(1) en RMSE y Winkler a $T=200$ — la única victoria de Chronos en el bloque, posible por misspecificación del benchmark. El modelo joint de Chronos supera al independiente en todos los $T$ y métricas, validando que captura parcialmente la dependencia de largo plazo.**

---

## Resumen comparativo — Exps 2.1–2.7

| Exp | DGP | $k$ | $T=50$ ganador (RMSE) | $T=200$ ganador (RMSE) | CRPS gap $T=50$ | Nota calibración |
|-----|-----|-----|-----------------------|------------------------|-----------------|------------------|
| 2.1 | VAR(1) baja interdep. | 2 | VAR (7%) | VAR (1%) | VAR (+73%) | VAR near-nominal; Chronos sobrecobertura 2× |
| 2.2 | VAR(1) alta interdep. | 2 | VAR (6%) | VAR (5%) | VAR (+60%) | VAR subcov T=50 (cov\_95=0.888) |
| 2.3 | VAR(2) | 2 | **VAR(1)** (misspec.) | Empate | VAR (+68%) | VAR(1) < VAR(2) a T=50 |
| 2.4 | VAR(1) trivariado | 3 | VAR (4%) | Empate (0.4%) | VAR (+56%) | VAR near-nominal; brecha RMSE se cierra |
| 2.5 | VAR(1) pentavariado | 5 | Empate (0.3%) | Empate (0.5%) | VAR (+40%) | Paridad RMSE en toda T; VAR mejor CRPS |
| 2.6 | VAR+GARCH diagonal | 2 | VAR≈VAR+G (0%) | VAR≈VAR+G (0%) | VAR (+60%) | VAR+GARCH peor calibración que VAR simple |
| 2.7 | VECM cointegrado | 2 | **VECM (43%)** | VECM (5%); Chronos > VAR | VECM mejor | VAR colapsa T=50; Chronos gana a VAR(1) T=200 |

---

## Conclusiones transversales

### 1. La ventaja del VAR en RMSE disminuye con la dimensión $k$
Mientras a $k=2$ la ventaja es 6–7% a $T=50$, a $k=3$ cae a 4% y a $k=5$ hay empate. La hipótesis de que Chronos escala mejor con la dimensión se confirma en RMSE puntual, pero no en CRPS: incluso con $k=5$, el CRPS de Chronos es 40% peor a $T=50$.

### 2. El CRPS revela brechas ocultas por el RMSE
En todos los experimentos con DGP VAR correctamente especificado, la brecha VAR–Chronos en CRPS es 3–5× mayor que en RMSE. A $T=200$ y $k=5$, el RMSE es prácticamente idéntico pero el CRPS de Chronos sigue siendo 21% peor. Las distribuciones predictivas de Chronos son sistemáticamente más anchas de lo necesario, lo que resulta en sobrecobertura (cov\_95 entre $0.960$ y $0.996$) y Winkler inferior.

### 3. Chronos-joint supera a Chronos-ind solo marginalmente
En los experimentos donde ambos compiten (2.1 y 2.7), la diferencia en RMSE entre joint e ind. es $< 1\%$ a $T=50$ y a $T=200$. La API multivariada nativa de Chronos no produce mejoras dramáticas frente a $k$ llamadas univariadas independientes en estos DGPs. La ventaja del joint aumenta en 2.7 (cointegración) pero sigue siendo pequeña en RMSE ($< 0.5\%$).

### 4. El modelo VAR(1) misspecificado supera al VAR(2) correcto a $T=50$
En exp 2.3 con $A_2 = 0.1\,I$, VAR(1) domina a VAR(2) a $T=50$ porque la varianza de estimación de los parámetros adicionales supera el sesgo por omisión. Este resultado generaliza el principio de parsimonia: con muestras cortas, subparameterizar puede ser mejor que especificar correctamente. A $T=200$ los dos modelos son equivalentes.

### 5. El GARCH diagonal falla con muestras cortas
El VAR+GARCH-diag produce intervalos más estrechos y mal calibrados que el VAR estándar a $T=50$ (cov\_95$=0.749$ vs $0.908$). La estimación MLE de GARCH con $\sim\!26$ observaciones es insuficiente y sesga los parámetros de volatilidad hacia valores que subestiman la varianza incondicional. El modelo que debería mejorar la calibración la empeora. A $T=200$ la diferencia se reduce pero no desaparece.

### 6. La cointegración es el caso donde Chronos supera al clásico misspecificado
En exp 2.7, Chronos-joint supera a VAR(1) en RMSE a $T=200$ ($3.784$ vs $3.814$), siendo la única victoria de RMSE de Chronos en el bloque. Pero el VECM correctamente especificado sigue dominando en CRPS. Esto replica la lección de exp 1.4 (univariado): Chronos puede ser más robusto que un modelo clásico **misspecificado**, pero pierde frente al modelo **correctamente especificado**.

### 7. La no-estacionariedad cointegrada produce subcovertura universal
En exp 2.7, todos los modelos paramétricos tienen cov\_95 por debajo del nominal ($0.777$–$0.835$): el proceso $I(1)$ hace que la varianza del error de pronóstico crezca con $h$ de forma que los intervalos basados en varianza asintótica constante quedan sistemáticamente estrechos. Chronos produce intervalos más anchos ($\sim\!2\times$ los del VECM) y alcanza cobertura nominal, pero a costa de mayor ancho y Winkler comparable.

### 8. Implicación para la tesis
En el dominio multivariado, el patrón es consistente con el univariado: el modelo correctamente especificado domina en CRPS y calibración en todos los escenarios. La ventaja en RMSE puntual disminuye con $T$ y con $k$, llegando a empate cuando la dimensión es alta ($k=5$) o el parámetro a estimar es débil (VAR(2) con $A_2$ pequeño). La excepción es la misspecificación deliberada (VAR aplicado a datos cointegrados), donde Chronos supera al benchmark incorrecto pero no al modelo correcto. La brecha distribucional (CRPS) es más robusta y persistente que la brecha puntual (RMSE) — incluso cuando el RMSE converge, Chronos produce distribuciones predictivas más anchas de lo que justifica la incertidumbre real.
