# Guía de métricas de evaluación

Las métricas se calculan sobre `R` simulaciones Monte Carlo.  
Sea $e_{r,h} = y_{T+h}^{(r)} - \hat{y}_{T+h}^{(r)}$ el error de pronóstico en el horizonte $h$ de la réplica $r$.

---

## Métricas de error puntual

| Métrica | Definición | Referencia |
|---------|-----------|-----------|
| **bias** | $\frac{1}{R}\sum_r e_{r,h}$ | 0 — sesgo nulo |
| **variance** | $\frac{1}{R}\sum_r (e_{r,h} - \overline{e}_h)^2$ | menor es mejor |
| **mse** | $\frac{1}{R}\sum_r e_{r,h}^2$ | menor es mejor |
| **rmse** | $\sqrt{\text{MSE}}$ | en unidades de $Y_t$ |
| **mae** | $\frac{1}{R}\sum_r |e_{r,h}|$ | en unidades de $Y_t$ |

**Descomposición**: $\text{MSE} = \text{Bias}^2 + \text{Varianza}$  
Un modelo puede tener bajo MSE con bias nulo y alta varianza, o con bias no nulo y baja varianza — la descomposición distingue ambos casos.

**RMSE vs MAE**: RMSE penaliza más los errores grandes (cuadráticos). Si la distribución de errores tiene colas pesadas, MAE es más representativo del error típico.

---

## Métricas de intervalos de predicción

| Métrica | Definición | Referencia |
|---------|-----------|-----------|
| **cov_80** | Fracción de réplicas en que $y_{T+h}$ cae dentro del IP al 80% | 0.80 |
| **cov_95** | Ídem para IP al 95% | 0.95 |
| **width_80** | Amplitud promedio del IP al 80%: $\frac{1}{R}\sum_r (u_{r,h} - l_{r,h})$ | menor es mejor |
| **width_95** | Ídem para IP al 95% | menor es mejor |

**Calibración**: un modelo está bien calibrado si `cov_80 ≈ 0.80` y `cov_95 ≈ 0.95`.  
- `cov < nominal` → intervalos demasiado estrechos (sobreconfianza)  
- `cov > nominal` → intervalos demasiado anchos (subconfianza)

**Trade-off cobertura / amplitud**: cobertura alta siempre se puede lograr ensanchando el intervalo. La comparación útil es entre modelos con cobertura similar: a igual cobertura, gana el de menor amplitud.

---

## Estructura de la tabla

```
T       — tamaño muestral de entrenamiento (obs previas al horizonte)
R       — número de réplicas Monte Carlo
Modelo  — modelo evaluado
Bloque  — promedio sobre h=1–12 (corto plazo) o h=13–24 (mediano plazo)
```

Los valores en la tabla son **promedios sobre todos los horizontes del bloque** y sobre las `R` réplicas.

---

## Métricas probabilísticas

| Métrica | Definición | Referencia |
|---------|-----------|-----------|
| **crps** | $\frac{1}{R}\sum_r \text{CRPS}(\hat{F}_{r,h},\, y_{T+h}^{(r)})$ | menor es mejor; 0 = perfecto |

**CRPS (Continuous Ranked Probability Score):** puntaje propio estricto que evalúa la distribución predictiva completa, no solo la media. Se calcula de dos formas según el modelo:

- **ETS:** `crps_ensemble(y_true, sims)` donde `sims` es la matriz de 500 trayectorias simuladas. El CRPS ensemble suma las diferencias absolutas entre la distribución empírica y el dato real.
- **Theta:** `crps_gaussian(y_true, mu, sigma)` derivando `sigma` del intervalo 95%: `sigma = (hi_95 − lo_95) / (2 × norm.ppf(0.975))`. Asume distribución predictiva gaussiana.
- **Chronos:** `crps_ensemble` con los cuantiles provistos por el modelo.
- **SeasonalNaive:** no disponible (`supports_crps=False`); la columna aparece como NaN.

Un CRPS bajo indica que la distribución predictiva es a la vez precisa (concentrada en el valor real) y bien calibrada. A diferencia del RMSE, penaliza tanto el error de nivel como la sobreconfianza.

---

## Winkler Score (Interval Score)

| Métrica | Definición | Referencia |
|---------|-----------|-----------|
| **winkler_80** | Puntuación Winkler para IP al 80% | menor es mejor |
| **winkler_95** | Puntuación Winkler para IP al 95% | menor es mejor |

Para un intervalo $(l_{r,h},\, u_{r,h})$ a nivel $(1-\alpha)$, la puntuación de Winkler es:

$$W_\alpha = (u - l) + \frac{2}{\alpha}\max(l - y,\, 0) + \frac{2}{\alpha}\max(y - u,\, 0)$$

- El primer término penaliza la **amplitud** del intervalo.
- Los dos últimos penalizan **las salidas** (misses), multiplicadas por $2/\alpha$.
- Para $\alpha=0.20$ (IP 80%): la penalización por salida es ×10; para $\alpha=0.05$ (IP 95%): ×40.
- El score unifica cobertura y amplitud: un modelo que ensancha artificialmente sus intervalos para igualar la cobertura de otro tendrá mayor winkler. Gana el modelo de menor winkler dado igual o mejor cobertura.

**Nota:** `winkler_80` y `winkler_95` son NaN para SeasonalNaive (no produce intervalos de predicción).
