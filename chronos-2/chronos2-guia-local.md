# Chronos-2: Guía de uso local sin GPU

Análisis basado en los notebooks `chronos-2-quickstart.ipynb` y `deploy-chronos-to-amazon-sagemaker.ipynb`.

---

## ¿Qué es Chronos-2?

Chronos-2 es un modelo de forecasting de series temporales desarrollado por Amazon. Está basado en la arquitectura T5 y fue entrenado sobre un corpus masivo de series temporales reales y sintéticas. Sus principales ventajas sobre versiones anteriores:

- Ventana de contexto de **8192 tokens** (vs. 512 del Chronos original, 2048 del Chronos-Bolt)
- Soporte nativo para **covariables** (pasadas y futuras, numéricas y categóricas)
- **Cross-learning**: puede compartir información entre múltiples series al predecir
- Fine-tuning integrado (full y LoRA)

---

## ¿Necesito cuentas en servicios externos?

| Servicio | ¿Necesario para uso local? | Detalle |
|---|---|---|
| **HuggingFace** | No obligatorio | El modelo `amazon/chronos-2` es público. Se descarga automáticamente con `from_pretrained()` sin cuenta ni token. |
| **AWS** | No | Solo se necesita para el segundo notebook (despliegue en SageMaker). Para uso local, no se usa ningún servicio de AWS. |
| **GPU / CUDA** | No | Se puede correr en CPU con `device_map="cpu"`. La GPU acelera el proceso pero no es requisito. |
| **PyPI** | No (es gratuito) | `pip install` funciona sin cuenta. |

**Resumen: para correr Chronos-2 localmente, solo necesitás Python y pip.**

---

## Requisitos previos

- **Python**: 3.9 o superior recomendado
- **RAM**: mínimo 8 GB; se recomiendan 16 GB para el modelo base con contextos largos
- **Disco**: ~2-4 GB para descargar los pesos del modelo (se cachean localmente en `~/.cache/huggingface/hub`)
- **Conexión a internet**: solo la primera vez, para descargar el modelo

---

## Instalación paso a paso

### 1. Crear un entorno virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install "chronos-forecasting[extras]>=2.2" matplotlib
```

El extra `[extras]` incluye soporte para covariables. Sin él, solo está disponible el forecasting univariado básico.

---

## Uso mínimo en CPU

Este es el caso más simple: forecasting de una sola serie temporal sin GPU.

```python
import pandas as pd
from chronos import BaseChronosPipeline

# Cargar el modelo en CPU
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",   # cambiar a "cuda:0" si tenés GPU
)

# Armar un DataFrame en formato largo (id, timestamp, target)
df = pd.DataFrame({
    "item_id": ["serie_1"] * 100,
    "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
    "target": range(100),  # reemplazar con tus datos reales
})

# Predecir
predictions = pipeline.predict_df(
    df,
    prediction_length=14,       # cuántos pasos hacia adelante
    quantile_levels=[0.1, 0.5, 0.9],  # percentiles
)

print(predictions)
```

El resultado es un DataFrame con columnas `item_id`, `timestamp`, `mean`, `0.1`, `0.5`, `0.9`.

> **Nota sobre tiempos en CPU**: una predicción de 1000 series con contexto largo puede tardar varios minutos. Para exploración inicial, usar pocas series o reducir el historial pasado.

---

## Forecasting con covariables

Chronos-2 acepta variables externas que pueden mejorar la predicción.

```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)

# DataFrame con covariables pasadas (históricas) y futuras (conocidas)
df = pd.DataFrame({
    "item_id": ["energia"] * 200,
    "timestamp": pd.date_range("2023-01-01", periods=200, freq="h"),
    "target": ...,             # precio de energía (variable objetivo)
    "carga_red": ...,          # covariable pasada
    "generacion_solar": ...,   # covariable pasada Y futura (se conoce el pronóstico)
})

predictions = pipeline.predict_df(
    df,
    prediction_length=24,
    target_column="target",
    past_covariate_columns=["carga_red"],
    future_covariate_columns=["generacion_solar"],
    quantile_levels=[0.1, 0.5, 0.9],
)
```

Ejemplos de uso de covariables del notebook:
- **Energía**: target = precio; covariables = demanda de carga, generación renovable
- **Retail**: target = ventas; covariables = promociones, feriados, apertura del local

---

## Cross-learning

Cuando tenés muchas series relacionadas (ej. ventas de distintos productos), activar `cross_learning=True` permite que el modelo comparta información entre ellas al predecir:

```python
predictions = pipeline.predict_df(
    df,
    prediction_length=14,
    cross_learning=True,  # por defecto False
)
```

---

## Fine-tuning con LoRA (opcional)

Si tenés datos propios y querés adaptar el modelo:

```python
pipeline.train(
    train_data=df_train,
    validation_data=df_val,
    finetune_mode="lora",      # "full" para fine-tuning completo
    learning_rate=1e-5,        # recomendado para LoRA
    num_steps=100,
    batch_size=8,
)
```

El fine-tuning con LoRA es más liviano en memoria que el modo `"full"`.

---

## ¿Cuándo usar el notebook de SageMaker?

El segundo notebook (`deploy-chronos-to-amazon-sagemaker.ipynb`) es para despliegue en producción en AWS. Solo es relevante si:

- Necesitás un **endpoint HTTP** para consultas desde otras aplicaciones
- Querés procesar **millones de series** en batch
- Necesitás escalar automáticamente (Serverless)
- Tenés presupuesto para instancias de AWS (`ml.m5.xlarge` y superiores)

**Requiere**: cuenta de AWS, IAM role con permisos SageMaker, bucket S3.

Para experimentación, análisis y proyectos personales, el quickstart local es suficiente.

---

## Limitaciones sin GPU

| Aspecto | Con GPU | Sin GPU (CPU) |
|---|---|---|
| Velocidad | Rápida (ej. 18s para 1M filas) | Lenta (minutos para muchas series) |
| Batch size recomendado | 256+ | 8-32 |
| Modelos viables | `amazon/chronos-2` (todos los tamaños) | Usar versiones small/tiny si hay poca RAM |
| Fine-tuning | Viable | Muy lento; preferir LoRA con pocos pasos |

Variantes de modelo por tamaño (de menor a mayor RAM/tiempo):

| Modelo | Parámetros | Uso recomendado |
|---|---|---|
| `amazon/chronos-t5-tiny` | 8M | CPU con RAM limitada |
| `amazon/chronos-t5-mini` | 20M | CPU con RAM moderada |
| `amazon/chronos-t5-small` | 46M | CPU con buena RAM |
| `amazon/chronos-2-small` | 28M | CPU, variante Chronos-2 liviana |
| `amazon/chronos-t5-base` | 200M | CPU con 16 GB RAM |
| `amazon/chronos-bolt-base` | 205M | CPU con 16 GB RAM (más rápido) |
| `amazon/chronos-2` | 120M | CPU viable con float16 |
| `amazon/chronos-t5-large` | 710M | GPU recomendada; ver sección siguiente |

Para arrancar sin GPU se recomienda probar con `amazon/chronos-t5-small` y luego escalar.

---

## Usar el modelo más grande (chronos-t5-large, 710M) sin GPU

710M parámetros en float32 ocupa ~2.8 GB solo de pesos. Con activaciones y buffers intermedios, el proceso puede llegar a usar 8-16 GB de RAM dependiendo del tamaño del batch y el contexto. Para hacerlo viable en CPU:

### Truco clave: cargar en half precision (float16 o bfloat16)

Reduce el uso de RAM a la mitad (~1.4 GB de pesos):

```python
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cpu",
    torch_dtype=torch.bfloat16,  # mitad de RAM vs float32
)
```

> Usar `bfloat16` es preferible a `float16` en CPU: más estable numéricamente y soportado por PyTorch sin hardware especial.

### Reducir batch size al predecir

El parámetro `batch_size` controla cuántas series se procesan simultáneamente. Bajarlo reduce el pico de RAM:

```python
predictions = pipeline.predict_df(
    df,
    prediction_length=14,
    quantile_levels=[0.1, 0.5, 0.9],
    batch_size=4,   # default 256; bajar a 4-16 en CPU sin GPU
)
```

### Esquema de RAM aproximado

| Configuración | RAM aproximada |
|---|---|
| float32 + batch_size=32 | ~10-14 GB |
| bfloat16 + batch_size=16 | ~5-7 GB |
| bfloat16 + batch_size=4 | ~3-5 GB |

Con 8 GB de RAM disponibles, `bfloat16` + `batch_size=4` debería funcionar aunque será lento.

---

## Paso a paso: actualizar el venv y ejecutar el notebook

### 1. Activar el venv

Desde la raíz del proyecto (`MECTESIS/`):

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Cuando esté activo, el prompt muestra `(venv)` al inicio.

### 2. Actualizar chronos-forecasting a 2.2+

La versión instalada (2.1.0) no incluye `Chronos2Pipeline`. Actualizarla y agregar los extras de covariables:

```powershell
pip install "chronos-forecasting[extras]>=2.2"
```

> Esto también actualiza las dependencias necesarias (`autogluon.timeseries`, etc.).

### 3. Registrar el venv como kernel de Jupyter

Solo hace falta hacerlo una vez. Esto le dice a Jupyter que existe un kernel llamado `mectesis-venv`:

```powershell
python -m ipykernel install --user --name mectesis-venv --display-name "Python (venv MECTESIS)"
```

### 4. Abrir el notebook

```powershell
jupyter lab
```

Luego abrir `chronos-2/chronos2-test.ipynb`. En la esquina superior derecha de JupyterLab, verificar que el kernel seleccionado sea **"Python (venv MECTESIS)"**. Si no lo está, hacer clic ahí y seleccionarlo.

### 5. Ejecutar todas las celdas

`Kernel > Restart Kernel and Run All Cells`

La primera ejecución descarga los pesos del modelo desde HuggingFace (~500 MB) y los cachea en `~/.cache/huggingface/hub`. Las ejecuciones siguientes son instantáneas en la carga.

---

## Recursos útiles

- Repositorio oficial: [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- Modelo en HuggingFace: [huggingface.co/amazon/chronos-2](https://huggingface.co/amazon/chronos-2)
- Paper Chronos original: [arxiv.org/abs/2403.07815](https://arxiv.org/abs/2403.07815)
- Documentación SageMaker JumpStart: solo relevante si usás AWS
