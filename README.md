# MECTESIS

**Time Series Foundation Models vs Classical Econometric Models**
*Monte Carlo Simulations for Master's Thesis in Econometrics*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Descripción

Este repositorio implementa simulaciones Monte Carlo para comparar el desempeño predictivo de **Time Series Foundation Models (TSFM)** vs **modelos econométricos clásicos** bajo diferentes Data Generating Processes (DGP).

**Contexto**: Tesis de Maestría en Econometría 2025-2026
**Autor**: David Guzzi

### Caso Simple Implementado

- **DGP**: Proceso AR(1)
- **Modelos**: ARIMA(1,0,0) vs Chronos-T5-Tiny
- **Métricas**: Sesgo, Varianza, MSE, RMSE por horizonte
- **Simulaciones**: 500 réplicas Monte Carlo

---

## Instalación Rápida

### 1. Clonar repositorio

```bash
git clone https://github.com/DavidGuzzi/MECTESIS.git
cd MECTESIS
```

### 2. Crear entorno virtual

```bash
# Crear venv
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate
```


## Estructura del Proyecto

```
MECTESIS/
├── mectesis/                 # Paquete principal (modular)
│   ├── dgp/                 # Data Generating Processes
│   ├── models/              # Modelos ARIMA + Chronos
│   ├── metrics/             # Métricas de evaluación
│   └── simulation/          # Motor Monte Carlo
├── experiments/configs/     # Configuraciones YAML
├── scripts/                 # Scripts ejecutables
├── prueba.py               # Script original (referencia)
├── notas.md                # Documentación completa
└── requirements.txt        # Dependencias
```