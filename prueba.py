# ======================================================================
#   SIMULACIONES MONTE CARLO PARA TESIS
#   DGP: AR(1)
#   MODELOS: ARIMA(1,0,0) vs TSFM (amazon/chronos-bolt-tiny)
#   MÉTRICAS: Sesgo, Varianza, MSE, RMSE por horizonte + agregados
# ======================================================================

# ------------------------- IMPORTS ------------------------------------
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from chronos import Chronos2Pipeline

rng = np.random.default_rng(12345)   # reproducibilidad total


# ======================================================================
# 1. DGP: AR(1)
# ======================================================================

def simulate_ar1(T: int, phi: float, mu: float = 0.0, sigma: float = 1.0,
                 burn_in: int = 200) -> np.ndarray:
    """
    Simula un proceso AR(1): y_t = mu + phi (y_{t-1} - mu) + e_t.
    Descarta burn_in inicial para asegurar estacionariedad efectiva.
    """
    total_T = T + burn_in
    y = np.empty(total_T)
    y[0] = mu

    for t in range(1, total_T):
        eps = rng.normal(0.0, sigma)
        y[t] = mu + phi * (y[t-1] - mu) + eps

    return y[burn_in:]


# ======================================================================
# 2. MODELO CLÁSICO: ARIMA(1,0,0)
# ======================================================================

def forecast_arima(y_train: np.ndarray, horizon: int) -> np.ndarray:
    """
    Ajusta ARIMA(1,0,0) y produce pronóstico multi-step.
    """
    model = ARIMA(y_train, order=(1, 0, 0))
    res = model.fit()
    fcst = res.get_forecast(steps=horizon)
    return np.array(fcst.predicted_mean)


# ======================================================================
# 3. MODELO TSFM LIVIANO: amazon/chronos-bolt-tiny
# ======================================================================

# Cargamos el modelo más liviano disponible
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    device_map="cpu"     # corre perfectamente sin GPU
)

def forecast_tsfm_chronos(y_train: np.ndarray, horizon: int) -> np.ndarray:
    """
    Pronóstico multi-step usando el TSFM más liviano:
    amazon/chronos-bolt-tiny.
    """
    # Chronos requiere formato tabular con timestamp
    df = pd.DataFrame({
        "item_id": ["series_1"] * len(y_train),
        "timestamp": pd.date_range("2000-01-01", periods=len(y_train), freq="D"),
        "target": y_train,
    })

    fcst = pipeline.predict(
        df,
        prediction_length=horizon,
        quantile_levels=[0.5]      # usamos la mediana como pronóstico puntual
    )

    # Filtramos horizonte y ordenamos por fecha
    fcst_h = fcst[fcst["quantile"] == 0.5].sort_values("timestamp")
    return fcst_h["mean"].to_numpy()


# ======================================================================
# 4. UNA SIMULACIÓN INDIVIDUAL
# ======================================================================

def one_simulation(T: int,
                   horizon: int,
                   phi: float,
                   mu: float = 0.0,
                   sigma: float = 1.0) -> dict:
    """
    Genera una serie AR(1), separa train/test, y genera pronósticos
    con ARIMA y con TSFM.
    Devuelve errores (verdad - predicción).
    """
    # Simular serie
    y = simulate_ar1(T=T, phi=phi, mu=mu, sigma=sigma)

    # División train / test
    T_train = T - horizon
    y_train = y[:T_train]
    y_test = y[T_train:]
    assert len(y_test) == horizon

    # Pronósticos
    yhat_arima = forecast_arima(y_train, horizon)
    yhat_tsfm = forecast_tsfm_chronos(y_train, horizon)

    # Errores
    errors_arima = y_test - yhat_arima
    errors_tsfm = y_test - yhat_tsfm

    return {
        "errors_arima": errors_arima,
        "errors_tsfm": errors_tsfm
    }


# ======================================================================
# 5. BUCLE MONTE CARLO + ESTADÍSTICAS
# ======================================================================

def monte_carlo(T: int,
                horizon: int,
                phi: float,
                mu: float,
                sigma: float,
                n_sim: int = 1000) -> dict:
    """
    Ejecuta n_sim simulaciones Monte Carlo.
    Calcula sesgo, varianza, MSE, RMSE por horizonte y medidas agregadas.
    """

    # Matrices de errores: (n_sim, horizon)
    e_arima = np.empty((n_sim, horizon))
    e_tsfm = np.empty((n_sim, horizon))

    for s in range(n_sim):
        res = one_simulation(T, horizon, phi, mu, sigma)
        e_arima[s] = res["errors_arima"]
        e_tsfm[s] = res["errors_tsfm"]

    # Función para métricas
    def summarize(errors_mat):
        bias = errors_mat.mean(axis=0)
        var = errors_mat.var(axis=0, ddof=1)
        mse = (errors_mat**2).mean(axis=0)
        rmse = np.sqrt(mse)

        df = pd.DataFrame({
            "horizon": np.arange(1, horizon + 1),
            "bias": bias,
            "variance": var,
            "mse": mse,
            "rmse": rmse
        })

        # Métricas agregadas (fila final)
        df_agg = pd.DataFrame({
            "horizon": ["avg_all"],
            "bias": [bias.mean()],
            "variance": [var.mean()],
            "mse": [mse.mean()],
            "rmse": [rmse.mean()]
        })

        return pd.concat([df, df_agg], ignore_index=True)

    return {
        "arima": summarize(e_arima),
        "tsfm": summarize(e_tsfm)
    }


# ======================================================================
# 6. EJEMPLO DE CORRIDA
# ======================================================================

if __name__ == "__main__":
    T = 200          # longitud de cada serie
    H = 12           # horizonte a pronosticar
    phi = 0.7        # persistencia
    mu = 0.0
    sigma = 1.0
    N = 500          # número de simulaciones Monte Carlo

    results = monte_carlo(T=T,
                          horizon=H,
                          phi=phi,
                          mu=mu,
                          sigma=sigma,
                          n_sim=N)

    print("\n=== RESULTADOS ARIMA ===")
    print(results["arima"])

    print("\n=== RESULTADOS TSFM (chronos-bolt-tiny) ===")
    print(results["tsfm"])