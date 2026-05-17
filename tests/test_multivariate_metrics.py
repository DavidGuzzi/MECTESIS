"""
Unit tests for mectesis.metrics.multivariate.

Run with:  python -m pytest tests/test_multivariate_metrics.py -v
       or: python tests/test_multivariate_metrics.py
"""

import numpy as np
from mectesis.metrics.multivariate import trace_msfe, avg_marginal_crps


def test_trace_msfe_zero_errors():
    """Errores cero -> trace MSFE cero en todos los horizontes."""
    errors = np.zeros((100, 24, 3))
    out = trace_msfe(errors)
    assert out.shape == (24,)
    assert np.allclose(out, 0.0)


def test_trace_msfe_equals_sum_mse_marginal():
    """trace_msfe(h) debe igualar sum_i MSE_i(h) calculado por separado."""
    rng = np.random.default_rng(42)
    n_sim, horizon, k = 500, 12, 4
    errors = rng.standard_normal((n_sim, horizon, k))

    tm = trace_msfe(errors)
    # Reference: sum of per-variable MSE
    mse_per_var = np.mean(errors ** 2, axis=0)   # (horizon, k)
    sum_mse = np.sum(mse_per_var, axis=1)         # (horizon,)

    assert tm.shape == sum_mse.shape == (horizon,)
    assert np.allclose(tm, sum_mse, atol=1e-12), \
        f"trace_msfe != sum(MSE_i): max diff = {np.max(np.abs(tm - sum_mse))}"


def test_trace_msfe_scales_with_error_magnitude():
    """trace_msfe debe escalar cuadraticamente con la escala del error."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((100, 12, 2))
    tm_base = trace_msfe(base)
    tm_2x   = trace_msfe(2 * base)
    assert np.allclose(tm_2x, 4 * tm_base, atol=1e-10)


def test_trace_msfe_shape_validation():
    """trace_msfe debe rechazar input con dimensiones incorrectas."""
    try:
        trace_msfe(np.zeros((10, 5)))   # 2D
    except ValueError:
        pass
    else:
        raise AssertionError("Should have raised ValueError for 2D input")


def test_avg_marginal_crps_constant_across_vars():
    """Si CRPS es identico en todas las variables, el promedio coincide."""
    rng = np.random.default_rng(7)
    n_sim, horizon, k = 200, 6, 4
    # Mismo valor en todas las vars para cada (sim, h)
    crps_one_var = rng.uniform(0.1, 1.0, size=(n_sim, horizon, 1))
    crps_repeated = np.broadcast_to(crps_one_var, (n_sim, horizon, k)).copy()

    acrps = avg_marginal_crps(crps_repeated)
    # Tiene que coincidir con el promedio MC del CRPS univariado
    crps_uni_avg = crps_one_var[:, :, 0].mean(axis=0)
    assert acrps.shape == (horizon,)
    assert np.allclose(acrps, crps_uni_avg, atol=1e-12)


def test_avg_marginal_crps_is_linear_combination():
    """Promedio sobre vars + promedio sobre sims = promedio total."""
    rng = np.random.default_rng(11)
    n_sim, horizon, k = 100, 8, 3
    crps = rng.uniform(0, 2, size=(n_sim, horizon, k))

    acrps = avg_marginal_crps(crps)
    # Alternativa: promediar sobre sims primero, despues sobre vars
    ref = np.mean(np.mean(crps, axis=0), axis=1)
    assert np.allclose(acrps, ref, atol=1e-12)


def test_avg_marginal_crps_shape_validation():
    """avg_marginal_crps debe rechazar input no-3D."""
    try:
        avg_marginal_crps(np.zeros((10, 5)))
    except ValueError:
        pass
    else:
        raise AssertionError("Should have raised ValueError for 2D input")


if __name__ == "__main__":
    # Ejecutable directo sin pytest
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)} tests OK")
