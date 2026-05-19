"""
Unit tests for mectesis.metrics.crps_from_quantiles.

Run with:  python -m pytest tests/test_crps_from_quantiles.py -v
       or: python tests/test_crps_from_quantiles.py
"""

import numpy as np
from mectesis.metrics import crps_from_quantiles


def test_crps_from_quantiles_matches_crps_gaussian():
    """
    Con K=99 cuantiles uniformes sobre una N(mu, sigma), el estimador
    trapezoidal debe coincidir con crps_gaussian dentro de 0.5%.
    """
    from properscoring import crps_gaussian
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    n = 10
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y = rng.standard_normal(n)
    levels = np.linspace(0.01, 0.99, 99)
    q = norm.ppf(levels[None, :], loc=mu[:, None], scale=sigma[:, None])

    crps_quant = crps_from_quantiles(y, q, levels)
    crps_exact = crps_gaussian(y, mu, sigma)
    rel_err = np.abs(crps_quant - crps_exact) / crps_exact
    assert rel_err.max() < 0.005, f"max rel err = {rel_err.max():.4f}"


def test_crps_from_quantiles_shapes_univariate_and_multivariate():
    """Forma de salida: (horizon,) para univariado, (horizon, k) para multivariado."""
    rng = np.random.default_rng(1)
    horizon, k, K = 12, 3, 19
    levels = np.linspace(0.05, 0.95, K)

    # Univariado
    y_uni = rng.standard_normal(horizon)
    q_uni = np.sort(rng.standard_normal((horizon, K)), axis=1)
    out_uni = crps_from_quantiles(y_uni, q_uni, levels)
    assert out_uni.shape == (horizon,)
    assert np.all(np.isfinite(out_uni))
    assert np.all(out_uni >= 0)

    # Multivariado
    y_mv = rng.standard_normal((horizon, k))
    q_mv = np.sort(rng.standard_normal((horizon, k, K)), axis=2)
    out_mv = crps_from_quantiles(y_mv, q_mv, levels)
    assert out_mv.shape == (horizon, k)
    assert np.all(np.isfinite(out_mv))
    assert np.all(out_mv >= 0)


def test_crps_from_quantiles_zero_for_degenerate_distribution():
    """
    Si todos los cuantiles colapsan al valor observado y_true,
    el CRPS debe ser exactamente 0 (la distribucion es delta en y).
    """
    horizon, K = 8, 19
    levels = np.linspace(0.05, 0.95, K)
    y = np.full(horizon, 1.7)
    q = np.broadcast_to(y[:, None], (horizon, K)).copy()
    out = crps_from_quantiles(y, q, levels)
    assert np.allclose(out, 0.0, atol=1e-12)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)} tests OK")
