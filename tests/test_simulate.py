"""simulate() must match the closed-form answer for two correlated Normals.  Run: pytest"""
import math

from simulate import simulate


def test_matches_closed_form_with_correlation():
    mu_h, s_h, mu_a, s_a, rho = 115.0, 11.6, 110.0, 11.4, 0.28
    sd = math.sqrt(s_h**2 + s_a**2 - 2 * rho * s_h * s_a)
    expected = 0.5 * (1 + math.erf((mu_h - mu_a) / (sd * math.sqrt(2))))
    r = simulate(mu_h, s_h, mu_a, s_a, n=200_000, rho=rho, seed=1)
    assert abs(r["home_win_prob"] - expected) < 0.005
    assert abs(r["margin_mean"] - 5.0) < 0.1
    # 5th-95th percentile of the margin is +/- 1.645 sd
    assert abs((r["margin_pct"][95] - r["margin_pct"][5]) - 2 * 1.645 * sd) < 0.3
