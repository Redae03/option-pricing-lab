"""
core/black_scholes.py
=====================
Formule analytique de Black-Scholes-Merton (1973).
Calcul exact du prix d'options européennes Call et Put,
ainsi que des lettres grecques complètes (Δ, Γ, Θ, ν, ρ).

Référence : Black, F. & Scholes, M. (1973).
            The Pricing of Options and Corporate Liabilities.
            Journal of Political Economy, 81(3), 637–654.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Literal
import time


# ── Dataclass résultat ──────────────────────────────────────────────────────

@dataclass
class BSResult:
    """Résultat complet d'un pricing Black-Scholes."""
    # Paramètres
    option_type: str
    S: float
    K: float
    r: float
    sigma: float
    T: float
    q: float = 0.0          # dividende continu (extension Merton)

    # Valeurs intermédiaires
    d1: float = 0.0
    d2: float = 0.0
    N_d1: float = 0.0
    N_d2: float = 0.0
    discount_factor: float = 1.0

    # Prix
    price: float = 0.0
    call_price: float = 0.0
    put_price: float = 0.0

    # Grecques
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0       # par jour calendaire
    vega: float = 0.0        # pour +1% de sigma
    rho: float = 0.0         # pour +1% de r

    # Meta
    elapsed_s: float = 0.0
    moneyness: str = ""
    intrinsic_value: float = 0.0
    time_value: float = 0.0


# ── Fonctions auxiliaires ───────────────────────────────────────────────────

def _check_params(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """Valide les paramètres d'entrée."""
    if S <= 0:
        raise ValueError(f"Le prix initial S doit être strictement positif (reçu : {S})")
    if K <= 0:
        raise ValueError(f"Le strike K doit être strictement positif (reçu : {K})")
    if sigma <= 0:
        raise ValueError(f"La volatilité sigma doit être strictement positive (reçu : {sigma})")
    if T <= 0:
        raise ValueError(f"La maturité T doit être strictement positive (reçu : {T})")
    if r < 0:
        raise ValueError(f"Le taux sans risque r doit être positif ou nul (reçu : {r})")


def calculate_d1_d2(
    S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0
) -> tuple[float, float]:
    """
    Calcule d₁ et d₂ de la formule de Black-Scholes.

    d₁ = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    Parameters
    ----------
    S     : Prix initial du sous-jacent
    K     : Prix d'exercice (strike)
    r     : Taux sans risque continu
    sigma : Volatilité annualisée
    T     : Maturité en années
    q     : Taux de dividende continu (0 par défaut)

    Returns
    -------
    (d1, d2) : tuple de flottants
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def call_price(
    S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0
) -> float:
    """
    Prix d'un Call européen par la formule de Black-Scholes.

    C = S·e^(-qT)·N(d₁) - K·e^(-rT)·N(d₂)
    """
    _check_params(S, K, r, sigma, T)
    d1, d2 = calculate_d1_d2(S, K, r, sigma, T, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(
    S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0
) -> float:
    """
    Prix d'un Put européen par la formule de Black-Scholes.

    P = K·e^(-rT)·N(-d₂) - S·e^(-qT)·N(-d₁)
    """
    _check_params(S, K, r, sigma, T)
    d1, d2 = calculate_d1_d2(S, K, r, sigma, T, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def payoff_call(S_T: float | np.ndarray, K: float) -> float | np.ndarray:
    """Payoff du Call à maturité : max(S_T - K, 0)."""
    return np.maximum(S_T - K, 0.0)


def payoff_put(S_T: float | np.ndarray, K: float) -> float | np.ndarray:
    """Payoff du Put à maturité : max(K - S_T, 0)."""
    return np.maximum(K - S_T, 0.0)


# ── Fonction principale ─────────────────────────────────────────────────────

def black_scholes_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> BSResult:
    """
    Calcul complet Black-Scholes : prix + grecques + métadonnées.

    Parameters
    ----------
    S           : Prix initial du sous-jacent
    K           : Strike (prix d'exercice)
    r           : Taux sans risque (continu, annualisé)
    sigma       : Volatilité (annualisée)
    T           : Maturité en années
    option_type : 'call' ou 'put'
    q           : Dividende continu (extension Merton, 0 par défaut)

    Returns
    -------
    BSResult : dataclass contenant prix, grecques et méta-informations.

    Examples
    --------
    >>> res = black_scholes_price(100, 100, 0.05, 0.20, 1.0, 'call')
    >>> round(res.price, 4)
    10.4506
    """
    t_start = time.perf_counter()
    _check_params(S, K, r, sigma, T)

    opt = option_type.lower().strip()
    if opt not in ("call", "put"):
        raise ValueError(f"option_type doit être 'call' ou 'put' (reçu : {option_type})")

    sqrt_T   = np.sqrt(T)
    d1, d2   = calculate_d1_d2(S, K, r, sigma, T, q)
    df       = np.exp(-r * T)           # facteur d'actualisation
    dq       = np.exp(-q * T)           # facteur dividende

    N_d1     = norm.cdf(d1)
    N_d2     = norm.cdf(d2)
    N_md1    = norm.cdf(-d1)
    N_md2    = norm.cdf(-d2)
    n_d1     = norm.pdf(d1)             # densité en d1

    c_price  = S * dq * N_d1 - K * df * N_d2
    p_price  = K * df * N_md2 - S * dq * N_md1
    price    = c_price if opt == "call" else p_price

    # ── Delta ─────────────────────────────────────────────────────────────
    delta = dq * N_d1 if opt == "call" else -dq * N_md1

    # ── Gamma (identique Call / Put) ──────────────────────────────────────
    gamma = n_d1 * dq / (S * sigma * sqrt_T)

    # ── Theta (par jour calendaire) ───────────────────────────────────────
    common = -S * n_d1 * sigma * dq / (2 * sqrt_T)
    if opt == "call":
        theta = (common - r * K * df * N_d2 + q * S * dq * N_d1) / 365
    else:
        theta = (common + r * K * df * N_md2 - q * S * dq * N_md1) / 365

    # ── Véga (pour +1% de sigma) ──────────────────────────────────────────
    vega = S * dq * n_d1 * sqrt_T * 0.01

    # ── Rhô (pour +1% de r) ───────────────────────────────────────────────
    if opt == "call":
        rho = K * T * df * N_d2 * 0.01
    else:
        rho = -K * T * df * N_md2 * 0.01

    # ── Moneyness ─────────────────────────────────────────────────────────
    ratio = S / K
    if ratio > 1.02:
        moneyness = "ITM (dans la monnaie)"
    elif ratio < 0.98:
        moneyness = "OTM (hors de la monnaie)"
    else:
        moneyness = "ATM (à la monnaie)"

    intrinsic = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    time_val  = price - intrinsic

    elapsed = time.perf_counter() - t_start

    return BSResult(
        option_type     = opt,
        S=S, K=K, r=r, sigma=sigma, T=T, q=q,
        d1=d1, d2=d2,
        N_d1=N_d1, N_d2=N_d2,
        discount_factor = df,
        price           = price,
        call_price      = c_price,
        put_price       = p_price,
        delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
        elapsed_s       = elapsed,
        moneyness       = moneyness,
        intrinsic_value = intrinsic,
        time_value      = time_val,
    )


# ── Fonctions utilitaires supplémentaires ───────────────────────────────────

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Calcule la volatilité implicite par la méthode de Newton-Raphson.

    Parameters
    ----------
    market_price : Prix de marché observé de l'option
    tol          : Tolérance de convergence (défaut 1e-6)
    max_iter     : Nombre maximum d'itérations (défaut 200)

    Returns
    -------
    sigma_impl : Volatilité implicite (float), ou NaN si non convergé.
    """
    sigma = 0.20    # valeur initiale
    for _ in range(max_iter):
        res   = black_scholes_price(S, K, r, sigma, T, option_type)
        diff  = res.price - market_price
        # Véga numérique en unités de sigma (pas de %)
        vega_raw = res.vega * 100
        if abs(vega_raw) < 1e-10:
            return float("nan")
        sigma -= diff / vega_raw
        sigma  = max(sigma, 1e-6)
        if abs(diff) < tol:
            return sigma
    return float("nan")


def parity_check(
    S: float, K: float, r: float, sigma: float, T: float
) -> dict:
    """
    Vérifie la parité Call-Put : C - P = S·e^(-qT) - K·e^(-rT).

    Returns
    -------
    dict avec 'lhs', 'rhs', 'error', 'verified'
    """
    res_c = black_scholes_price(S, K, r, sigma, T, "call")
    res_p = black_scholes_price(S, K, r, sigma, T, "put")
    lhs   = res_c.price - res_p.price
    rhs   = S - K * np.exp(-r * T)
    error = abs(lhs - rhs)
    return {
        "call_price"  : res_c.price,
        "put_price"   : res_p.price,
        "lhs_C_minus_P": lhs,
        "rhs_S_minus_PV_K": rhs,
        "absolute_error": error,
        "verified"    : error < 1e-8,
    }


def sensitivity_table(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: str = "call",
) -> list[dict]:
    """
    Génère un tableau de sensibilité ±10 % sur chaque paramètre.

    Returns
    -------
    list[dict] avec colonnes: scenario, param, base_val, lo_val, hi_val,
                              price_base, price_lo, price_hi, delta_lo, delta_hi
    """
    base = black_scholes_price(S, K, r, sigma, T, option_type)
    scenarios = [
        ("S₀ ±10%",    "S",     S,     S*0.9,         S*1.1),
        ("σ ±5pt",     "sigma", sigma, max(sigma-0.05, 0.01), sigma+0.05),
        ("T ×½ / ×2",  "T",     T,     T/2,           T*2),
        ("r ±2pt",     "r",     r,     max(r-0.02,0),  r+0.02),
    ]
    rows = []
    for label, param, base_val, lo_val, hi_val in scenarios:
        kwargs = dict(S=S, K=K, r=r, sigma=sigma, T=T, option_type=option_type)
        kwargs_lo = {**kwargs, param: lo_val}
        kwargs_hi = {**kwargs, param: hi_val}
        p_lo = black_scholes_price(**kwargs_lo).price
        p_hi = black_scholes_price(**kwargs_hi).price
        rows.append({
            "scenario"    : label,
            "param"       : param,
            "base_val"    : base_val,
            "lo_val"      : lo_val,
            "hi_val"      : hi_val,
            "price_base"  : base.price,
            "price_lo"    : p_lo,
            "price_hi"    : p_hi,
            "delta_lo"    : p_lo - base.price,
            "delta_hi"    : p_hi - base.price,
        })
    return rows


# ── Point d'entrée de test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Black-Scholes — Test de référence")
    print("=" * 60)
    params = dict(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
    for otype in ("call", "put"):
        res = black_scholes_price(**params, option_type=otype)
        print(f"\n  {otype.upper()} européen")
        print(f"  {'Prix':<20} : {res.price:.6f}")
        print(f"  {'d1':<20} : {res.d1:.6f}")
        print(f"  {'d2':<20} : {res.d2:.6f}")
        print(f"  {'Delta':<20} : {res.delta:.6f}")
        print(f"  {'Gamma':<20} : {res.gamma:.8f}")
        print(f"  {'Theta/jour':<20} : {res.theta:.6f}")
        print(f"  {'Vega (+1% sig)':<20} : {res.vega:.6f}")
        print(f"  {'Rho (+1% r)':<20} : {res.rho:.6f}")
        print(f"  {'Moneyness':<20} : {res.moneyness}")
        print(f"  {'Temps':<20} : {res.elapsed_s*1000:.4f} ms")

    print("\n  Parité Call-Put :")
    parity = parity_check(**params)
    print(f"  C - P = {parity['lhs_C_minus_P']:.6f}")
    print(f"  S - Ke^(-rT) = {parity['rhs_S_minus_PV_K']:.6f}")
    print(f"  Vérifiée : {parity['verified']}")
    print("=" * 60)
