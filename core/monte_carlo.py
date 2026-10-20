"""
core/monte_carlo.py
===================
Méthode de Monte-Carlo pour le pricing d'options européennes.
Utilise la solution exacte du mouvement brownien géométrique (MBG).

Principe : S_T = S₀ · exp[(r - σ²/2)·T + σ·√T·Z],  Z ~ N(0,1)
Estimateur : Ĉ = e^(-rT) · (1/M) · Σᵢ max(Sᵢ_T - K, 0)

Référence : Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
            Springer. Chapters 1-2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import time

from .black_scholes import black_scholes_price


# ── Dataclass résultat ──────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Résultat complet d'une simulation Monte-Carlo."""
    # Paramètres
    option_type: str
    S: float
    K: float
    r: float
    sigma: float
    T: float
    M: int

    # Résultats principaux
    price: float = 0.0
    bs_reference: float = 0.0
    absolute_error: float = 0.0
    relative_error_pct: float = 0.0

    # Statistiques
    mean_payoff: float = 0.0
    std_payoff: float = 0.0
    std_error: float = 0.0
    discount_factor: float = 1.0
    itm_pct: float = 0.0        # % simulations dans la monnaie

    # Intervalle de confiance
    ci_95_lo: float = 0.0
    ci_95_hi: float = 0.0
    ci_width: float = 0.0

    # Données pour graphiques
    terminal_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    payoffs: np.ndarray = field(default_factory=lambda: np.array([]))
    convergence_M: list = field(default_factory=list)
    convergence_prices: list = field(default_factory=list)
    convergence_ci_lo: list = field(default_factory=list)
    convergence_ci_hi: list = field(default_factory=list)

    # Meta
    elapsed_s: float = 0.0
    simulations_per_sec: float = 0.0
    seed_used: int = -1


# ── Fonctions principales ───────────────────────────────────────────────────

def simulate_terminal_prices(
    S: float,
    r: float,
    sigma: float,
    T: float,
    M: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simule M prix terminaux S_T selon la solution exacte du MBG.

    S_T = S · exp[(r - σ²/2)·T + σ·√T·Z],   Z ~ N(0,1)

    Parameters
    ----------
    S     : Prix initial
    r     : Taux sans risque
    sigma : Volatilité annualisée
    T     : Maturité (années)
    M     : Nombre de simulations
    seed  : Graine aléatoire (pour reproductibilité)

    Returns
    -------
    np.ndarray de forme (M,) : prix terminaux simulés
    """
    rng   = np.random.default_rng(seed)
    Z     = rng.standard_normal(M)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S * np.exp(drift + diffusion)


def confidence_interval(
    price: float, std_error: float, level: float = 0.95
) -> tuple[float, float]:
    """
    Calcule l'intervalle de confiance pour l'estimateur MC.

    IC_{level} = [price ± z · std_error]

    Parameters
    ----------
    price      : Prix estimé
    std_error  : Erreur standard de l'estimateur actualisé
    level      : Niveau de confiance (0.95 par défaut)

    Returns
    -------
    (lower, upper) : tuple des bornes de l'IC
    """
    from scipy.stats import norm as _norm
    z  = _norm.ppf((1 + level) / 2)
    return price - z * std_error, price + z * std_error


def convergence_curve(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: str = "call",
    M_max: int = 20000,
    n_points: int = 25,
    seed: int | None = 42,
) -> tuple[list, list, list, list]:
    """
    Calcule la courbe de convergence du prix MC en fonction de M.

    Returns
    -------
    (M_vals, prices, ci_lo, ci_hi) : listes de longueur n_points
    """
    M_vals    = np.unique(np.round(np.logspace(2, np.log10(M_max), n_points)).astype(int))
    prices    = []
    ci_lo_lst = []
    ci_hi_lst = []

    rng   = np.random.default_rng(seed)
    Z_all = rng.standard_normal(M_max)
    drift = (r - 0.5 * sigma ** 2) * T
    S_T_all = S * np.exp(drift + sigma * np.sqrt(T) * Z_all)
    df = np.exp(-r * T)

    for m in M_vals:
        S_T = S_T_all[:m]
        pf  = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
        mu  = pf.mean()
        se  = pf.std(ddof=1) / np.sqrt(m) * df
        p   = mu * df
        lo, hi = confidence_interval(p, se)
        prices.append(p)
        ci_lo_lst.append(lo)
        ci_hi_lst.append(hi)

    return M_vals.tolist(), prices, ci_lo_lst, ci_hi_lst


def payoff_distribution(
    S_T: np.ndarray, K: float, option_type: str = "call"
) -> np.ndarray:
    """Calcule les payoffs à maturité pour un vecteur de prix terminaux."""
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


# ── Fonction principale ─────────────────────────────────────────────────────

def monte_carlo_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    M: int = 10_000,
    option_type: Literal["call", "put"] = "call",
    seed: int | None = None,
    compute_convergence: bool = True,
    n_convergence_points: int = 20,
    store_samples: int = 2000,
) -> MCResult:
    """
    Pricing Monte-Carlo complet avec statistiques et courbe de convergence.

    Parameters
    ----------
    S                     : Prix initial
    K                     : Strike
    r                     : Taux sans risque
    sigma                 : Volatilité annualisée
    T                     : Maturité (années)
    M                     : Nombre de simulations (défaut 10 000)
    option_type           : 'call' ou 'put'
    seed                  : Graine aléatoire (None = aléatoire)
    compute_convergence   : Calcule la courbe de convergence (défaut True)
    n_convergence_points  : Nombre de points sur la courbe (défaut 20)
    store_samples         : Nombre d'échantillons stockés (défaut 2000)

    Returns
    -------
    MCResult : dataclass complet

    Examples
    --------
    >>> res = monte_carlo_price(100, 100, 0.05, 0.20, 1.0, M=50000)
    >>> abs(res.price - 10.45) < 0.5   # proche de BS
    True
    """
    t_start = time.perf_counter()
    opt     = option_type.lower().strip()
    df      = np.exp(-r * T)

    # ── Simulation vectorisée ──────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    Z     = rng.standard_normal(M)
    drift = (r - 0.5 * sigma ** 2) * T
    S_T   = S * np.exp(drift + sigma * np.sqrt(T) * Z)
    pf    = np.maximum(S_T - K, 0.0) if opt == "call" else np.maximum(K - S_T, 0.0)

    # ── Statistiques de l'estimateur ───────────────────────────────────────
    mean_pf = pf.mean()
    std_pf  = pf.std(ddof=1)
    se      = std_pf / np.sqrt(M) * df          # erreur standard actualisée
    price   = mean_pf * df
    lo, hi  = confidence_interval(price, se)
    itm_pct = (pf > 0).mean() * 100

    # ── Référence Black-Scholes ────────────────────────────────────────────
    bs_res  = black_scholes_price(S, K, r, sigma, T, opt)
    bs_ref  = bs_res.price
    abs_err = abs(price - bs_ref)
    rel_err = abs_err / bs_ref * 100 if bs_ref > 1e-8 else 0.0

    # ── Échantillons pour graphiques ───────────────────────────────────────
    n_store = min(store_samples, M)
    idx     = rng.integers(0, M, n_store)
    S_T_stored = S_T[idx]
    pf_stored  = pf[idx]

    elapsed = time.perf_counter() - t_start

    # ── Courbe de convergence ──────────────────────────────────────────────
    conv_M, conv_p, conv_lo, conv_hi = [], [], [], []
    if compute_convergence and M >= 500:
        conv_M, conv_p, conv_lo, conv_hi = convergence_curve(
            S, K, r, sigma, T, opt,
            M_max=M,
            n_points=n_convergence_points,
            seed=(seed or 42),
        )

    return MCResult(
        option_type         = opt,
        S=S, K=K, r=r, sigma=sigma, T=T, M=M,
        price               = price,
        bs_reference        = bs_ref,
        absolute_error      = abs_err,
        relative_error_pct  = rel_err,
        mean_payoff         = mean_pf,
        std_payoff          = std_pf,
        std_error           = se,
        discount_factor     = df,
        itm_pct             = itm_pct,
        ci_95_lo            = lo,
        ci_95_hi            = hi,
        ci_width            = hi - lo,
        terminal_prices     = S_T_stored,
        payoffs             = pf_stored,
        convergence_M       = conv_M,
        convergence_prices  = conv_p,
        convergence_ci_lo   = conv_lo,
        convergence_ci_hi   = conv_hi,
        elapsed_s           = elapsed,
        simulations_per_sec = M / elapsed if elapsed > 0 else 0,
        seed_used           = seed if seed is not None else -1,
    )


def ci_width_vs_M(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: str = "call",
    M_values: list[int] | None = None,
) -> dict:
    """
    Calcule la largeur de l'IC 95 % pour différentes valeurs de M.
    Démontre la convergence en O(1/√M).

    Returns
    -------
    dict avec 'M_values', 'ci_widths', 'theoretical_widths'
    """
    if M_values is None:
        M_values = [100, 250, 500, 1000, 2500, 5000, 10000]

    bs_ref = black_scholes_price(S, K, r, sigma, T, option_type).price
    widths, theo = [], []
    ref_width = None

    for m in M_values:
        res = monte_carlo_price(S, K, r, sigma, T, m, option_type,
                                compute_convergence=False)
        widths.append(res.ci_width)
        if ref_width is None:
            ref_width = res.ci_width
        theo.append(ref_width * np.sqrt(M_values[0] / m))

    return {"M_values": M_values, "ci_widths": widths,
            "theoretical_widths": theo, "bs_reference": bs_ref}


# ── Point d'entrée de test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Monte-Carlo — Test de référence")
    print("=" * 60)
    res = monte_carlo_price(100, 100, 0.05, 0.20, 1.0,
                            M=50_000, option_type="call", seed=42)
    print(f"\n  Prix MC       : {res.price:.6f}")
    print(f"  Référence BS  : {res.bs_reference:.6f}")
    print(f"  Erreur abs.   : {res.absolute_error:.6f}")
    print(f"  Erreur rel.   : {res.relative_error_pct:.4f} %")
    print(f"  IC 95%        : [{res.ci_95_lo:.4f} ; {res.ci_95_hi:.4f}]")
    print(f"  Largeur IC    : {res.ci_width:.4f}")
    print(f"  % ITM         : {res.itm_pct:.2f} %")
    print(f"  Temps         : {res.elapsed_s*1000:.2f} ms")
    print(f"  Sim/sec       : {res.simulations_per_sec:,.0f}")
    print("=" * 60)
