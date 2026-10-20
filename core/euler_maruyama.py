"""
core/euler_maruyama.py
======================
Discrétisation d'Euler-Maruyama pour le pricing d'options européennes.

L'EDS de Black-Scholes sous la mesure risque-neutre Q :
    dS_t = r·S_t·dt + σ·S_t·dW_t

Schéma d'Euler-Maruyama (discrétisation en N pas de taille Δt = T/N) :
    S_{t+Δt} = S_t + r·S_t·Δt + σ·S_t·√Δt·Z_j,    Z_j ~ N(0,1) i.i.d.

Biais de discrétisation : O(Δt) = O(1/N)  (fort, en norme L²)

Références :
    Jourdain, B. (2012). Probabilités et Statistique — Discrétisation des EDS.
    Kloeden, P.E. & Platen, E. (1992). Numerical Solution of Stochastic
    Differential Equations. Springer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import time

from .black_scholes import black_scholes_price


# ── Dataclass résultat ──────────────────────────────────────────────────────

@dataclass
class EMResult:
    """Résultat complet d'une simulation Euler-Maruyama."""
    # Paramètres
    option_type: str
    S: float
    K: float
    r: float
    sigma: float
    T: float
    M: int
    N: int
    dt: float = 0.0

    # Résultats principaux
    price: float = 0.0
    bs_reference: float = 0.0
    bias: float = 0.0                   # biais = P_euler - P_BS
    absolute_bias: float = 0.0
    relative_bias_pct: float = 0.0

    # Statistiques
    mean_payoff: float = 0.0
    std_payoff: float = 0.0
    discount_factor: float = 1.0
    itm_pct: float = 0.0

    # Données pour graphiques
    terminal_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    payoffs: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_paths: np.ndarray = field(default_factory=lambda: np.array([]))   # shape (n_paths, N+1)
    time_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    # Sensibilité à N
    N_values: list = field(default_factory=list)
    prices_vs_N: list = field(default_factory=list)
    biases_vs_N: list = field(default_factory=list)

    # Meta
    elapsed_s: float = 0.0
    simulations_per_sec: float = 0.0


# ── Fonctions principales ───────────────────────────────────────────────────

def simulate_euler_paths(
    S: float,
    r: float,
    sigma: float,
    T: float,
    M: int,
    N: int,
    seed: int | None = None,
    n_store_paths: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simule M trajectoires discrètes du sous-jacent par le schéma d'Euler-Maruyama.

    Schéma : S_{j+1} = S_j + r·S_j·Δt + σ·S_j·√Δt·Z_j

    Parameters
    ----------
    S              : Prix initial
    r              : Taux sans risque
    sigma          : Volatilité
    T              : Maturité
    M              : Nombre de trajectoires
    N              : Nombre de pas de discrétisation
    seed           : Graine aléatoire
    n_store_paths  : Nombre de trajectoires complètes à conserver

    Returns
    -------
    (S_terminal, sample_paths)
        S_terminal   : np.ndarray (M,)     — prix terminaux S_T
        sample_paths : np.ndarray (n_store_paths, N+1) — trajectoires complètes
    """
    rng    = np.random.default_rng(seed)
    dt     = T / N
    sqrt_dt = np.sqrt(dt)

    # Pré-allouer toutes les incréments bruits en une seule fois
    Z = rng.standard_normal((M, N))     # shape (M, N)

    # Trajectoire vectorisée
    S_curr = np.full(M, float(S))
    paths_store = np.zeros((min(n_store_paths, M), N + 1))
    paths_store[:, 0] = S

    for j in range(N):
        dS       = r * S_curr * dt + sigma * S_curr * sqrt_dt * Z[:, j]
        S_curr   = np.maximum(S_curr + dS, 0.0)     # absorbe à 0
        if j < N - 1:
            paths_store[:, j + 1] = S_curr[:n_store_paths]
    paths_store[:, N] = S_curr[:n_store_paths]

    return S_curr, paths_store


def euler_maruyama_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    M: int = 10_000,
    N: int = 252,
    option_type: Literal["call", "put"] = "call",
    seed: int | None = None,
    n_store_paths: int = 8,
) -> float:
    """
    Prix estimé par Euler-Maruyama (résultat scalaire uniquement).
    Utiliser `euler_maruyama_full()` pour les statistiques complètes.
    """
    opt    = option_type.lower()
    df     = np.exp(-r * T)
    S_T, _ = simulate_euler_paths(S, r, sigma, T, M, N, seed, n_store_paths=0)
    pf     = np.maximum(S_T - K, 0) if opt == "call" else np.maximum(K - S_T, 0)
    return pf.mean() * df


def euler_terminal_distribution(
    S_T: np.ndarray, K: float, option_type: str = "call"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distribution terminale des prix et des payoffs.

    Returns
    -------
    (S_T, payoffs) : tableaux numpy
    """
    pf = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    return S_T, pf


def euler_step_sensitivity(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: str = "call",
    N_values: list[int] | None = None,
    M_per_step: int = 5000,
) -> dict:
    """
    Étudie l'influence du nombre de pas N sur le prix et le biais.

    Returns
    -------
    dict avec 'N_values', 'prices', 'biases', 'biases_rel', 'bs_reference'
    """
    if N_values is None:
        N_values = [1, 5, 10, 20, 50, 100, 252, 500, 1000]

    bs_ref = black_scholes_price(S, K, r, sigma, T, option_type).price
    prices, biases, biases_rel = [], [], []

    for n in N_values:
        p   = euler_maruyama_price(S, K, r, sigma, T, M_per_step, n, option_type)
        b   = abs(p - bs_ref)
        br  = b / bs_ref * 100 if bs_ref > 1e-8 else 0.0
        prices.append(p)
        biases.append(b)
        biases_rel.append(br)

    return {
        "N_values"    : N_values,
        "prices"      : prices,
        "biases"      : biases,
        "biases_rel"  : biases_rel,
        "bs_reference": bs_ref,
        "dt_values"   : [T / n for n in N_values],
    }


# ── Fonction principale complète ────────────────────────────────────────────

def euler_maruyama_full(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    M: int = 10_000,
    N: int = 252,
    option_type: Literal["call", "put"] = "call",
    seed: int | None = None,
    compute_step_sensitivity: bool = True,
    n_store_paths: int = 8,
    store_samples: int = 2000,
) -> EMResult:
    """
    Simulation Euler-Maruyama complète avec biais, trajectoires et sensibilité à N.

    Parameters
    ----------
    S                       : Prix initial du sous-jacent
    K                       : Strike
    r                       : Taux sans risque
    sigma                   : Volatilité annualisée
    T                       : Maturité (années)
    M                       : Nombre de trajectoires simulées
    N                       : Nombre de pas de discrétisation (Δt = T/N)
    option_type             : 'call' ou 'put'
    seed                    : Graine aléatoire
    compute_step_sensitivity: Calcule la sensibilité à N (défaut True)
    n_store_paths           : Nombre de trajectoires complètes à stocker
    store_samples           : Nombre de prix terminaux à stocker

    Returns
    -------
    EMResult : dataclass complet

    Examples
    --------
    >>> res = euler_maruyama_full(100, 100, 0.05, 0.20, 1.0, M=10000, N=252)
    >>> abs(res.price - 10.45) < 0.5
    True
    """
    t_start = time.perf_counter()
    opt     = option_type.lower().strip()
    dt      = T / N
    df      = np.exp(-r * T)

    # ── Simulation trajectoires ────────────────────────────────────────────
    S_T, sample_paths = simulate_euler_paths(
        S, r, sigma, T, M, N, seed, n_store_paths
    )

    # ── Payoffs et statistiques ────────────────────────────────────────────
    pf      = np.maximum(S_T - K, 0.0) if opt == "call" else np.maximum(K - S_T, 0.0)
    mean_pf = pf.mean()
    std_pf  = pf.std(ddof=1)
    price   = mean_pf * df
    itm_pct = (pf > 0).mean() * 100

    # ── Référence Black-Scholes ────────────────────────────────────────────
    bs_res  = black_scholes_price(S, K, r, sigma, T, opt)
    bs_ref  = bs_res.price
    bias    = price - bs_ref
    abs_b   = abs(bias)
    rel_b   = abs_b / bs_ref * 100 if bs_ref > 1e-8 else 0.0

    # ── Échantillons pour graphiques ───────────────────────────────────────
    rng    = np.random.default_rng((seed or 0) + 1)
    n_st   = min(store_samples, M)
    idx    = rng.integers(0, M, n_st)

    # ── Grille temporelle ──────────────────────────────────────────────────
    time_grid = np.linspace(0, T, N + 1)

    elapsed = time.perf_counter() - t_start

    # ── Sensibilité à N ────────────────────────────────────────────────────
    N_vals, p_N, b_N = [], [], []
    if compute_step_sensitivity:
        N_test = [n for n in [5, 10, 25, 50, 100, 252, 500, 1000] if n <= max(N * 4, 1000)]
        sens   = euler_step_sensitivity(
            S, K, r, sigma, T, opt, N_test, M_per_step=min(M, 3000)
        )
        N_vals = sens["N_values"]
        p_N    = sens["prices"]
        b_N    = sens["biases"]

    return EMResult(
        option_type         = opt,
        S=S, K=K, r=r, sigma=sigma, T=T, M=M, N=N,
        dt                  = dt,
        price               = price,
        bs_reference        = bs_ref,
        bias                = bias,
        absolute_bias       = abs_b,
        relative_bias_pct   = rel_b,
        mean_payoff         = mean_pf,
        std_payoff          = std_pf,
        discount_factor     = df,
        itm_pct             = itm_pct,
        terminal_prices     = S_T[idx],
        payoffs             = pf[idx],
        sample_paths        = sample_paths,
        time_grid           = time_grid,
        N_values            = N_vals,
        prices_vs_N         = p_N,
        biases_vs_N         = b_N,
        elapsed_s           = elapsed,
        simulations_per_sec = M / elapsed if elapsed > 0 else 0,
    )


# ── Comparaison des schémas ─────────────────────────────────────────────────

def compare_schemes(
    S: float, K: float, r: float, sigma: float, T: float,
    M: int = 5000, N: int = 252, option_type: str = "call",
) -> dict:
    """
    Compare le schéma d'Euler-Maruyama avec la solution exacte MC et B-S.

    Returns
    -------
    dict avec prix et erreurs pour les 3 méthodes
    """
    from .monte_carlo import monte_carlo_price
    bs  = black_scholes_price(S, K, r, sigma, T, option_type)
    mc  = monte_carlo_price(S, K, r, sigma, T, M, option_type,
                            compute_convergence=False)
    em  = euler_maruyama_full(S, K, r, sigma, T, M, N, option_type,
                               compute_step_sensitivity=False)
    return {
        "black_scholes"     : {"price": bs.price, "error": 0.0, "type": "exact"},
        "monte_carlo"       : {"price": mc.price, "error": mc.absolute_error, "type": "exact MBG"},
        "euler_maruyama"    : {"price": em.price, "error": em.absolute_bias, "type": "EDS discret"},
        "bias_euler_vs_MC"  : abs(em.price - mc.price),
    }


# ── Point d'entrée de test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Euler-Maruyama — Test de référence")
    print("=" * 60)
    res = euler_maruyama_full(
        100, 100, 0.05, 0.20, 1.0,
        M=10_000, N=252, option_type="call", seed=42,
        compute_step_sensitivity=True,
    )
    print(f"\n  Prix Euler    : {res.price:.6f}")
    print(f"  Référence BS  : {res.bs_reference:.6f}")
    print(f"  Biais absolu  : {res.absolute_bias:.6f}")
    print(f"  Biais relatif : {res.relative_bias_pct:.4f} %")
    print(f"  Δt = T/N      : {res.dt:.6f}")
    print(f"  % ITM         : {res.itm_pct:.2f} %")
    print(f"  Temps         : {res.elapsed_s*1000:.2f} ms")
    print(f"  Sim/sec       : {res.simulations_per_sec:,.0f}")
    print(f"\n  Sensibilité à N :")
    for n, p, b in zip(res.N_values, res.prices_vs_N, res.biases_vs_N):
        print(f"    N={n:>5}  prix={p:.4f}  biais={b:.4f}")
    print("=" * 60)
