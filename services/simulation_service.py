"""Services de simulation Monte-Carlo et Euler-Maruyama."""
from core.monte_carlo import monte_carlo_price
from core.euler_maruyama import euler_maruyama_full, euler_step_sensitivity


def compute_monte_carlo(params: dict, convergence: bool = True):
    return monte_carlo_price(
        params["S"], params["K"], params["r"], params["sigma"], params["T"],
        M=int(params["M"]), option_type=params["option_type"], seed=int(params.get("seed", 42)),
        compute_convergence=convergence,
    )


def compute_euler(params: dict, sensitivity: bool = False):
    return euler_maruyama_full(
        params["S"], params["K"], params["r"], params["sigma"], params["T"],
        M=int(params["M"]), N=int(params["N"]), option_type=params["option_type"],
        seed=int(params.get("seed", 42)), compute_step_sensitivity=sensitivity,
    )
