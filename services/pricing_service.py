"""Services de pricing analytique."""
from core.black_scholes import black_scholes_price, sensitivity_table


def compute_black_scholes(params: dict):
    return black_scholes_price(
        params["S"], params["K"], params["r"], params["sigma"], params["T"], params["option_type"]
    )
