from __future__ import annotations

import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

from config import DEFAULT_PARAMS, OUTPUT_DIR
from services.pricing_service import compute_black_scholes
from services.simulation_service import compute_monte_carlo, compute_euler
from services.ml_service import compute_ml_prediction, train_ml_model, predict_many, model_status
from services.chart_service import histogram, to_list
from services.history_service import load_history, save_scenario, clear_history
from services.export_service import export_comparison_csv, export_comparison_excel, export_comparison_pdf
from core.comparison import build_comparison_rows, best_method
from core.black_scholes import sensitivity_table

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def parse_params(source) -> dict:
    """Lit les paramètres depuis un formulaire ou JSON."""
    src = source or {}
    def get(name, default):
        if hasattr(src, "get"):
            return src.get(name, src.get(name.lower(), default))
        return default
    option_type = str(get("option_type", DEFAULT_PARAMS["option_type"])).lower()
    if option_type in ["Call", "CALL"]:
        option_type = "call"
    if option_type in ["Put", "PUT"]:
        option_type = "put"
    return {
        "option_type": option_type if option_type in ("call", "put") else "call",
        "S": float(get("S", get("S0", DEFAULT_PARAMS["S"]))),
        "K": float(get("K", DEFAULT_PARAMS["K"])),
        "r": float(get("r", DEFAULT_PARAMS["r"])),
        "sigma": float(get("sigma", DEFAULT_PARAMS["sigma"])),
        "T": float(get("T", DEFAULT_PARAMS["T"])),
        "M": int(float(get("M", DEFAULT_PARAMS["M"]))),
        "N": int(float(get("N", DEFAULT_PARAMS["N"]))),
        "seed": int(float(get("seed", DEFAULT_PARAMS["seed"]))),
    }


def fmt(x, digits=6):
    try:
        return round(float(x), digits)
    except Exception:
        return x


def compute_all(params: dict, include_heavy_charts: bool = True) -> dict:
    """Calcule toutes les méthodes et prépare les données JSON."""
    bs = compute_black_scholes(params)
    mc = compute_monte_carlo(params, convergence=include_heavy_charts)
    em = compute_euler(params, sensitivity=include_heavy_charts)
    ml = compute_ml_prediction(params, model_type="rf")

    rows = build_comparison_rows(bs, mc, em, ml)
    best = best_method(rows)

    payload = {
        "params": params,
        "black_scholes": {
            "price": fmt(bs.price),
            "call_price": fmt(bs.call_price),
            "put_price": fmt(bs.put_price),
            "d1": fmt(bs.d1),
            "d2": fmt(bs.d2),
            "delta": fmt(bs.delta),
            "gamma": fmt(bs.gamma, 8),
            "theta": fmt(bs.theta),
            "vega": fmt(bs.vega),
            "rho": fmt(bs.rho),
            "moneyness": bs.moneyness,
            "intrinsic_value": fmt(bs.intrinsic_value),
            "time_value": fmt(bs.time_value),
            "elapsed_s": fmt(bs.elapsed_s, 8),
        },
        "monte_carlo": {
            "price": fmt(mc.price),
            "bs_reference": fmt(mc.bs_reference),
            "absolute_error": fmt(mc.absolute_error),
            "relative_error_pct": fmt(mc.relative_error_pct),
            "ci_95_lo": fmt(mc.ci_95_lo),
            "ci_95_hi": fmt(mc.ci_95_hi),
            "std_payoff": fmt(mc.std_payoff),
            "std_error": fmt(mc.std_error),
            "itm_pct": fmt(mc.itm_pct),
            "elapsed_s": fmt(mc.elapsed_s, 6),
        },
        "euler": {
            "price": fmt(em.price),
            "bs_reference": fmt(em.bs_reference),
            "bias": fmt(em.bias),
            "absolute_bias": fmt(em.absolute_bias),
            "relative_bias_pct": fmt(em.relative_bias_pct),
            "dt": fmt(em.dt),
            "itm_pct": fmt(em.itm_pct),
            "elapsed_s": fmt(em.elapsed_s, 6),
        },
        "ml": {
            "price": fmt(ml["price"]),
            "model_name": ml.get("model_name"),
            "elapsed_s": fmt(ml.get("elapsed_s", 0), 6),
            "is_trained": ml.get("is_trained", False),
            "mae": fmt(ml.get("mae"), 6) if ml.get("mae") is not None else None,
            "rmse": fmt(ml.get("rmse"), 6) if ml.get("rmse") is not None else None,
            "r2": fmt(ml.get("r2"), 6) if ml.get("r2") is not None else None,
        },
        "rows": rows,
        "best": best,
        "charts": {
            "prices": {
                "labels": [r["method"] for r in rows],
                "values": [fmt(r["price"]) for r in rows],
            },
            "errors": {
                "labels": [r["method"] for r in rows if r["method"] != "Black-Scholes"],
                "values": [fmt(r["relative_error"], 4) for r in rows if r["method"] != "Black-Scholes"],
            },
            "times": {
                "labels": [r["method"] for r in rows],
                "values": [max(float(r["time"]), 0.000001) for r in rows],
            },
            "mc_terminal_hist": histogram(mc.terminal_prices, bins=24),
            "mc_payoff_hist": histogram(mc.payoffs, bins=24),
            "mc_convergence": {
                "M": mc.convergence_M,
                "prices": [fmt(x) for x in mc.convergence_prices],
                "lo": [fmt(x) for x in mc.convergence_ci_lo],
                "hi": [fmt(x) for x in mc.convergence_ci_hi],
                "bs": fmt(bs.price),
            },
            "euler_terminal_hist": histogram(em.terminal_prices, bins=24),
            "euler_paths": {
                "time": to_list(em.time_grid, max_points=400),
                "paths": [to_list(path, max_points=400) for path in em.sample_paths[:8]],
            },
            "euler_steps": {
                "N": em.N_values,
                "prices": [fmt(x) for x in em.prices_vs_N],
                "biases": [fmt(x) for x in em.biases_vs_N],
            },
        }
    }
    return payload


@app.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard", title="Tableau de bord", subtitle="Sélectionnez une méthode de pricing ou comparez-les toutes")


@app.route("/black-scholes")
def black_scholes_page():
    return render_template("black_scholes.html", active="black", title="Black-Scholes", subtitle="Formule analytique pour options européennes")


@app.route("/monte-carlo")
def monte_carlo_page():
    return render_template("monte_carlo.html", active="mc", title="Monte-Carlo", subtitle="Simulation classique du prix terminal")


@app.route("/euler-maruyama")
def euler_maruyama_page():
    return render_template("euler_maruyama.html", active="euler", title="Euler-Maruyama", subtitle="Discrétisation de l'EDS de Black-Scholes")


@app.route("/machine-learning")
def machine_learning_page():
    return render_template("machine_learning.html", active="ml", title="Machine Learning", subtitle="Surrogate model pour approximer le prix")


@app.route("/comparaison")
def comparaison_page():
    return render_template("comparaison.html", active="comparison", title="Comparaison", subtitle="Comparaison Black-Scholes, Monte-Carlo, Euler-Maruyama et Machine Learning")


@app.route("/sensibilite")
def sensibilite_page():
    return render_template("sensibilite.html", active="sensitivity", title="Analyse de sensibilité", subtitle="Impact des paramètres sur le prix")


@app.route("/historique")
def historique_page():
    return render_template("historique.html", active="history", title="Historique", subtitle="Scénarios sauvegardés", history=load_history())


@app.route("/export")
def export_page():
    return render_template("export.html", active="export", title="Export", subtitle="Rapports et données")


@app.route("/api/compute-comparison", methods=["POST"])
def api_compute_comparison():
    params = parse_params(request.get_json(silent=True) or request.form)
    try:
        return jsonify(compute_all(params, include_heavy_charts=True))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/train-ml", methods=["POST"])
def api_train_ml():
    payload = request.get_json(silent=True) or {}
    model_type = payload.get("model_type", "rf")
    n_samples = min(max(int(payload.get("n_samples", 3000)), 200), 20000)
    dtype = payload.get("option_types", "both")
    options = {
        "option_types": ["call", "put"] if dtype == "both" else [dtype],
        "sigma_range": tuple(payload.get("sigma_range", (0.05, 0.80))),
        "train_ratio": float(payload.get("train_ratio", 0.8)),
        "complexity": int(payload["complexity"]) if payload.get("complexity") else None,
        "learning_curve": bool(payload.get("learning_curve", False)),
    }
    try:
        res = train_ml_model(model_type=model_type, n_samples=n_samples, **options)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ml/predict", methods=["POST"])
def api_ml_predict():
    """Prédictions du modèle ML entraîné pour une liste d'options."""
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows") or []
    if not rows:
        return jsonify({"error": "Aucune option à évaluer"}), 400
    try:
        return jsonify(predict_many(rows, payload.get("model_type", "rf")))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ml/status")
def api_ml_status():
    return jsonify(model_status(request.args.get("model_type", "rf")))


@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    params = parse_params(request.get_json(silent=True) or request.form)
    try:
        raw = sensitivity_table(params["S"], params["K"], params["r"], params["sigma"], params["T"], params["option_type"])
        rows = []
        for item in raw:
            rows.append({"variable": item["scenario"] + " (bas)", "value": item["lo_val"], "price": item["price_lo"]})
            rows.append({"variable": item["scenario"] + " (base)", "value": item["base_val"], "price": item["price_base"]})
            rows.append({"variable": item["scenario"] + " (haut)", "value": item["hi_val"], "price": item["price_hi"]})
        return jsonify({"rows": rows, "raw": raw})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/save-scenario", methods=["POST"])
def api_save_scenario():
    payload = request.get_json(silent=True) or {}
    item = save_scenario(payload)
    return jsonify({"ok": True, "item": item})


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    clear_history()
    return jsonify({"ok": True})


@app.route("/api/export/<kind>", methods=["POST"])
def api_export(kind):
    params = parse_params((request.get_json(silent=True) or {}).get("params", DEFAULT_PARAMS))
    data = compute_all(params, include_heavy_charts=False)
    rows = data["rows"]
    if kind == "csv":
        path = export_comparison_csv(rows)
    elif kind == "excel":
        path = export_comparison_excel(rows)
    elif kind == "pdf":
        path = export_comparison_pdf(rows, params)
    else:
        return jsonify({"error": "Type d'export inconnu"}), 400
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
