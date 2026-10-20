/* Fonctions numériques partagées par toutes les pages.
 * normCDF : algorithme de Hart (1968), précision proche de la double précision
 * (l'approximation d'Abramowitz-Stegun utilisée auparavant avait une erreur ~1e-7).
 * mlPredictMany : prédictions du vrai modèle scikit-learn entraîné côté serveur.
 */
(function () {
  function normCDF(x) {
    var z = Math.abs(x), c;
    if (z > 37) c = 0;
    else {
      var e = Math.exp(-z * z / 2);
      if (z < 7.07106781186547) {
        var n = ((((((0.0352624965998911 * z + 0.700383064443688) * z + 6.37396220353165) * z + 33.912866078383) * z + 112.079291497871) * z + 221.213596169931) * z + 220.206867912376);
        var d = (((((((0.0883883476483184 * z + 1.75566716318264) * z + 16.064177579207) * z + 86.7807322029461) * z + 296.564248779674) * z + 637.333633378831) * z + 793.826512519948) * z + 440.413735824752);
        c = e * n / d;
      } else {
        c = e / (z + 1 / (z + 2 / (z + 3 / (z + 4 / (z + 0.65))))) / 2.506628274631;
      }
    }
    return x > 0 ? 1 - c : c;
  }

  async function mlPredictMany(rows, modelType) {
    var res = await fetch('/api/ml/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rows: rows, model_type: modelType || 'rf' })
    });
    var data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Prédiction ML indisponible');
    return data; /* { prices: [...], model_name, elapsed_s, metrics } */
  }

  window.OPL = { normCDF: normCDF, mlPredictMany: mlPredictMany };
})();
