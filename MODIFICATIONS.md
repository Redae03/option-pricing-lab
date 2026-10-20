# Modifications apportées (octobre 2026)

## 1. Machine Learning réellement branché sur scikit-learn
Avant : les pages calculaient un faux « prix ML » en JavaScript (prix Black-Scholes + bruit aléatoire), et la page Machine Learning simulait l'entraînement, les métriques et l'importance des variables.
Après : toutes les pages appellent le vrai modèle via l'API.
- `static/js/pricing_core.js` (nouveau) : `OPL.mlPredictMany()` interroge `/api/ml/predict`.
- `templates/dashboard.html`, `comparaison.html`, `export.html`, `historique.html` : prix ML issu du modèle entraîné.
- `templates/machine_learning.html` : pipeline réel (`/api/train-ml`) — dataset, MAE/RMSE/R², résidus, importance des variables du Random Forest, courbe d'apprentissage, prédiction.

## 2. Backend
- `app.py` : nouvelles routes `/api/ml/predict` (prédictions groupées) et `/api/ml/status` ; `/api/train-ml` accepte le type d'option, la plage de σ, le ratio train/test et la complexité.
- `ml/train_model.py` : le Random Forest apprend C/K (MAE 3,22 → 1,80) ; renvoie les prédictions de test, les importances et la courbe d'apprentissage.
- `ml/predict_model.py` : prédiction groupée ; sans modèle, ne renvoie plus le prix Black-Scholes présenté comme une prédiction ML.
- `services/ml_service.py` : entraînement automatique au premier appel si aucun modèle n'existe.

## 3. Précision numérique
La fonction de répartition normale utilisait l'approximation d'Abramowitz-Stegun (erreur ~1e-7). Elle est remplacée dans toutes les pages par l'algorithme de Hart, précis à la double précision.

## 4. Divers
Ajout de `.gitignore` et `README.md` ; les dossiers `models/`, `data/` et `outputs/` sont conservés vides.
