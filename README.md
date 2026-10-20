# Option Pricing Lab

Application web Flask de valorisation d'options européennes, réalisée par **Reda Erraoui** (ENSA Agadir, filière Finance et Ingénierie Décisionnelle) sous l'encadrement de Pr. B. El Asri.

**[Démonstration en ligne](https://redae03.github.io/#simulateur-opl)** · **[Lire le rapport (PDF)](rapport.pdf)**

## Méthodes comparées

| Méthode | Principe | Erreur | Coût |
|---|---|---|---|
| Black-Scholes-Merton | Formule fermée, grecques, volatilité implicite (Newton-Raphson) | Référence | O(1) |
| Monte-Carlo | Solution exacte du mouvement brownien géométrique, IC à 95 % | Statistique, en O(1/√M) | O(M) |
| Euler-Maruyama | Discrétisation de l'EDS sur N pas | Statistique + biais de discrétisation en O(1/N) | O(M·N) |
| Machine Learning | Random Forest (ou MLP) scikit-learn entraîné sur des prix Black-Scholes | Erreur d'approximation | O(arbres × profondeur) en prédiction |

## Fonctionnalités

- Calcul et comparaison des quatre méthodes (prix, écarts, temps de calcul)
- Grecques (Δ, Γ, Θ, ν, ρ), parité call-put, analyse de sensibilité
- Convergence de Monte-Carlo et biais d'Euler-Maruyama selon N
- Pipeline Machine Learning complet : génération du dataset (hypercube latin), entraînement, métriques (MAE, RMSE, R²), importance des variables, courbe d'apprentissage, prédiction
- Historique des scénarios et export CSV, Excel, JSON, Markdown et PDF

## Résultats Machine Learning

Dataset de 3 000 options (S₀ ∈ [50, 200], K ∈ [40, 220], r ∈ [0, 10 %], σ ∈ [5 %, 80 %], T ∈ [0,1 ; 3] ans), découpage 80/20.

| Modèle | Cible apprise | MAE (test) | R² (test) | Call ATM (BS = 10,4506) |
|---|---|---:|---:|---:|
| Random Forest (180 arbres) | Prix | 3,22 | 0,979 | 11,75 |
| Random Forest (180 arbres) | Prix / K | 1,80 | 0,991 | 10,03 |
| MLP (64, 32) | Prix | 1,38 | 0,998 | 11,41 |

Le Random Forest apprend le prix rapporté au strike (C/K) : la formule de Black-Scholes est homogène de degré 1 en (S, K), ce qui réduit l'espace à apprendre et divise l'erreur moyenne par près de deux. Les prédictions restent des approximations, peu fiables en dehors des plages d'entraînement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Puis ouvrir http://127.0.0.1:5000. Au premier calcul, le modèle ML est entraîné automatiquement s'il n'existe pas encore (quelques secondes).

## Structure

```
├── app.py              # routes Flask et API JSON
├── config.py
├── core/               # Black-Scholes, Monte-Carlo, Euler-Maruyama, métriques
├── ml/                 # génération du dataset, entraînement, prédiction
├── services/           # couche entre l'API et les modèles
├── templates/          # pages HTML (Jinja2)
└── static/             # CSS et JavaScript (Chart.js)
```

## API

| Route | Rôle |
|---|---|
| `POST /api/compute-comparison` | Calcule les quatre méthodes pour un jeu de paramètres |
| `POST /api/train-ml` | Génère le dataset, entraîne et évalue le modèle |
| `POST /api/ml/predict` | Prédictions du modèle entraîné pour une liste d'options |
| `GET /api/ml/status` | Indique si un modèle est entraîné et ses métriques |
| `POST /api/sensitivity` | Tableau de sensibilité Black-Scholes |
| `POST /api/export/<csv\|excel\|pdf>` | Export des résultats |

## Contact

Reda Erraoui · [Portfolio](https://redae03.github.io) · redaerraouie@gmail.com
