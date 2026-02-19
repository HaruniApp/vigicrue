# ML — Prédiction de crues (bassin Vilaine Amont)

Pipeline de Machine Learning pour prédire les hauteurs d'eau et débits aux 11 stations hydrométriques du bassin Vilaine Amont, en utilisant les données historiques H/Q, les précipitations et l'humidité du sol Open-Meteo.

## Architecture du modèle

**Station-Attention avec Quantile Regression** :
- Per-station LSTM (partagé) → Cross-station Multi-Head Attention → Shared Decoders (H + Q)
- 3 quantiles par horizon : q10, q50, q90 (intervalles de confiance natifs)
- Pinball loss avec pénalité asymétrique en crue (H ≥ 800mm à Châteaubourg)
- 11 stations × 24 horizons (1-24h) × 3 quantiles = 1296 sorties

## Infrastructure

**NVIDIA DGX Spark** — Grace Blackwell GB10, CUDA 13.0, 128 Go mémoire unifiée.

## Installation

```bash
cd ml
python3 -m venv venv
source venv/bin/activate

# Tout installer (torch CUDA 13.0 + dépendances PyPI)
pip install -r requirements.txt
```

Vérifier que le GPU est bien détecté :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GB10
```

## Étapes

### 1. Collecte des données

```bash
# Données hydrologiques (H + Q) — 11 stations, depuis 2000
python collect_hydro.py

# Précipitations + humidité du sol Open-Meteo — 11 points géographiques
python collect_meteo.py
```

Les CSV sont stockés dans `data/raw/` (gitignored).

Les scripts sont **incrémentaux** : relancer ne récupère que les nouvelles données depuis le dernier timestamp collecté. La date de fin est automatiquement aujourd'hui.

Options utiles :
- `--start 2010-01-01` pour ajuster la date de début
- `--station J706062001` pour ne collecter qu'une station (hydro uniquement)
- `--full` pour forcer une collecte complète (ignore les CSV existants)

### 2. Validation des données

```bash
python validate_data.py
```

Vérifie pour chaque station : fichiers présents, plage temporelle, taux de couverture horaire, trous > 24h, statistiques min/max/moyenne.

### 3. Préparation du dataset

```bash
python prepare_dataset.py
```

Fusionne les 11 stations, interpole les trous, crée les features (H, Q, dH, dQ, précip, humidité du sol — 7 vars/station), normalise et découpe en fenêtres glissantes.

Split chronologique :
- **Train** : 2000-01-01 → 2025-06-30 (25.5 ans, inclut la crue majeure de janvier 2025)
- **Validation** : 2025-07-01 → 2025-12-31 (6 mois, montée automnale)
- **Test** : 2026-01-01 → aujourd'hui (hiver 2025-2026)

Les fichiers `.npy` et métadonnées sont dans `data/processed/`.

### 4. Entraînement

```bash
python train_station_attention.py --hidden 128 --lr 5e-4 --attention-heads 8 --attn-layers 3
```

Options : `--epochs 100 --batch-size 256 --patience 20 --lstm-layers 2`

Inclut le suréchantillonnage automatique des crues (x2/x4/x8 selon le seuil) et affiche les métriques de calibration des quantiles.

### 5. Évaluation

```bash
python evaluate.py
```

Affiche les métriques (NSE, RMSE, MAE) par station et par horizon sur les jeux val/test.

### 6. Export ONNX

```bash
python export_onnx.py --model station_attn
```

Exporte le modèle en ONNX dans `models/onnx/`, prêt pour l'inférence dans le backend Node.js via `onnxruntime-node`. Inclut les métadonnées quantile (`n_quantiles`, `quantiles`).

### 7. Debug temps réel

```bash
python debug_realtime.py
```

Fetch les données temps réel des 11 stations et lance l'inférence ONNX. Affiche les prédictions q50 avec intervalles [q10 – q90] pour vérification.

### 8. Déploiement

```bash
cp models/onnx/station_attn.onnx ../../backend/models/
cp models/onnx/station_attn_meta.json ../../backend/models/
cp models/onnx/norm_params.json ../../backend/models/
```

## Structure

```
ml/
├── config.py                    # stations, coordonnées, paramètres
├── collect_hydro.py             # collecte H+Q (Hydro EauFrance)
├── collect_meteo.py             # collecte précipitations + humidité du sol (Open-Meteo)
├── validate_data.py             # validation des données brutes
├── prepare_dataset.py           # nettoyage, features, fenêtrage
├── train_station_attention.py   # modèle Station-Attention + quantile regression
├── evaluate.py                  # métriques par station/horizon
├── export_onnx.py               # export ONNX + métadonnées
├── debug_realtime.py            # test inférence temps réel
├── requirements.txt             # dépendances Python
├── data/                        # gitignored
│   ├── raw/                     # CSV bruts par station
│   └── processed/               # dataset unifié (.npy)
└── models/                      # gitignored
    ├── checkpoints/             # poids PyTorch + résultats JSON
    └── onnx/                    # modèle exporté + métadonnées
```

## Métriques

Résultats sur le jeu de test — hiver 2025-2026 (Châteaubourg, q50) :

| Horizon | NSE | RMSE |
|---------|------|------|
| t+1h | 0.9999 | 6 mm |
| t+6h | 0.9968 | 36 mm |
| t+12h | 0.9884 | 69 mm |
| t+24h | 0.9699 | 112 mm |

Calibration des quantiles : P(y < q10) ≈ 10%, P(y < q90) ≈ 85% (cibles 10%/90%).
