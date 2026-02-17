"""Validation des données brutes collectées avant préparation du dataset.

Vérifie pour chaque station :
- Présence des fichiers CSV (H, Q, précipitations)
- Plage temporelle couverte
- Nombre de points et taux de couverture horaire
- Trous majeurs (>24h)
- Statistiques de base (min, max, moyenne)

Usage:
    python validate_data.py
"""

from datetime import timedelta

import pandas as pd

from config import RAW_DIR, STATIONS


def validate_csv(path, value_col: str) -> dict | None:
    """Valide un fichier CSV et retourne les statistiques."""
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return {"status": "vide", "count": 0}

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]

    # Plage et couverture
    start, end = ts.min(), ts.max()
    total_hours = (end - start).total_seconds() / 3600
    expected_points = int(total_hours) + 1
    coverage = len(df) / expected_points * 100 if expected_points > 0 else 0

    # Trous > 24h
    diffs = ts.diff().dropna()
    big_gaps = diffs[diffs > timedelta(hours=24)]
    gaps_info = []
    for idx in big_gaps.index:
        gap_start = ts.iloc[idx - 1]
        gap_end = ts.iloc[idx]
        gap_hours = (gap_end - gap_start).total_seconds() / 3600
        gaps_info.append((gap_start, gap_end, gap_hours))

    # Stats sur les valeurs
    values = df[value_col].dropna()

    return {
        "status": "ok",
        "count": len(df),
        "start": start,
        "end": end,
        "total_hours": int(total_hours),
        "coverage_pct": round(coverage, 1),
        "nan_count": int(df[value_col].isna().sum()),
        "gaps_24h": gaps_info,
        "min": round(float(values.min()), 2) if len(values) > 0 else None,
        "max": round(float(values.max()), 2) if len(values) > 0 else None,
        "mean": round(float(values.mean()), 2) if len(values) > 0 else None,
    }


def main():
    print("Validation des données brutes")
    print(f"Dossier : {RAW_DIR}")
    print("=" * 80)

    total_files = 0
    missing_files = 0
    all_ok = True

    for station in STATIONS:
        code = station["code"]
        label = station["label"]
        print(f"\n{'─'*80}")
        print(f"  {label} ({code}) — {station['river']}, {station['position']}")
        print(f"{'─'*80}")

        for var, col in [("h", "h"), ("q", "q"), ("precip", "precipitation")]:
            path = RAW_DIR / f"{code}_{var}.csv"
            total_files += 1
            result = validate_csv(path, col)

            if result is None:
                print(f"  {var.upper():>6} : ✗ FICHIER ABSENT")
                missing_files += 1
                if var != "q":  # Q manquant est normal pour les barrages
                    all_ok = False
                continue

            if result["status"] == "vide":
                print(f"  {var.upper():>6} : ✗ FICHIER VIDE")
                all_ok = False
                continue

            # Affichage
            status = "✓" if result["coverage_pct"] > 80 else "⚠"
            if result["coverage_pct"] < 50:
                status = "✗"
                all_ok = False

            print(f"  {var.upper():>6} : {status} {result['count']:>8} points  "
                  f"| {result['start'].strftime('%Y-%m-%d')} → {result['end'].strftime('%Y-%m-%d')}  "
                  f"| couverture {result['coverage_pct']:>5.1f}%  "
                  f"| min={result['min']}  max={result['max']}  moy={result['mean']}")

            if result["nan_count"] > 0:
                print(f"           NaN: {result['nan_count']}")

            if result["gaps_24h"]:
                print(f"           Trous >24h : {len(result['gaps_24h'])}")
                for gap_start, gap_end, hours in result["gaps_24h"][:5]:
                    print(f"             {gap_start.strftime('%Y-%m-%d %H:%M')} → "
                          f"{gap_end.strftime('%Y-%m-%d %H:%M')} ({hours:.0f}h)")
                if len(result["gaps_24h"]) > 5:
                    print(f"             ... et {len(result['gaps_24h']) - 5} autres trous")

    # Résumé
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ")
    print(f"{'='*80}")
    print(f"  Fichiers : {total_files - missing_files}/{total_files} présents")
    if all_ok:
        print(f"  ✓ Données prêtes pour prepare_dataset.py")
    else:
        print(f"  ⚠ Problèmes détectés — vérifier avant de lancer prepare_dataset.py")


if __name__ == "__main__":
    main()
