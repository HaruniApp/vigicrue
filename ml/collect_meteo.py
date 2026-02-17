"""Collecte des précipitations horaires depuis Open-Meteo Historical Weather API.

Open-Meteo accepte des plages longues (plusieurs années) en une seule requête,
mais on découpe en segments annuels pour fiabilité.

Mode incrémental : si un CSV existe déjà, ne récupère que les données
postérieures au dernier timestamp présent.

Usage:
    python collect_meteo.py
    python collect_meteo.py --start 2000-01-01
    python collect_meteo.py --full   # ignore les CSV existants
"""

import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    COLLECT_END_DATE,
    COLLECT_START_DATE,
    METEO_BASE_URL,
    RAW_DIR,
    STATIONS,
)

MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_DELAY = 0.5  # Open-Meteo est généreux mais restons polis


def fetch_precipitation(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Récupère les précipitations horaires pour un point géographique."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation",
        "timezone": "Europe/Paris",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(METEO_BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            precip = hourly.get("precipitation", [])

            if not times:
                return None

            df = pd.DataFrame({"timestamp": pd.to_datetime(times), "precipitation": precip})
            return df

        except (requests.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} ({e})")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  ÉCHEC: {e}")
                return None


def generate_yearly_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Découpe en segments annuels."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    ranges = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=365), end)
        ranges.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end
    return ranges


def get_last_timestamp(csv_path) -> str | None:
    """Retourne le dernier timestamp d'un CSV existant, ou None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if df.empty:
            return None
        last = df["timestamp"].max()
        return last.strftime("%Y-%m-%d")
    except Exception:
        return None


def collect_station_meteo(station: dict, start_date: str, end_date: str, full: bool = False) -> None:
    """Collecte les précipitations pour une station."""
    code = station["code"]
    label = station["label"]
    lat, lon = station["lat"], station["lon"]

    out_path = RAW_DIR / f"{code}_precip.csv"
    existing_df = None
    effective_start = start_date

    # Mode incrémental
    if not full:
        last_ts = get_last_timestamp(out_path)
        if last_ts is not None:
            if last_ts >= end_date:
                print(f"\n  {label} ({code}) — déjà à jour, skip")
                return
            effective_start = last_ts
            existing_df = pd.read_csv(out_path, parse_dates=["timestamp"])
            print(f"\n  {label} ({code}) — incrémental depuis {last_ts}")
        else:
            print(f"\n  {label} ({code}) — lat={lat}, lon={lon}")
    else:
        print(f"\n  {label} ({code}) — lat={lat}, lon={lon}")

    ranges = generate_yearly_ranges(effective_start, end_date)
    all_dfs = []

    for seg_start, seg_end in tqdm(ranges, desc=f"  {code}", leave=False):
        df = fetch_precipitation(lat, lon, seg_start, seg_end)
        if df is not None:
            all_dfs.append(df)
        time.sleep(REQUEST_DELAY)

    if not all_dfs and existing_df is None:
        print(f"  ⚠ Aucune donnée météo pour {code}")
        return

    new_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Fusionner avec les données existantes
    if existing_df is not None and not new_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    elif existing_df is not None:
        df = existing_df
    else:
        df = new_df

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"  ✓ {len(df)} points → {out_path.name}")
    print(f"    Plage: {df['timestamp'].min()} → {df['timestamp'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Collecte des précipitations Open-Meteo")
    parser.add_argument("--start", default=COLLECT_START_DATE, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default=COLLECT_END_DATE, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true", help="Collecte complète (ignore les CSV existants)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    mode = "complète" if args.full else "incrémentale"
    print(f"Collecte Open-Meteo ({mode}) : {args.start} → {args.end}")
    print(f"Stations : {len(STATIONS)}")

    for station in STATIONS:
        collect_station_meteo(station, args.start, args.end, full=args.full)

    print("\n✓ Collecte météo terminée.")


if __name__ == "__main__":
    main()
