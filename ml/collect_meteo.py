"""Collecte des précipitations horaires depuis Open-Meteo Historical Weather API.

Open-Meteo accepte des plages longues (plusieurs années) en une seule requête,
mais on découpe en segments annuels pour fiabilité.

Usage:
    python collect_meteo.py
    python collect_meteo.py --start 2000-01-01 --end 2025-02-01
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


def collect_station_meteo(station: dict, start_date: str, end_date: str) -> None:
    """Collecte les précipitations pour une station."""
    code = station["code"]
    label = station["label"]
    lat, lon = station["lat"], station["lon"]

    print(f"\n  {label} ({code}) — lat={lat}, lon={lon}")

    ranges = generate_yearly_ranges(start_date, end_date)
    all_dfs = []

    for seg_start, seg_end in tqdm(ranges, desc=f"  {code}", leave=False):
        df = fetch_precipitation(lat, lon, seg_start, seg_end)
        if df is not None:
            all_dfs.append(df)
        time.sleep(REQUEST_DELAY)

    if not all_dfs:
        print(f"  ⚠ Aucune donnée météo pour {code}")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    out_path = RAW_DIR / f"{code}_precip.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ {len(df)} points → {out_path.name}")
    print(f"    Plage: {df['timestamp'].min()} → {df['timestamp'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Collecte des précipitations Open-Meteo")
    parser.add_argument("--start", default=COLLECT_START_DATE, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default=COLLECT_END_DATE, help="Date de fin (YYYY-MM-DD)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Collecte Open-Meteo : {args.start} → {args.end}")
    print(f"Stations : {len(STATIONS)}")

    for station in STATIONS:
        collect_station_meteo(station, args.start, args.end)

    print("\n✓ Collecte météo terminée.")


if __name__ == "__main__":
    main()
