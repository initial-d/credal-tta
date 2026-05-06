"""
Download NOAA Weather data (Winter Storm Uri, Feb 2021) using meteostat.

This script downloads hourly temperature data from Dallas/Fort Worth area
covering Jan-Mar 2021 (including Winter Storm Uri ~Feb 15) and saves it
as data/noaa_weather.npy in the format expected by credal_tta.

Usage:
    pip install meteostat
    python download_noaa.py
"""

import numpy as np
import os

try:
    from meteostat import Point, Hourly
    from datetime import datetime

    # Dallas/Fort Worth, TX - epicenter of Winter Storm Uri
    location = Point(32.8968, -97.038)  # DFW Airport coordinates

    # Time range: Jan 1 - Mar 31, 2021
    start = datetime(2021, 1, 1)
    end = datetime(2021, 3, 31, 23, 59)

    print("Downloading hourly temperature data from meteostat...")
    print("Location: Dallas/Fort Worth, TX")
    print(f"Period: {start.date()} to {end.date()}")

    data = Hourly(location, start, end)
    df = data.fetch()

    if df.empty:
        raise ValueError("No data returned from meteostat")

    # 'temp' column is temperature in Celsius
    temp = df['temp'].interpolate().values
    print(f"Downloaded {len(temp)} hourly observations")
    print(f"Temperature range: {temp.min():.1f}°C to {temp.max():.1f}°C")

    # Reshape to (1, T) - single station
    weather_data = temp.reshape(1, -1)

    os.makedirs("data", exist_ok=True)
    np.save("data/noaa_weather.npy", weather_data)
    print(f"\nSaved to data/noaa_weather.npy, shape: {weather_data.shape}")
    print("Done!")

except ImportError:
    print("meteostat not installed. Trying alternative approach with open-meteo API...")
    import urllib.request
    import json

    # Use Open-Meteo free API (no key needed)
    # Dallas/Fort Worth coordinates
    lat, lon = 32.8968, -97.038
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date=2021-01-01&end_date=2021-03-31"
        f"&hourly=temperature_2m&timezone=America/Chicago"
    )

    print("Downloading hourly temperature data from Open-Meteo API...")
    print("Location: Dallas/Fort Worth, TX")
    print("Period: 2021-01-01 to 2021-03-31")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode())

    temps = result['hourly']['temperature_2m']
    # Replace None values with interpolation
    temp_array = np.array(temps, dtype=float)
    # Simple forward-fill for NaN
    mask = np.isnan(temp_array)
    if mask.any():
        idx = np.where(~mask, np.arange(len(temp_array)), 0)
        np.maximum.accumulate(idx, out=idx)
        temp_array = temp_array[idx]

    print(f"Downloaded {len(temp_array)} hourly observations")
    print(f"Temperature range: {temp_array.min():.1f}°C to {temp_array.max():.1f}°C")

    weather_data = temp_array.reshape(1, -1)

    os.makedirs("data", exist_ok=True)
    np.save("data/noaa_weather.npy", weather_data)
    print(f"\nSaved to data/noaa_weather.npy, shape: {weather_data.shape}")
    print("Done!")
