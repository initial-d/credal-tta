"""
Dataset Utilities - Synthetic and Real-World Loaders
"""

import numpy as np
from typing import Tuple, List


# ==================== Synthetic Datasets (Section 5.1) ====================

def generate_sin_freq_shift(
    T: int = 1000,
    shift_point: int = 500,
    freq_before: float = 10.0,
    freq_after: float = 30.0,
    amplitude: float = 1.0,
    noise_std: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    SinFreq: Sinusoidal signal with abrupt frequency shift
    
    x_t = A * sin(2π * f_t / P) + ε_t
    where f_t changes at shift_point
    
    Args:
        T: Total length
        shift_point: Regime transition time
        freq_before: Frequency before shift
        freq_after: Frequency after shift
        amplitude: Signal amplitude
        noise_std: Gaussian noise standard deviation
        seed: Random seed
        
    Returns:
        time_series: Generated data
        shift_point: Regime shift location
    """
    np.random.seed(seed)
    
    # Generate time indices
    t = np.arange(T)
    
    # Piecewise frequency
    freq = np.where(t < shift_point, freq_before, freq_after)
    
    # Sinusoidal signal
    signal = amplitude * np.sin(2 * np.pi * freq * t / 100)
    
    # Add noise
    noise = np.random.normal(0, noise_std, T)
    time_series = signal + noise
    
    return time_series, shift_point


def generate_step_mean_shift(
    T: int = 1000,
    shift_point: int = 400,
    mean_before: float = 0.0,
    mean_after: float = 5.0,
    std: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    StepMean: Gaussian process with abrupt mean shift
    
    Args:
        T: Total length
        shift_point: Regime transition time
        mean_before: Mean before shift
        mean_after: Mean after shift
        std: Standard deviation (constant)
        seed: Random seed
        
    Returns:
        time_series: Generated data
        shift_point: Regime shift location
    """
    np.random.seed(seed)
    
    # Piecewise mean
    means = np.where(np.arange(T) < shift_point, mean_before, mean_after)
    
    # Gaussian samples
    time_series = np.random.normal(means, std, T)
    
    return time_series, shift_point


def generate_gradual_drift(
    T: int = 1000,
    drift_start: int = 400,
    drift_end: int = 600,
    freq_before: float = 10.0,
    freq_after: float = 30.0,
    amplitude: float = 1.0,
    noise_std: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    """
    GradualDrift: Sinusoidal with linear frequency transition
    
    Used in Section 6.5 for elastic window ablation
    
    Args:
        T: Total length
        drift_start: Start of drift period
        drift_end: End of drift period
        freq_before: Initial frequency
        freq_after: Final frequency
        amplitude: Signal amplitude
        noise_std: Noise level
        seed: Random seed
        
    Returns:
        time_series: Generated data
        drift_start: Drift onset time
    """
    np.random.seed(seed)
    
    t = np.arange(T)
    
    # Piecewise-linear frequency evolution
    freq = np.zeros(T)
    freq[t < drift_start] = freq_before
    freq[t >= drift_end] = freq_after
    
    # Linear interpolation during drift
    drift_mask = (t >= drift_start) & (t < drift_end)
    drift_progress = (t[drift_mask] - drift_start) / (drift_end - drift_start)
    freq[drift_mask] = freq_before + drift_progress * (freq_after - freq_before)
    
    # Sinusoidal signal
    signal = amplitude * np.sin(2 * np.pi * freq * t / 100)
    noise = np.random.normal(0, noise_std, T)
    
    return signal + noise, drift_start


# ==================== Real-World Dataset Loaders ====================

def load_uci_electricity(
    customer_id: int = 0,
    data_path: str = "data/electricity.npy"
) -> np.ndarray:
    """
    Load UCI Electricity dataset (Section 5.2 in paper)
    
    Dataset: Hourly electricity consumption from 370 customers
    Source: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
    
    Args:
        customer_id: Customer index (0-369)
        data_path: Path to preprocessed .npy file
        
    Returns:
        time_series: Electricity consumption (hourly)
    """
    try:
        data = np.load(data_path)
        return data[customer_id]
    except FileNotFoundError:
        print(f"WARNING: {data_path} not found. Generating synthetic substitute.")
        # Generate synthetic data with seasonal patterns
        T = 26304  # ~3 years hourly
        t = np.arange(T)
        daily = np.sin(2 * np.pi * t / 24)
        weekly = 0.5 * np.sin(2 * np.pi * t / (24 * 7))
        trend = 0.0001 * t
        noise = np.random.normal(0, 0.2, T)
        return 5 + daily + weekly + trend + noise


def load_sp500_crisis(
    start_date: str = "2020-01-01",
    end_date: str = "2020-06-30",
    data_path: str = "data/sp500.csv"
) -> Tuple[np.ndarray, int]:
    """
    Load S&P 500 data covering COVID-19 crash (Section 5.3 in paper)
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date
        data_path: Path to CSV file
        
    Returns:
        time_series: Daily close prices
        crash_point: March 2020 crash onset (approx. day 50)
    """
    try:
        import pandas as pd
        df = pd.read_csv(data_path, parse_dates=['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        prices = df['Close'].values
        crash_point = 50  # Approximate: early March
        return prices, crash_point
    except (FileNotFoundError, ImportError):
        print(f"WARNING: {data_path} not found. Generating synthetic crisis.")
        # Synthetic crash: stable + sudden drop + recovery
        T = 120
        pre_crash = np.random.normal(3000, 50, 50)
        crash = np.linspace(3000, 2200, 10)
        post_crash = np.random.normal(2400, 100, T - 60)
        return np.concatenate([pre_crash, crash, post_crash]), 50


def load_bitcoin_crash(
    start_date: str = "2021-04-01",
    end_date: str = "2021-07-31",
    data_path: str = "data/bitcoin.csv"
) -> Tuple[np.ndarray, int]:
    """
    Load Bitcoin hourly prices covering May 2021 crash
    
    Args:
        start_date: Start date
        end_date: End date
        data_path: Path to CSV file
        
    Returns:
        time_series: Hourly BTC/USD prices
        crash_point: May 19 crash (approx. hour 1200)
    """
    try:
        import pandas as pd
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        prices = df['close'].values
        crash_point = 1200  # Approximate
        return prices, crash_point
    except (FileNotFoundError, ImportError):
        print(f"WARNING: {data_path} not found. Generating synthetic crash.")
        T = 2200
        pre = np.random.normal(60000, 2000, 1200)
        crash = np.linspace(60000, 30000, 24)
        post = np.random.normal(35000, 3000, T - 1224)
        return np.concatenate([pre, crash, post]), 1200


# ==================== Dataset Registry ====================

DATASETS = {
    'SinFreq': generate_sin_freq_shift,
    'StepMean': generate_step_mean_shift,
    'GradualDrift': generate_gradual_drift,
}


def get_dataset(name: str, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Unified dataset loader
    
    Args:
        name: Dataset name (SinFreq, StepMean, etc.)
        **kwargs: Dataset-specific arguments
        
    Returns:
        time_series: Data
        shift_point: Known regime shift location
    """
    if name in DATASETS:
        return DATASETS[name](**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
