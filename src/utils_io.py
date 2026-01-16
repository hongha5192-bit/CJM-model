"""
I/O utilities for the CJM project
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging


def load_config(config_path="configs/daily.yaml"):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data, filepath):
    """Save data to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logging.info(f"Saved JSON to {filepath}")


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_parquet(df, filepath):
    """Save DataFrame to parquet file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logging.info(f"Saved parquet to {filepath}")


def load_parquet(filepath):
    """Load parquet file"""
    return pd.read_parquet(filepath)


def save_csv(df, filepath):
    """Save DataFrame to CSV file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logging.info(f"Saved CSV to {filepath}")


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def setup_logging(name=None, level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if name:
        return logging.getLogger(name)
    return logging.getLogger()