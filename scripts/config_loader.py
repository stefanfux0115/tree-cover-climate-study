"""
Configuration loader utility for JSON config files
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_name: Name of config file (without .json extension)
        
    Returns:
        Dictionary containing configuration
    """
    config_path = CONFIG_DIR / f"{config_name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_download_config() -> Dict[str, Any]:
    """Load download configuration"""
    return load_config("download-config")

def load_data_processing_config() -> Dict[str, Any]:
    """Load data processing configuration"""
    return load_config("data-processing")

def load_ml_analysis_config() -> Dict[str, Any]:
    """Load ML analysis configuration"""
    return load_config("ml-analysis")

# Backward compatibility - recreate the old config structure
def get_download_paths_and_settings():
    """
    Recreate the old download_config.py structure for backward compatibility
    """
    config = load_download_config()
    
    # Base paths (recreate the old structure)
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Dataset directories
    ERA5_DIR = RAW_DATA_DIR / "era5"
    TREE_DIR = RAW_DATA_DIR / "tree_change_30m"
    ELEC_DIR = RAW_DATA_DIR / "elec_1km"
    
    # Ensure directories exist
    for dir_path in [ERA5_DIR, TREE_DIR, ELEC_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Extract settings from JSON config
    CHINA_BOUNDS = config["china_bounds"]
    temporal_range = config["temporal_range"]
    YEARS = temporal_range["years"]
    START_YEAR = temporal_range["start_year"]
    END_YEAR = temporal_range["end_year"]
    
    # ERA5 configuration with area bounds included
    ERA5_CONFIG = config["era5_config"].copy()
    ERA5_CONFIG["area"] = [
        CHINA_BOUNDS["north"], CHINA_BOUNDS["west"],
        CHINA_BOUNDS["south"], CHINA_BOUNDS["east"]
    ]
    
    DOWNLOAD_CONFIG = config["download_settings"]
    VALIDATION_CONFIG = config["validation"]
    CDS_CONFIG = config["cds_api"]
    SYSTEM_REQUIREMENTS = config["system_requirements"]
    # Fix python_version format (convert list to tuple)
    if 'python_version' in SYSTEM_REQUIREMENTS and isinstance(SYSTEM_REQUIREMENTS['python_version'], list):
        SYSTEM_REQUIREMENTS['python_version'] = tuple(SYSTEM_REQUIREMENTS['python_version'])
    
    # Create log config with dynamic filename
    from datetime import datetime
    LOG_CONFIG = {
        'filename': LOGS_DIR / f'download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    return {
        'ERA5_DIR': ERA5_DIR,
        'TREE_DIR': TREE_DIR,
        'ELEC_DIR': ELEC_DIR,
        'PROCESSED_DATA_DIR': PROCESSED_DATA_DIR,
        'LOGS_DIR': LOGS_DIR,
        'CHINA_BOUNDS': CHINA_BOUNDS,
        'YEARS': YEARS,
        'START_YEAR': START_YEAR,
        'END_YEAR': END_YEAR,
        'ERA5_CONFIG': ERA5_CONFIG,
        'DOWNLOAD_CONFIG': DOWNLOAD_CONFIG,
        'VALIDATION_CONFIG': VALIDATION_CONFIG,
        'CDS_CONFIG': CDS_CONFIG,
        'SYSTEM_REQUIREMENTS': SYSTEM_REQUIREMENTS,
        'LOG_CONFIG': LOG_CONFIG
    }

def get_data_processing_settings():
    """
    Get data processing settings in the expected format
    """
    config = load_data_processing_config()
    
    # Extract temperature bins in the old format
    temp_bins = config["temperature_bins"]
    TEMPERATURE_BINS = {
        "bin_width": 5,
        "bins": temp_bins
    }
    
    # Extract other settings
    CANOPY_THRESHOLDS = {
        "event_threshold": config["processing_settings"]["canopy_event_threshold"]
    }
    
    return {
        'TEMPERATURE_BINS': TEMPERATURE_BINS,
        'CANOPY_THRESHOLDS': CANOPY_THRESHOLDS,
        'HUMIDITY_BINS': config["humidity_bins"],
        'PRECIPITATION_BINS': config["precipitation_bins"],
        'SOCIOECONOMIC_BINS': config["socioeconomic_bins"],
        'PROCESSING_SETTINGS': config["processing_settings"],
        'DATA_INTEGRATION': config["data_integration"]
    }