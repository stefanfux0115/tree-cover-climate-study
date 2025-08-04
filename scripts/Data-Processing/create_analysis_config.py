"""
Create configuration for the analysis pipeline.

This script sets up the configuration parameters for:
- Temperature binning
- Grid specifications
- Time periods
- File paths
"""

import os
import json

def create_analysis_config():
    """Create analysis configuration dictionary."""
    
    config = {
        "project": {
            "name": "Urban Tree Canopy Effects on Electricity Consumption",
            "description": "Analysis of tree canopy cooling effects on electricity demand in 280 Chinese cities",
            "base_dir": r"G:\research\junmao-tang\code-test\code-space"
        },
        
        "data_sources": {
            "electricity": {
                "path": "data/raw/elec_1km/China_1km_Ele_201204_201912",
                "format": "tif",
                "resolution": "1km",
                "temporal": "monthly",
                "period": "2012-04 to 2019-12"
            },
            "tree_canopy": {
                "path": "data/raw/tree_change_30m",
                "format": "zip/tif",
                "resolution": "30m",
                "temporal": "yearly",
                "years": [2015, 2016, 2017, 2018, 2019]
            },
            "temperature": {
                "path": "data/raw/era5",
                "format": "nc",
                "resolution": "0.25 degree",
                "temporal": "hourly",
                "period": "2012-2019"
            },
            "city_boundaries": {
                "path": "data/shp",
                "format": "shp",
                "admin_levels": ["adm0", "adm1", "adm2"]
            }
        },
        
        "temperature_bins": {
            "method": "uniform",
            "bin_width": 5,  # degrees Celsius
            "bins": [
                {"name": "bin_10_15", "min": 10, "max": 15, "label": "10-15°C"},
                {"name": "bin_15_20", "min": 15, "max": 20, "label": "15-20°C"},
                {"name": "bin_20_25", "min": 20, "max": 25, "label": "20-25°C (reference)"},
                {"name": "bin_25_30", "min": 25, "max": 30, "label": "25-30°C"},
                {"name": "bin_30_35", "min": 30, "max": 35, "label": "30-35°C"},
                {"name": "bin_35_40", "min": 35, "max": 40, "label": "35-40°C"},
                {"name": "bin_40_plus", "min": 40, "max": 50, "label": "40°C+"}
            ],
            "reference_bin": "bin_20_25"
        },
        
        "canopy_thresholds": {
            "event_threshold": 5,  # percentage points
            "min_canopy": 0,
            "max_canopy": 100
        },
        
        "grid_specifications": {
            "resolution": 1000,  # meters
            "crs": "EPSG:4326",
            "extent": {
                "description": "To be determined from electricity rasters"
            }
        },
        
        "panel_structure": {
            "spatial_unit": "1km grid",
            "temporal_unit": "month",
            "panel_vars": [
                "grid_id",
                "year",
                "month",
                "city_id",
                "city_name",
                "province",
                "ln_elec_kwh",
                "elec_kwh",
                "canopy_share",
                "days_in_bin_10_15",
                "days_in_bin_15_20",
                "days_in_bin_20_25",
                "days_in_bin_25_30",
                "days_in_bin_30_35",
                "days_in_bin_35_40",
                "days_in_bin_40_plus"
            ]
        },
        
        "regression_specifications": {
            "baseline": {
                "dependent": "ln_elec_kwh",
                "independent": [
                    "days_in_bin_*",
                    "canopy_share",
                    "days_in_bin_* × canopy_share"
                ],
                "fixed_effects": ["grid_id", "year_month"],
                "cluster": "city_id"
            },
            "event_study": {
                "treatment": "canopy_event",
                "window": [-3, 5],
                "reference_period": -1
            }
        },
        
        "output_paths": {
            "processed": {
                "grid_monthly": "data/processed/grid_monthly",
                "temperature_daily": "data/processed/temperature_daily",
                "canopy_yearly": "data/processed/canopy_yearly",
                "spatial": "data/processed/spatial",
                "analysis_ready": "data/processed/analysis_ready"
            },
            "results": {
                "regressions": "data/results/regressions",
                "figures": "data/results/figures",
                "tables": "data/results/tables"
            }
        }
    }
    
    return config

def save_config(config, base_dir):
    """Save configuration to JSON file."""
    config_dir = os.path.join(base_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "analysis_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to: {config_path}")
    
    # Also create a simplified config for quick reference
    simple_config = {
        "temperature_bins": config["temperature_bins"]["bins"],
        "canopy_event_threshold": config["canopy_thresholds"]["event_threshold"],
        "panel_vars": config["panel_structure"]["panel_vars"],
        "years": list(range(2012, 2020))
    }
    
    simple_config_path = os.path.join(config_dir, "analysis_params.json")
    with open(simple_config_path, 'w') as f:
        json.dump(simple_config, f, indent=4)
    
    print(f"Simple parameters saved to: {simple_config_path}")

def main():
    # Create configuration
    config = create_analysis_config()
    
    # Save to file
    base_dir = config["project"]["base_dir"]
    save_config(config, base_dir)
    
    print("\nAnalysis configuration created successfully!")
    print("\nKey parameters:")
    print(f"- Temperature bins: {len(config['temperature_bins']['bins'])}")
    print(f"- Bin width: {config['temperature_bins']['bin_width']}°C")
    print(f"- Canopy event threshold: {config['canopy_thresholds']['event_threshold']}pp")
    print(f"- Panel variables: {len(config['panel_structure']['panel_vars'])}")

if __name__ == "__main__":
    main()