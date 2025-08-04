"""
Validation module for downloaded ERA5 data
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.config_loader import get_download_paths_and_settings
from download_utils import setup_logging, get_file_size_mb

# Load configuration
config_vars = get_download_paths_and_settings()
ERA5_DIR = config_vars['ERA5_DIR']
VALIDATION_CONFIG = config_vars['VALIDATION_CONFIG']
CHINA_BOUNDS = config_vars['CHINA_BOUNDS']
YEARS = config_vars['YEARS']

try:
    import xarray as xr
    import netCDF4
except ImportError:
    logging.error("Required packages not installed. Please run: pip install xarray netCDF4")
    sys.exit(1)


class ERA5Validator:
    """Validate downloaded ERA5 NetCDF files"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.validation_config = VALIDATION_CONFIG
    
    def validate_file_size(self, file_path: Path) -> Tuple[bool, str]:
        """Check if file size is within expected range"""
        size_mb = get_file_size_mb(file_path)
        min_size = self.validation_config['min_file_size_mb']
        max_size = self.validation_config['max_file_size_mb']
        
        if size_mb < min_size:
            return False, f"File too small: {size_mb:.1f}MB (expected >{min_size}MB)"
        elif size_mb > max_size:
            return False, f"File too large: {size_mb:.1f}MB (expected <{max_size}MB)"
        
        return True, f"File size OK: {size_mb:.1f}MB"
    
    def validate_netcdf_structure(self, file_path: Path) -> Tuple[bool, str]:
        """Validate NetCDF file structure and variables"""
        try:
            with xr.open_dataset(file_path) as ds:
                # Check required variables
                required_vars = self.validation_config['required_variables']
                missing_vars = [v for v in required_vars if v not in ds.variables]
                
                if missing_vars:
                    return False, f"Missing required variables: {missing_vars}"
                
                # Check dimensions
                expected_dims = ['time', 'latitude', 'longitude']
                missing_dims = [d for d in expected_dims if d not in ds.dims]
                
                if missing_dims:
                    return False, f"Missing dimensions: {missing_dims}"
                
                return True, "NetCDF structure valid"
                
        except Exception as e:
            return False, f"Failed to open NetCDF: {str(e)}"
    
    def validate_spatial_coverage(self, file_path: Path) -> Tuple[bool, str]:
        """Check if spatial coverage matches China bounds"""
        try:
            with xr.open_dataset(file_path) as ds:
                lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
                lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
                
                # Check if bounds approximately match (with some tolerance)
                tolerance = 1.0  # degree
                
                if abs(lat_min - CHINA_BOUNDS['south']) > tolerance:
                    return False, f"Southern bound mismatch: {lat_min} vs {CHINA_BOUNDS['south']}"
                if abs(lat_max - CHINA_BOUNDS['north']) > tolerance:
                    return False, f"Northern bound mismatch: {lat_max} vs {CHINA_BOUNDS['north']}"
                if abs(lon_min - CHINA_BOUNDS['west']) > tolerance:
                    return False, f"Western bound mismatch: {lon_min} vs {CHINA_BOUNDS['west']}"
                if abs(lon_max - CHINA_BOUNDS['east']) > tolerance:
                    return False, f"Eastern bound mismatch: {lon_max} vs {CHINA_BOUNDS['east']}"
                
                return True, f"Spatial coverage OK: [{lat_min:.1f}-{lat_max:.1f}°N, {lon_min:.1f}-{lon_max:.1f}°E]"
                
        except Exception as e:
            return False, f"Failed to check spatial coverage: {str(e)}"
    
    def validate_temporal_coverage(self, file_path: Path, year: int) -> Tuple[bool, str]:
        """Check temporal coverage for the year"""
        try:
            with xr.open_dataset(file_path) as ds:
                times = ds.time.values
                
                # Check if we have hourly data
                expected_hours = 365 * 24  # or 366 for leap years
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    expected_hours = 366 * 24
                
                actual_hours = len(times)
                
                if actual_hours < expected_hours * 0.95:  # Allow 5% missing data
                    return False, f"Insufficient temporal coverage: {actual_hours}/{expected_hours} hours"
                
                # Check year matches
                years_in_data = np.unique([pd.Timestamp(t).year for t in times])
                if len(years_in_data) != 1 or years_in_data[0] != year:
                    return False, f"Year mismatch: expected {year}, found {years_in_data}"
                
                return True, f"Temporal coverage OK: {actual_hours} hours for year {year}"
                
        except Exception as e:
            return False, f"Failed to check temporal coverage: {str(e)}"
    
    def validate_data_quality(self, file_path: Path) -> Tuple[bool, str]:
        """Check data quality and reasonable value ranges"""
        try:
            with xr.open_dataset(file_path) as ds:
                results = []
                
                # Check 2m temperature
                if 't2m' in ds.variables:
                    temp_data = ds['t2m']
                    # Check for NaN values
                    nan_count = np.isnan(temp_data.values).sum()
                    total_count = temp_data.size
                    nan_percentage = (nan_count / total_count) * 100
                    
                    if nan_percentage > 5:
                        return False, f"Too many NaN values in t2m: {nan_percentage:.1f}%"
                    
                    # Check temperature range (in Kelvin)
                    temp_min = float(np.nanmin(temp_data.values))
                    temp_max = float(np.nanmax(temp_data.values))
                    
                    min_allowed = self.validation_config['temp_min_kelvin']
                    max_allowed = self.validation_config['temp_max_kelvin']
                    
                    if temp_min < min_allowed or temp_max > max_allowed:
                        return False, f"Temperature out of range: [{temp_min:.1f}-{temp_max:.1f}]K"
                    
                    results.append(f"t2m: [{temp_min:.1f}-{temp_max:.1f}]K, NaN: {nan_percentage:.2f}%")
                
                # Check 2m dewpoint temperature
                if 'd2m' in ds.variables:
                    dewpoint_data = ds['d2m']
                    # Check for NaN values
                    nan_count = np.isnan(dewpoint_data.values).sum()
                    total_count = dewpoint_data.size
                    nan_percentage = (nan_count / total_count) * 100
                    
                    if nan_percentage > 5:
                        return False, f"Too many NaN values in d2m: {nan_percentage:.1f}%"
                    
                    # Check dewpoint range (should be <= temperature)
                    dew_min = float(np.nanmin(dewpoint_data.values))
                    dew_max = float(np.nanmax(dewpoint_data.values))
                    
                    if dew_min < min_allowed or dew_max > max_allowed:
                        return False, f"Dewpoint out of range: [{dew_min:.1f}-{dew_max:.1f}]K"
                    
                    results.append(f"d2m: [{dew_min:.1f}-{dew_max:.1f}]K, NaN: {nan_percentage:.2f}%")
                
                if not results:
                    return False, "No temperature variables found"
                
                return True, f"Data quality OK - {'; '.join(results)}"
                
        except Exception as e:
            return False, f"Failed to check data quality: {str(e)}"
    
    def validate_single_file(self, file_path: Path, year: int) -> Dict[str, Tuple[bool, str]]:
        """Run all validations on a single file"""
        self.logger.info(f"Validating {file_path.name}...")
        
        validations = {
            'file_size': self.validate_file_size(file_path),
            'structure': self.validate_netcdf_structure(file_path),
            'spatial': self.validate_spatial_coverage(file_path),
            'temporal': self.validate_temporal_coverage(file_path, year),
            'quality': self.validate_data_quality(file_path)
        }
        
        all_passed = all(v[0] for v in validations.values())
        
        if all_passed:
            self.logger.info(f"✓ {file_path.name} - All validations passed")
        else:
            self.logger.warning(f"✗ {file_path.name} - Some validations failed")
            for check, (passed, msg) in validations.items():
                if not passed:
                    self.logger.warning(f"  - {check}: {msg}")
        
        return validations
    
    def validate_all_files(self, years: List[int] = None) -> Dict[str, Dict]:
        """Validate all ERA5 files"""
        if years is None:
            years = YEARS
        
        results = {}
        files_found = 0
        files_valid = 0
        
        self.logger.info(f"Starting validation of ERA5 files for years {years[0]}-{years[-1]}")
        
        for year in years:
            file_path = ERA5_DIR / f"t2m_{year}.nc"
            
            if file_path.exists():
                files_found += 1
                validations = self.validate_single_file(file_path, year)
                results[str(year)] = validations
                
                if all(v[0] for v in validations.values()):
                    files_valid += 1
            else:
                self.logger.warning(f"File not found: {file_path.name}")
                results[str(year)] = {'missing': (False, 'File not found')}
        
        # Summary
        self.logger.info("\n" + "="*50)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Files found: {files_found}/{len(years)}")
        self.logger.info(f"Files valid: {files_valid}/{files_found}")
        
        if files_found < len(years):
            missing_years = [y for y in years if not (ERA5_DIR / f"t2m_{y}.nc").exists()]
            self.logger.warning(f"Missing years: {missing_years}")
        
        return results


def main():
    """Main function for standalone execution"""
    validator = ERA5Validator()
    results = validator.validate_all_files()
    
    # Return success if all files are valid
    all_valid = all(
        all(v[0] for v in year_results.values() if isinstance(v, tuple))
        for year_results in results.values()
        if 'missing' not in year_results
    )
    
    return all_valid


if __name__ == "__main__":
    # Import pandas for timestamp handling
    try:
        import pandas as pd
    except ImportError:
        logging.error("pandas not installed. Please run: pip install pandas")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1)