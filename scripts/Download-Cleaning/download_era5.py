"""
ERA5-Land temperature data download module
Supports both yearly and monthly downloads to handle large data requests
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.config_loader import get_download_paths_and_settings
from download_utils import (
    setup_logging, retry_with_backoff, validate_file_exists,
    format_eta, get_file_size_mb, check_cds_credentials
)

# Load configuration
config_vars = get_download_paths_and_settings()
ERA5_DIR = config_vars['ERA5_DIR']
ERA5_CONFIG = config_vars['ERA5_CONFIG']
YEARS = config_vars['YEARS']
DOWNLOAD_CONFIG = config_vars['DOWNLOAD_CONFIG']
CDS_CONFIG = config_vars['CDS_CONFIG']

try:
    import cdsapi
    import xarray as xr
except ImportError:
    logging.error("Required packages not installed. Please run: pip install cdsapi xarray")
    sys.exit(1)


class ERA5Downloader:
    """Handle ERA5 data downloads with resume capability"""
    
    def __init__(self, monthly_mode: bool = False):
        self.logger = setup_logging()
        self.client = None
        self.monthly_mode = monthly_mode
        self.setup_cds_client()
    
    def setup_cds_client(self):
        """Initialize CDS API client"""
        if not check_cds_credentials():
            raise RuntimeError("CDS API credentials not configured")
        
        try:
            self.client = cdsapi.Client()
            self.logger.info("CDS API client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            raise
    
    def get_output_filename(self, year: int, month: Optional[int] = None) -> Path:
        """Generate output filename for a given year or month"""
        if month is not None:
            return ERA5_DIR / f"t2m_{year}_{month:02d}.nc"
        return ERA5_DIR / f"t2m_{year}.nc"
    
    def check_existing_file(self, year: int) -> bool:
        """Check if file for year already exists and is valid"""
        file_path = self.get_output_filename(year)
        
        if validate_file_exists(file_path, min_size_mb=1000):
            size_mb = get_file_size_mb(file_path)
            self.logger.info(f"File for {year} already exists ({size_mb:.1f}MB). Skipping download.")
            return True
        
        return False
    
    def download_month(self, year: int, month: int) -> bool:
        """Download ERA5 data for a single month"""
        output_file = self.get_output_filename(year, month)
        
        # Check if already exists
        if validate_file_exists(output_file, min_size_mb=50):
            size_mb = get_file_size_mb(output_file)
            self.logger.info(f"File for {year}-{month:02d} already exists ({size_mb:.1f}MB). Skipping.")
            return True
        
        tmp_file = output_file.with_suffix('.nc.tmp')
        
        self.logger.info(f"Downloading {year}-{month:02d}...")
        
        # Prepare request parameters for one month
        request_params = {
            'variable': ERA5_CONFIG['variable'],
            'year': str(year),
            'month': f'{month:02d}',
            'day': ERA5_CONFIG['day'],
            'time': ERA5_CONFIG['time'],
            'area': ERA5_CONFIG['area'],
            'format': ERA5_CONFIG['format'],
            'product_type': ERA5_CONFIG['product_type']
        }
        
        try:
            start_time = time.time()
            
            def download_with_cds():
                return self.client.retrieve(
                    CDS_CONFIG['dataset'],
                    request_params,
                    str(tmp_file)
                )
            
            # Download with retry logic
            retry_with_backoff(download_with_cds)
            
            # Move temp file to final location
            tmp_file.rename(output_file)
            
            elapsed = time.time() - start_time
            size_mb = get_file_size_mb(output_file)
            
            self.logger.info(f"Downloaded {year}-{month:02d}: {size_mb:.1f}MB in {format_eta(elapsed)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {year}-{month:02d}: {e}")
            if tmp_file.exists():
                tmp_file.unlink()
            return False
    
    def merge_monthly_files(self, year: int) -> bool:
        """Merge monthly files into yearly file"""
        yearly_file = self.get_output_filename(year)
        
        # Check if yearly file already exists
        if validate_file_exists(yearly_file, min_size_mb=1000):
            self.logger.info(f"Yearly file for {year} already exists. Skipping merge.")
            return True
        
        self.logger.info(f"Merging monthly files for year {year}...")
        
        try:
            # Find all monthly files for this year
            monthly_files = []
            for month in range(1, 13):
                month_file = self.get_output_filename(year, month)
                if month_file.exists():
                    monthly_files.append(month_file)
            
            if len(monthly_files) < 12:
                self.logger.warning(f"Only {len(monthly_files)}/12 monthly files found for {year}")
                return False
            
            # Open and concatenate all monthly files
            datasets = []
            for f in sorted(monthly_files):
                ds = xr.open_dataset(f)
                datasets.append(ds)
            
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            
            # Sort by time
            combined = combined.sortby('time')
            
            # Save to yearly file
            combined.to_netcdf(yearly_file)
            
            # Close datasets
            for ds in datasets:
                ds.close()
            combined.close()
            
            size_mb = get_file_size_mb(yearly_file)
            self.logger.info(f"Successfully merged {year} data: {size_mb:.1f}MB")
            
            # Optionally remove monthly files after successful merge
            # for f in monthly_files:
            #     f.unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to merge files for {year}: {e}")
            return False
    
    def download_year(self, year: int) -> bool:
        """Download ERA5 data for a single year"""
        if self.monthly_mode:
            # Download month by month
            self.logger.info(f"Downloading year {year} month by month...")
            
            successful_months = []
            failed_months = []
            
            for month in range(1, 13):
                if self.download_month(year, month):
                    successful_months.append(month)
                else:
                    failed_months.append(month)
                
                # Rate limiting between requests
                if month < 12:
                    time.sleep(DOWNLOAD_CONFIG['rate_limit_delay'])
            
            self.logger.info(f"Year {year}: {len(successful_months)}/12 months downloaded successfully")
            
            if failed_months:
                self.logger.warning(f"Failed months: {failed_months}")
                return False
            
            # Merge monthly files into yearly file
            return self.merge_monthly_files(year)
        else:
            # Try to download full year (original method)
            if self.check_existing_file(year):
                return True
            
            output_file = self.get_output_filename(year)
            tmp_file = output_file.with_suffix('.nc.tmp')
            
            self.logger.info(f"Starting download for year {year}...")
            
            # Prepare request parameters
            request_params = {
                'variable': ERA5_CONFIG['variable'],
                'year': str(year),
                'month': ERA5_CONFIG['month'],
                'day': ERA5_CONFIG['day'],
                'time': ERA5_CONFIG['time'],
                'area': ERA5_CONFIG['area'],
                'format': ERA5_CONFIG['format'],
                'product_type': ERA5_CONFIG['product_type']
            }
        
        try:
            # Use temporary file during download
            start_time = time.time()
            
            def download_with_cds():
                return self.client.retrieve(
                    CDS_CONFIG['dataset'],
                    request_params,
                    str(tmp_file)
                )
            
            # Download with retry logic
            retry_with_backoff(download_with_cds)
            
            # Move temp file to final location
            tmp_file.rename(output_file)
            
            elapsed = time.time() - start_time
            size_mb = get_file_size_mb(output_file)
            
            self.logger.info(
                f"Successfully downloaded {year}: {size_mb:.1f}MB in {format_eta(elapsed)}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {year}: {e}")
            # Clean up temporary file if exists
            if tmp_file.exists():
                tmp_file.unlink()
            return False
    
    def download_all_years(self, years: List[int] = None) -> Dict[str, List[int]]:
        """Download ERA5 data for all specified years"""
        if years is None:
            years = YEARS
        
        successful = []
        failed = []
        
        self.logger.info(f"Starting ERA5 downloads for {len(years)} years: {years[0]}-{years[-1]}")
        
        for i, year in enumerate(years):
            self.logger.info(f"\nProgress: {i+1}/{len(years)} years")
            
            if self.download_year(year):
                successful.append(year)
            else:
                failed.append(year)
            
            # Rate limiting between requests
            if i < len(years) - 1:
                time.sleep(DOWNLOAD_CONFIG['rate_limit_delay'])
        
        return {'successful': successful, 'failed': failed}
    
    def estimate_total_download_size(self, years: List[int] = None) -> float:
        """Estimate total download size in GB"""
        if years is None:
            years = YEARS
        
        # Filter out already downloaded years
        years_to_download = [y for y in years if not self.check_existing_file(y)]
        
        # Estimate ~3GB per year (for 2 variables)
        estimated_gb = len(years_to_download) * 3.0
        
        self.logger.info(f"Estimated download size: {estimated_gb:.1f}GB for {len(years_to_download)} years")
        return estimated_gb


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ERA5-Land data")
    parser.add_argument('--monthly', action='store_true', help='Download month by month')
    parser.add_argument('--years', nargs='+', type=int, help='Years to download')
    
    args = parser.parse_args()
    
    downloader = ERA5Downloader(monthly_mode=args.monthly)
    
    # Estimate download size
    years = args.years if args.years else None
    downloader.estimate_total_download_size(years)
    
    # Start downloads
    start_time = time.time()
    results = downloader.download_all_years(years)
    
    # Print summary
    from scripts.download_utils import print_download_summary
    print_download_summary(
        [f"Year {y}" for y in results['successful']],
        [f"Year {y}" for y in results['failed']],
        start_time
    )
    
    return len(results['failed']) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)