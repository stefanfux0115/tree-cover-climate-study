"""
Master download orchestrator for ERA5 and tree cover data pipeline
"""
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.config_loader import get_download_paths_and_settings
from download_utils import (
    setup_logging, check_disk_space, check_python_version,
    clean_partial_downloads, print_download_summary
)
from download_era5 import ERA5Downloader
from validate_downloads import ERA5Validator
from download_zenodo import ZenodoDownloader

# Load configuration
config_vars = get_download_paths_and_settings()
YEARS = config_vars['YEARS']
SYSTEM_REQUIREMENTS = config_vars['SYSTEM_REQUIREMENTS']
ERA5_DIR = config_vars['ERA5_DIR']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land temperature data and tree cover data for China (2012-2019)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data (ERA5 and tree cover) for all years (2012-2019)
  python download_master.py --all
  
  # Download only ERA5 data
  python download_master.py --era5
  
  # Download only tree cover data
  python download_master.py --tree-cover
  
  # Download specific years
  python download_master.py --years 2015 2016 2017 --all
  
  # Download month by month (for large ERA5 requests)
  python download_master.py --monthly --years 2019 --era5
  
  # Validate existing downloads only
  python download_master.py --validate-only
  
  # Force re-download even if files exist
  python download_master.py --force --all
  
  # Specify custom year range
  python download_master.py --start-year 2014 --end-year 2016 --all
        """
    )
    
    parser.add_argument(
        '--years', 
        nargs='+', 
        type=int,
        help='Specific years to download (e.g., --years 2015 2016 2017)'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2012,
        help='Start year for download range (default: 2012)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2019,
        help='End year for download range (default: 2019)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing files without downloading'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean partial downloads before starting'
    )
    
    parser.add_argument(
        '--monthly',
        action='store_true',
        help='Download ERA5 data month by month (use for large requests)'
    )
    
    # Data source selection
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download both ERA5 and tree cover data'
    )
    
    parser.add_argument(
        '--era5',
        action='store_true',
        help='Download only ERA5 temperature data'
    )
    
    parser.add_argument(
        '--tree-cover',
        action='store_true',
        help='Download only tree cover data from Zenodo'
    )
    
    parser.add_argument(
        '--zenodo-token',
        type=str,
        default='wnxoFG6QwvjKqe9KpqkTy3GSDsjnZxoHHOglHozG930aub6hFLNLTFgbLzfH',
        help='Zenodo API token for downloading tree cover data'
    )
    
    return parser.parse_args()


def pre_flight_checks(logger):
    """Run system checks before starting downloads"""
    logger.info("Running pre-flight checks...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check disk space
    if not check_disk_space(Path.cwd(), SYSTEM_REQUIREMENTS['min_disk_space_gb']):
        return False
    
    logger.info("All pre-flight checks passed")
    return True




def main():
    """Main orchestrator function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("China Urban Analysis Data Download Pipeline")
    logger.info("="*60)
    
    # Determine years to process
    if args.years:
        years = sorted(args.years)
    else:
        years = list(range(args.start_year, args.end_year + 1))
    
    # Validate year range
    valid_years = [y for y in years if 2000 <= y <= 2030]
    if len(valid_years) != len(years):
        invalid = [y for y in years if y not in valid_years]
        logger.error(f"Invalid years specified: {invalid}")
        return False
    
    logger.info(f"Target years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Determine what to download
    download_era5 = args.all or args.era5 or (not args.tree_cover)
    download_tree = args.all or args.tree_cover
    
    if not download_era5 and not download_tree:
        logger.error("No data source specified. Use --all, --era5, or --tree-cover")
        return False
    
    logger.info("Data sources to download:")
    if download_era5:
        logger.info("  - ERA5 temperature data")
    if download_tree:
        logger.info("  - Tree cover data from Zenodo")
    
    # Run pre-flight checks
    if not pre_flight_checks(logger):
        logger.error("Pre-flight checks failed. Exiting.")
        return False
    
    # Clean partial downloads if requested
    if args.clean:
        logger.info("Cleaning partial downloads...")
        clean_partial_downloads(ERA5_DIR, "*.tmp")
    
    # Validation only mode (currently only for ERA5)
    if args.validate_only:
        if download_era5:
            logger.info("Running validation only mode for ERA5...")
            validator = ERA5Validator()
            results = validator.validate_all_files(years)
            return all(
                all(v[0] for v in year_results.values() if isinstance(v, tuple))
                for year_results in results.values()
                if 'missing' not in year_results
            )
        else:
            logger.warning("Validation only mode is currently only supported for ERA5 data")
            return True
    
    # Track overall success
    overall_success = True
    
    # Download ERA5 data if requested
    if download_era5:
        logger.info("\n" + "="*60)
        logger.info("ERA5 Temperature Data Download")
        logger.info("="*60)
        
        # Force mode - remove existing files
        if args.force:
            logger.warning("Force mode enabled - existing ERA5 files will be overwritten")
            for year in years:
                file_path = ERA5_DIR / f"t2m_{year}.nc"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed existing file: {file_path.name}")
        
        # Initialize downloader
        try:
            downloader = ERA5Downloader(monthly_mode=args.monthly)
        except Exception as e:
            logger.error(f"Failed to initialize ERA5 downloader: {e}")
            overall_success = False
        else:
            # Log download mode
            if args.monthly:
                logger.info("Monthly download mode enabled - will download data month by month")
            
            # Estimate download size
            estimated_gb = downloader.estimate_total_download_size(years)
            
            if estimated_gb > 0:
                logger.info(f"\nEstimated ERA5 download: ~{estimated_gb:.1f}GB")
                logger.info("Starting ERA5 downloads in 5 seconds... (Press Ctrl+C to cancel)")
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    logger.info("\nDownload cancelled by user")
                    return False
            
            # Start downloads
            start_time = time.time()
            download_results = downloader.download_all_years(years)
            
            # Print download summary
            print_download_summary(
                [f"Year {y}" for y in download_results['successful']],
                [f"Year {y}" for y in download_results['failed']],
                start_time
            )
            
            # Run validation on downloaded files
            if download_results['successful']:
                logger.info("\nRunning validation on downloaded ERA5 files...")
                validator = ERA5Validator()
                validation_results = validator.validate_all_files(download_results['successful'])
                
                # Check if all downloaded files are valid
                all_valid = all(
                    all(v[0] for v in year_results.values() if isinstance(v, tuple))
                    for year in download_results['successful']
                    for year_results in [validation_results.get(str(year), {})]
                    if year_results
                )
                
                if not all_valid:
                    logger.warning("Some downloaded ERA5 files failed validation checks")
                    overall_success = False
            
            # Check if any downloads failed
            if download_results['failed']:
                logger.error(f"\nFailed to download {len(download_results['failed'])} ERA5 files")
                overall_success = False
            else:
                logger.info("\n✓ All ERA5 downloads completed successfully!")
                logger.info(f"ERA5 data location: {ERA5_DIR}")
    
    # Download tree cover data if requested
    if download_tree:
        try:
            zenodo_downloader = ZenodoDownloader(api_token=args.zenodo_token)
            
            # Estimate download size
            estimated_gb = zenodo_downloader.estimate_download_size(years)
            if estimated_gb > 0:
                logger.info(f"\nEstimated tree cover download: ~{estimated_gb:.1f}GB")
                logger.info("Starting tree cover downloads in 5 seconds... (Press Ctrl+C to cancel)")
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    logger.info("\nDownload cancelled by user")
                    return False
            
            # Download tree cover data
            tree_results = zenodo_downloader.download_all_records(years)
            
            if tree_results['failed_files'] > 0:
                logger.error(f"Failed to download {tree_results['failed_files']} tree cover files")
                overall_success = False
            else:
                logger.info("\n✓ All tree cover downloads completed successfully!")
                
                # Validate downloads
                logger.info("\nValidating tree cover downloads...")
                validation_results = zenodo_downloader.validate_downloads(years)
                logger.info(f"Valid files: {validation_results['valid_files']}")
                logger.info(f"Missing files: {validation_results['missing_files']}")
                logger.info(f"Corrupt files: {validation_results['corrupt_files']}")
                
                if validation_results['missing_files'] > 0 or validation_results['corrupt_files'] > 0:
                    logger.warning("Some tree cover files need to be re-downloaded")
                    overall_success = False
                    
        except Exception as e:
            logger.error(f"Failed to download tree cover data: {e}")
            overall_success = False
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Download Pipeline Summary")
    logger.info("="*60)
    
    if overall_success:
        logger.info("✓ All requested downloads completed successfully!")
        logger.info("\nData locations:")
        if download_era5:
            logger.info(f"  - ERA5 data: {ERA5_DIR}")
        if download_tree:
            logger.info(f"  - Tree cover data: data/raw/tree_cover_zenodo/")
        logger.info("\nNext steps:")
        logger.info("1. Verify the downloaded files")
        logger.info("2. Proceed with data processing using unified_data_processor.py")
    else:
        logger.error("Some downloads failed. Please check the logs above.")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)