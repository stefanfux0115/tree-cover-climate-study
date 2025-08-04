"""
Shared utilities for download operations
"""
import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.config_loader import get_download_paths_and_settings

# Load configuration
config_vars = get_download_paths_and_settings()
DOWNLOAD_CONFIG = config_vars['DOWNLOAD_CONFIG']
SYSTEM_REQUIREMENTS = config_vars['SYSTEM_REQUIREMENTS']
LOG_CONFIG = config_vars['LOG_CONFIG']
CDS_CONFIG = config_vars['CDS_CONFIG']


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG['level']),
        format=LOG_CONFIG['format'],
        handlers=[
            logging.FileHandler(LOG_CONFIG['filename']),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if enough disk space is available"""
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    
    if available_gb < required_gb:
        logging.error(f"Insufficient disk space. Required: {required_gb}GB, Available: {available_gb:.2f}GB")
        return False
    
    logging.info(f"Disk space check passed. Available: {available_gb:.2f}GB")
    return True


def check_python_version():
    """Check if Python version meets requirements"""
    required = SYSTEM_REQUIREMENTS['python_version']
    current = sys.version_info[:2]
    
    if current < required:
        logging.error(f"Python {required[0]}.{required[1]}+ required, but {current[0]}.{current[1]} found")
        return False
    
    logging.info(f"Python version check passed: {current[0]}.{current[1]}")
    return True


def retry_with_backoff(func, *args, **kwargs):
    """Retry function with exponential backoff"""
    max_retries = DOWNLOAD_CONFIG['max_retries']
    retry_delay = DOWNLOAD_CONFIG['retry_delay']
    retry_backoff = DOWNLOAD_CONFIG['retry_backoff']
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            
            wait_time = retry_delay * (retry_backoff ** attempt)
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
            time.sleep(wait_time)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB"""
    if file_path.exists():
        return file_path.stat().st_size / (1024**2)
    return 0


def create_progress_bar(total_size: int, desc: str) -> tqdm:
    """Create a progress bar for downloads"""
    return tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=desc,
        ncols=100
    )


def format_eta(seconds: float) -> str:
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds/3600)
        minutes = int((seconds%3600)/60)
        return f"{hours}h {minutes}m"


def validate_file_exists(file_path: Path, min_size_mb: float = 1) -> bool:
    """Check if file exists and has reasonable size"""
    if not file_path.exists():
        return False
    
    size_mb = get_file_size_mb(file_path)
    if size_mb < min_size_mb:
        logging.warning(f"File {file_path.name} is too small: {size_mb:.2f}MB")
        return False
    
    return True


def clean_partial_downloads(directory: Path, pattern: str = "*.tmp"):
    """Clean up partial download files"""
    for tmp_file in directory.glob(pattern):
        try:
            tmp_file.unlink()
            logging.info(f"Removed partial download: {tmp_file.name}")
        except Exception as e:
            logging.warning(f"Could not remove {tmp_file.name}: {e}")


def estimate_download_time(file_size_bytes: int, speed_mbps: float = 10) -> float:
    """Estimate download time in seconds based on file size and connection speed"""
    # Convert Mbps to bytes per second
    speed_bps = speed_mbps * 1024 * 1024 / 8
    return file_size_bytes / speed_bps


def setup_cds_credentials():
    """Setup CDS API credentials from config or environment"""
    # If credentials are configured in config file, set environment variables
    if CDS_CONFIG['key'] and CDS_CONFIG['key'] != 'YOUR_API_KEY_HERE':
        os.environ['CDSAPI_URL'] = CDS_CONFIG['url']
        os.environ['CDSAPI_KEY'] = CDS_CONFIG['key']
        logging.info("CDS credentials loaded from config file")
        return True
    
    # Check if already set in environment
    if os.environ.get('CDSAPI_URL') and os.environ.get('CDSAPI_KEY'):
        logging.info("CDS credentials found in environment variables")
        return True
    
    # Check .cdsapirc file
    cdsapirc_path = Path.home() / '.cdsapirc'
    if cdsapirc_path.exists():
        logging.info("CDS credentials found in .cdsapirc file")
        return True
    
    logging.error("CDS API credentials not found. Please configure either:")
    logging.error("1. Set your API key in config/download_config.py")
    logging.error("2. Set CDSAPI_URL and CDSAPI_KEY environment variables")
    logging.error("3. Create ~/.cdsapirc file with url and key")
    return False


def check_cds_credentials() -> bool:
    """Check if CDS API credentials are configured (legacy function)"""
    return setup_cds_credentials()


def print_download_summary(successful: list, failed: list, start_time: float):
    """Print summary of download results"""
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total time: {format_eta(elapsed)}")
    print(f"Successful downloads: {len(successful)}")
    
    if successful:
        print("\nSuccessfully downloaded:")
        for item in successful:
            print(f"  ✓ {item}")
    
    if failed:
        print(f"\nFailed downloads: {len(failed)}")
        for item in failed:
            print(f"  ✗ {item}")
    
    print("="*50)


def monitor_system_resources():
    """Monitor and log system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage('/')
    
    logging.debug(f"System resources - CPU: {cpu_percent}%, "
                 f"Memory: {memory.percent}%, "
                 f"Disk free: {disk.free / (1024**3):.1f}GB")