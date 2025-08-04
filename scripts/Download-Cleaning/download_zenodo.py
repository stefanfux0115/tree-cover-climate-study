"""
Zenodo tree cover data downloader
Handles downloading tree cover data from Zenodo records for China (2012-2019)
"""

import os
import requests
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


class ZenodoDownloader:
    """Handle Zenodo tree cover data downloads"""
    
    def __init__(self, api_token: str, base_dir: str = "data/raw/tree_cover_zenodo"):
        """
        Initialize Zenodo downloader
        
        Args:
            api_token: Zenodo API token for authentication
            base_dir: Base directory for storing downloaded files
        """
        self.api_token = api_token
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        
        # Zenodo record IDs for tree cover data
        self.record_ids = [
            "11047923",
            "11047925", 
            "11047917"
        ]
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_record_metadata(self, record_id: str) -> Optional[Dict]:
        """
        Fetch metadata for a Zenodo record
        
        Args:
            record_id: Zenodo record ID
            
        Returns:
            Record metadata or None if failed
        """
        url = f"https://zenodo.org/api/records/{record_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching metadata for record {record_id}: {e}")
            return None
    
    def download_file(self, file_info: Dict, output_dir: Path) -> bool:
        """
        Download a single file from Zenodo
        
        Args:
            file_info: File information from Zenodo API
            output_dir: Directory to save the file
            
        Returns:
            True if successful, False otherwise
        """
        filename = file_info['key']
        file_url = file_info['links']['self']
        file_size = file_info['size']
        
        output_path = output_dir / filename
        
        # Skip if already downloaded with correct size
        if output_path.exists() and output_path.stat().st_size == file_size:
            self.logger.info(f"  ✓ {filename} already downloaded")
            return True
        
        self.logger.info(f"  Downloading {filename} ({file_size / 1024 / 1024:.1f} MB)...")
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(file_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"    Downloaded successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"    Error downloading {filename}: {e}")
            # Remove partial download
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_record(self, record_id: str, years_filter: Optional[List[int]] = None) -> Dict:
        """
        Download all files from a Zenodo record
        
        Args:
            record_id: Zenodo record ID
            years_filter: Optional list of years to filter files
            
        Returns:
            Dictionary with download results
        """
        output_dir = self.base_dir / f"record_{record_id}"
        output_dir.mkdir(exist_ok=True)
        
        # Get record metadata
        metadata = self.get_record_metadata(record_id)
        if not metadata:
            return {'success': False, 'downloaded': 0, 'failed': 0, 'total': 0}
        
        files = metadata.get('files', [])
        record_title = metadata.get('metadata', {}).get('title', 'Unknown')
        
        self.logger.info(f"\nRecord {record_id} - {record_title}")
        self.logger.info(f"Total files: {len(files)}")
        
        # Filter files by year if specified
        if years_filter:
            filtered_files = []
            for file in files:
                filename = file['key']
                # Check if any year in the filter is in the filename
                for year in years_filter:
                    if str(year) in filename:
                        filtered_files.append(file)
                        break
            files = filtered_files
            self.logger.info(f"Files matching years {years_filter}: {len(files)}")
        
        # Download each file
        downloaded = 0
        failed = 0
        
        for file_info in files:
            if self.download_file(file_info, output_dir):
                downloaded += 1
            else:
                failed += 1
        
        return {
            'success': failed == 0,
            'downloaded': downloaded,
            'failed': failed,
            'total': len(files)
        }
    
    def download_all_records(self, years: List[int]) -> Dict:
        """
        Download tree cover data from all Zenodo records
        
        Args:
            years: List of years to download
            
        Returns:
            Dictionary with overall download results
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Downloading Tree Cover Data from Zenodo")
        self.logger.info("="*60)
        self.logger.info(f"Target years: {min(years)}-{max(years)} ({len(years)} years)")
        self.logger.info(f"Output directory: {self.base_dir}")
        
        results = {
            'successful_records': [],
            'failed_records': [],
            'total_files': 0,
            'downloaded_files': 0,
            'failed_files': 0
        }
        
        for record_id in self.record_ids:
            record_result = self.download_record(record_id, years)
            
            if record_result['success']:
                results['successful_records'].append(record_id)
            else:
                results['failed_records'].append(record_id)
            
            results['total_files'] += record_result['total']
            results['downloaded_files'] += record_result['downloaded']
            results['failed_files'] += record_result['failed']
        
        # Summary
        self.logger.info("\n" + "-"*60)
        self.logger.info("Tree Cover Download Summary")
        self.logger.info("-"*60)
        self.logger.info(f"Records processed: {len(self.record_ids)}")
        self.logger.info(f"Successful records: {len(results['successful_records'])}")
        self.logger.info(f"Failed records: {len(results['failed_records'])}")
        self.logger.info(f"Total files: {results['total_files']}")
        self.logger.info(f"Downloaded files: {results['downloaded_files']}")
        self.logger.info(f"Failed files: {results['failed_files']}")
        
        # List downloaded files by record
        self.logger.info("\nDownloaded files by record:")
        for record_id in self.record_ids:
            record_dir = self.base_dir / f"record_{record_id}"
            if record_dir.exists():
                files = list(record_dir.glob("*"))
                self.logger.info(f"  Record {record_id}: {len(files)} files")
        
        return results
    
    def estimate_download_size(self, years: List[int]) -> float:
        """
        Estimate total download size for given years
        
        Args:
            years: List of years to estimate
            
        Returns:
            Estimated size in GB
        """
        total_size = 0
        
        for record_id in self.record_ids:
            metadata = self.get_record_metadata(record_id)
            if metadata:
                files = metadata.get('files', [])
                
                # Filter by years
                for file in files:
                    filename = file['key']
                    for year in years:
                        if str(year) in filename:
                            # Check if file already exists
                            output_path = self.base_dir / f"record_{record_id}" / filename
                            if not output_path.exists() or output_path.stat().st_size != file['size']:
                                total_size += file['size']
                            break
        
        return total_size / (1024 ** 3)  # Convert to GB
    
    def validate_downloads(self, years: List[int]) -> Dict:
        """
        Validate downloaded files
        
        Args:
            years: List of years to validate
            
        Returns:
            Validation results
        """
        results = {
            'valid_files': 0,
            'missing_files': 0,
            'corrupt_files': 0,
            'details': {}
        }
        
        for record_id in self.record_ids:
            record_results = {
                'valid': [],
                'missing': [],
                'corrupt': []
            }
            
            # Get expected files from metadata
            metadata = self.get_record_metadata(record_id)
            if metadata:
                files = metadata.get('files', [])
                
                # Filter by years
                for file in files:
                    filename = file['key']
                    for year in years:
                        if str(year) in filename:
                            output_path = self.base_dir / f"record_{record_id}" / filename
                            
                            if not output_path.exists():
                                record_results['missing'].append(filename)
                                results['missing_files'] += 1
                            elif output_path.stat().st_size != file['size']:
                                record_results['corrupt'].append(filename)
                                results['corrupt_files'] += 1
                            else:
                                record_results['valid'].append(filename)
                                results['valid_files'] += 1
                            break
            
            results['details'][record_id] = record_results
        
        return results