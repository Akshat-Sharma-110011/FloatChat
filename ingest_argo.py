
"""
ARGO Data Fetcher - Downloads raw NetCDF files from ARGO oceanographic data repositories
"""

import os
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from urllib.parse import urljoin
import ftplib
import re

# Constants
DEFAULT_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
CHUNK_SIZE = 8192
SERVER_PAUSE = 0.5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARGODataFetcher:
    """
    Fetches ARGO NetCDF files from various ARGO data sources.
    Supports both HTTP and FTP protocols.
    """

    def __init__(self, base_dir: str = "data/raw/argo/"):
        """
        Initialize the ARGO data fetcher.

        Args:
            base_dir: Directory to save downloaded NetCDF files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # ARGO data sources
        self.sources = {
            'gdac': 'https://data-argo.ifremer.fr/',
            'usgodae': 'https://usgodae.org/pub/outgoing/argo/',
            'coriolis_ftp': 'ftp.ifremer.fr',
            'aoml': 'https://www.aoml.noaa.gov/phod/argo/'
        }

        # Common ARGO file patterns
        self.file_patterns = [
            re.compile(r'.*_prof\.nc$', re.IGNORECASE),  # Profile files
            re.compile(r'.*_tech\.nc$', re.IGNORECASE),  # Technical files
            re.compile(r'.*_meta\.nc$', re.IGNORECASE),  # Metadata files
            re.compile(r'.*_Rtraj\.nc$', re.IGNORECASE),  # Real-time trajectory files
            re.compile(r'.*_Dtraj\.nc$', re.IGNORECASE),  # Delayed-mode trajectory files
        ]

    def fetch_from_gdac(self,
                        dac_name: Optional[str] = None,
                        float_id: Optional[str] = None,
                        max_files: int = 100) -> List[str]:
        """
        Fetch files from ARGO GDAC (Global Data Assembly Centre).

        Args:
            dac_name: Specific DAC (Data Assembly Centre) name (e.g., 'aoml', 'coriolis')
            float_id: Specific float ID to download
            max_files: Maximum number of files to download

        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        base_url = self.sources['gdac']

        try:
            if dac_name and float_id:
                # Download specific float data
                url = f"{base_url}dac/{dac_name}/{float_id}/"
                downloaded_files.extend(self._download_from_directory(url, max_files))
            elif dac_name:
                # Download from specific DAC
                url = f"{base_url}dac/{dac_name}/"
                downloaded_files.extend(self._browse_and_download_dac(url, max_files))
            else:
                # Browse available DACs and download samples
                dacs_url = f"{base_url}dac/"
                downloaded_files.extend(self._browse_and_download_dac(dacs_url, max_files))

        except Exception as e:
            logger.error(f"Error fetching from GDAC: {e}") 

        return downloaded_files

    def fetch_recent_profiles(self, days_back: int = 7, max_files: int = 50) -> List[str]:
        """
        Fetch recent profile files from the last N days.

        Args:
            days_back: Number of days to look back
            max_files: Maximum number of files to download

        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        base_url = self.sources['gdac']

        try:
            # Get recent data from the GDAC
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # ARGO organizes data by date in some repositories
            for i in range(days_back):
                current_date = start_date + timedelta(days=i)
                date_str = current_date.strftime('%Y/%m/%d')

                # Try different date-based URL patterns
                urls_to_try = [
                    f"{base_url}geo/{date_str}/",
                    f"{base_url}profiles/{date_str}/",
                ]

                for url in urls_to_try:
                    try:
                        files = self._download_from_directory(url, max_files // days_back)
                        downloaded_files.extend(files)
                        if len(downloaded_files) >= max_files:
                            break
                    except:
                        continue

                if len(downloaded_files) >= max_files:
                    break

        except Exception as e:
            logger.error(f"Error fetching recent profiles: {e}")

        return downloaded_files

    def fetch_by_geographic_region(self,
                                   lat_min: float, lat_max: float,
                                   lon_min: float, lon_max: float,
                                   max_files: int = 100) -> List[str]:
        """
        Fetch files from a specific geographic region.

        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds  
            max_files: Maximum number of files to download

        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        base_url = self.sources['gdac']

        try:
            # ARGO geo directory structure (if available)
            geo_url = f"{base_url}geo/"
            downloaded_files.extend(self._download_from_directory(geo_url, max_files))

        except Exception as e:
            logger.error(f"Error fetching by geographic region: {e}")

        return downloaded_files

    def fetch_from_coriolis_ftp(self,
                                dac_name: Optional[str] = None,
                                float_id: Optional[str] = None,
                                max_files: int = 100) -> List[str]:
        """
        Fetch files from Coriolis FTP server.
        Args:
            dac_name: Specific DAC (Data Assembly Centre) name (e.g., 'coriolis')
            float_id: Specific float ID to download
            max_files: Maximum number of files to download
        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        ftp_host = self.sources['coriolis_ftp']

        try:
            with ftplib.FTP(ftp_host, timeout=DEFAULT_TIMEOUT) as ftp:
                ftp.login()
                logger.info(f"Connected to FTP: {ftp_host}")

                # Navigate to the ARGO directory
                ftp.cwd('ifremer/argo')

                if dac_name and float_id:
                    path = f"dac/{dac_name}/{float_id}/"
                    downloaded_files.extend(self._download_from_ftp_directory(ftp, path, max_files))
                elif dac_name:
                    path = f"dac/{dac_name}/"
                    downloaded_files.extend(self._browse_and_download_ftp_dac(ftp, path, max_files))
                else:
                    path = "dac/"
                    downloaded_files.extend(self._browse_and_download_ftp_dac(ftp, path, max_files))

        except ftplib.all_errors as e:
            logger.error(f"FTP Error fetching from Coriolis: {e}")
        except Exception as e:
            logger.error(f"General Error fetching from Coriolis FTP: {e}")

        return downloaded_files

    def _browse_and_download_dac(self, dac_url: str, max_files: int) -> List[str]:
        """Browse DAC directory and download NetCDF files."""
        downloaded_files = []

        try:
            response = requests.get(dac_url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()

            # Parse HTML to find subdirectories (float IDs)
            import re
            links = re.findall(r'href="([^"]*)"', response.text)

            for link in links:
                if len(downloaded_files) >= max_files:
                    break

                if link.endswith('/') and link not in ['../', './']:
                    float_url = urljoin(dac_url, link)
                    try:
                        files = self._download_from_directory(float_url, 5)  # Limit per float
                        downloaded_files.extend(files)
                    except:
                        continue

        except Exception as e:
            logger.error(f"Error browsing DAC {dac_url}: {e}")

        return downloaded_files

    def _download_from_directory(self, url: str, max_files: int) -> List[str]:
        """Download NetCDF files from a directory URL."""
        downloaded_files = []

        try:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()

            # Find NetCDF files in the HTML
            nc_files = []
            for pattern in self.file_patterns:
                matches = pattern.findall(response.text)
                nc_files.extend(matches)

            # Download files
            for nc_file in nc_files[:max_files]:
                if len(downloaded_files) >= max_files:
                    break

                file_url = urljoin(url, nc_file)
                local_path = self._download_file(file_url, nc_file)
                if local_path:
                    downloaded_files.append(local_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error downloading from directory {url}: {e}")
        except Exception as e:
            logger.error(f"Error downloading from directory {url}: {e}")

        return downloaded_files

    def _download_file(self, url: str, filename: str) -> Optional[str]:
        """Download a single file."""
        try:
            # Create subdirectory based on URL structure for organization
            url_parts = url.split('/')
            if 'dac' in url_parts:
                dac_idx = url_parts.index('dac')
                if dac_idx + 2 < len(url_parts):
                    dac_name = url_parts[dac_idx + 1]
                    float_id = url_parts[dac_idx + 2]
                    subdir = self.base_dir / dac_name / float_id
                else:
                    subdir = self.base_dir / "misc"
            else:
                subdir = self.base_dir / "misc"

            subdir.mkdir(parents=True, exist_ok=True)
            local_path = subdir / filename

            # Skip if file already exists
            if local_path.exists():
                logger.info(f"File already exists: {local_path}")
                return str(local_path)

            logger.info(f"Downloading: {url}")
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)

            logger.info(f"Downloaded: {local_path}")
            time.sleep(SERVER_PAUSE)  # Be respectful to the server
            return str(local_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error downloading {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"General Error downloading {url}: {e}")
            return None

    def _download_from_ftp_directory(self, ftp: ftplib.FTP, directory_path: str, max_files: int) -> List[str]:
        """Download NetCDF files from an FTP directory."""
        downloaded_files = []
        original_cwd = ftp.pwd()
        try:
            ftp.cwd(directory_path)
            files_in_dir = ftp.nlst()
            nc_files = []
            for filename in files_in_dir:
                for pattern in self.file_patterns:
                    if pattern.match(filename):
                        nc_files.append(filename)
                        break
            
            for nc_file in nc_files[:max_files]:
                if len(downloaded_files) >= max_files:
                    break
                local_path = self._download_ftp_file(ftp, nc_file, directory_path)
                if local_path:
                    downloaded_files.append(local_path)
        except ftplib.all_errors as e:
            logger.error(f"FTP Error downloading from directory {directory_path}: {e}")
        except Exception as e:
            logger.error(f"General Error downloading from FTP directory {directory_path}: {e}")
        finally:
            ftp.cwd(original_cwd) # Return to original directory
        return downloaded_files

    def _download_ftp_file(self, ftp: ftplib.FTP, filename: str, remote_path: str) -> Optional[str]:
        """Download a single file from FTP."""
        try:
            # Create subdirectory structure similar to HTTP downloads
            path_parts = remote_path.split('/')
            if 'dac' in path_parts:
                dac_idx = path_parts.index('dac')
                if dac_idx + 2 < len(path_parts):
                    dac_name = path_parts[dac_idx + 1]
                    float_id = path_parts[dac_idx + 2]
                    subdir = self.base_dir / dac_name / float_id
                else:
                    subdir = self.base_dir / "misc"
            else:
                subdir = self.base_dir / "misc"

            subdir.mkdir(parents=True, exist_ok=True)
            local_path = subdir / filename

            if local_path.exists():
                logger.info(f"File already exists: {local_path}")
                return str(local_path)
            
            logger.info(f"Downloading (FTP): {filename} to {local_path}")
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f"RETR {filename}", f.write, CHUNK_SIZE)
            
            logger.info(f"Downloaded (FTP): {local_path}")
            time.sleep(SERVER_PAUSE)
            return str(local_path)

        except ftplib.all_errors as e:
            logger.error(f"FTP Error downloading {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"General Error downloading {filename} via FTP: {e}")
            return None

    def _is_ftp_dir(self, ftp: ftplib.FTP, item: str) -> bool:
        """Checks if an item in an FTP listing is a directory."""
        try:
            ftp.cwd(item)
            ftp.cwd('..')
            return True
        except ftplib.error_perm:
            return False
        except Exception as e:
            logger.warning(f"Error checking FTP item type for {item}: {e}")
            return False

    def fetch_sample_data(self, num_files: int = 10) -> List[str]:
        """
        Fetch a sample of ARGO data files for testing.

        Args:
            num_files: Number of sample files to download

        Returns:
            List of downloaded file paths
        """
        logger.info(f"Fetching {num_files} sample ARGO files...")

        downloaded_files = []

        # Try different approaches to get sample data
        strategies = [
            lambda: self.fetch_from_gdac(dac_name="aoml", max_files=num_files // 2),
            lambda: self.fetch_from_gdac(dac_name="coriolis", max_files=num_files // 2),
            lambda: self.fetch_recent_profiles(days_back=30, max_files=num_files),
        ]

        for strategy in strategies:
            try:
                files = strategy()
                downloaded_files.extend(files)
                if len(downloaded_files) >= num_files:
                    break
            except Exception as e:
                logger.warning(f"Strategy failed: {e}")
                continue

        return downloaded_files[:num_files]


def main():
    """Example usage of the ARGO data fetcher."""

    # Initialize the fetcher
    fetcher = ARGODataFetcher()

    # Fetch sample data
    print("Fetching sample ARGO data...")
    sample_files = fetcher.fetch_sample_data(num_files=20)

    print(f"\nDownloaded {len(sample_files)} files:")
    for file_path in sample_files:
        print(f"  - {file_path}")

    # Example: Fetch from specific DAC
    # aoml_files = fetcher.fetch_from_gdac(dac_name="aoml", max_files=10)

    # Example: Fetch recent data
    # recent_files = fetcher.fetch_recent_profiles(days_back=7, max_files=15)

    # Example: Fetch by region (requires implementation of spatial filtering)
    # regional_files = fetcher.fetch_by_geographic_region(
    #     lat_min=30, lat_max=45, lon_min=-80, lon_max=-60, max_files=10
    # )


if __name__ == "__main__":
    main()