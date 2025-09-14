
"""
ARGO NetCDF to CSV Converter - Converts ARGO NetCDF files to CSV format
for easier data analysis and processing.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import warnings

# Suppress xarray warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARGONetCDFConverter:
    """
    Converts ARGO NetCDF files to CSV format with proper data organization.
    Handles different ARGO file types: profiles, trajectories, technical, and metadata.
    """

    def __init__(self, input_dir: str = "data/raw/argo/", output_dir: str = "data/processed/csv/"):
        """
        Initialize the converter.

        Args:
            input_dir: Directory containing NetCDF files
            output_dir: Directory to save CSV files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File type mappings
        self.file_types = {
            'prof': 'profiles',
            'tech': 'technical',
            'meta': 'metadata',
            'Rtraj': 'trajectories_realtime',
            'Dtraj': 'trajectories_delayed'
        }

        # Key variables for each file type
        self.key_variables = {
            'prof': [
                'JULD', 'LATITUDE', 'LONGITUDE', 'PRES', 'TEMP', 'PSAL',
                'PRES_QC', 'TEMP_QC', 'PSAL_QC', 'PROFILE_PRES_QC',
                'PROFILE_TEMP_QC', 'PROFILE_PSAL_QC', 'PLATFORM_NUMBER'
            ],
            'tech': [
                'TECHNICAL_PARAMETER_NAME', 'TECHNICAL_PARAMETER_VALUE',
                'PLATFORM_NUMBER', 'CYCLE_NUMBER'
            ],
            'meta': [
                'PLATFORM_NUMBER', 'PLATFORM_TYPE', 'FLOAT_SERIAL_NO',
                'FIRMWARE_VERSION', 'WMO_INST_TYPE', 'PROJECT_NAME',
                'DATA_CENTRE', 'LAUNCH_DATE', 'LAUNCH_LATITUDE', 'LAUNCH_LONGITUDE'
            ],
            'Rtraj': [
                'JULD', 'LATITUDE', 'LONGITUDE', 'PRES', 'TEMP', 'PSAL',
                'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'MEASUREMENT_CODE'
            ],
            'Dtraj': [
                'JULD', 'LATITUDE', 'LONGITUDE', 'PRES', 'TEMP', 'PSAL',
                'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'MEASUREMENT_CODE'
            ]
        }

    def convert_all_files(self) -> Dict[str, List[str]]:
        """
        Convert all NetCDF files in the input directory to CSV.

        Returns:
            Dictionary mapping file types to lists of converted file paths
        """
        converted_files = {file_type: [] for file_type in self.file_types.values()}

        logger.info(f"Starting conversion of NetCDF files in {self.input_dir}")

        # Walk through all NetCDF files
        for nc_file in self.input_dir.rglob("*.nc"):
            try:
                file_type = self._identify_file_type(nc_file.name)
                if file_type:
                    csv_path = self._convert_single_file(nc_file, file_type)
                    if csv_path:
                        converted_files[self.file_types[file_type]].append(str(csv_path))

            except Exception as e:
                logger.error(f"Error converting {nc_file}: {e}")
                continue

        # Log summary
        total_converted = sum(len(files) for files in converted_files.values())
        logger.info(f"Conversion complete. Converted {total_converted} files:")
        for file_type, files in converted_files.items():
            if files:
                logger.info(f"  {file_type}: {len(files)} files")

        return converted_files

    def _identify_file_type(self, filename: str) -> Optional[str]:
        """Identify the ARGO file type from filename."""
        for file_type in self.file_types.keys():
            if f"_{file_type}.nc" in filename:
                return file_type
        return None

    def _convert_single_file(self, nc_file: Path, file_type: str) -> Optional[Path]:
        """
        Convert a single NetCDF file to CSV.

        Args:
            nc_file: Path to NetCDF file
            file_type: Type of ARGO file (prof, tech, meta, etc.)

        Returns:
            Path to converted CSV file or None if failed
        """
        try:
            logger.info(f"Converting {nc_file.name} (type: {file_type})")

            # Load the NetCDF file
            with xr.open_dataset(nc_file) as ds:
                # Convert based on file type
                if file_type == 'prof':
                    df = self._convert_profile_file(ds)
                elif file_type == 'tech':
                    df = self._convert_technical_file(ds)
                elif file_type == 'meta':
                    df = self._convert_metadata_file(ds)
                elif file_type in ['Rtraj', 'Dtraj']:
                    df = self._convert_trajectory_file(ds)
                else:
                    logger.warning(f"Unknown file type: {file_type}")
                    return None

                if df is not None and not df.empty:
                    # Create output path
                    csv_filename = nc_file.stem + '.csv'
                    output_subdir = self.output_dir / self.file_types[file_type]
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    csv_path = output_subdir / csv_filename

                    # Save to CSV
                    df.to_csv(csv_path, index=False, na_rep='NaN')
                    logger.info(f"Saved CSV: {csv_path}")
                    return csv_path
                else:
                    logger.warning(f"No data extracted from {nc_file.name}")
                    return None

        except Exception as e:
            logger.error(f"Error converting {nc_file}: {e}")
            return None

    def _convert_profile_file(self, ds: xr.Dataset) -> Optional[pd.DataFrame]:
        """Convert profile NetCDF to DataFrame."""
        try:
            data_records = []

            # Get platform number
            platform_number = self._extract_scalar_value(ds.get('PLATFORM_NUMBER'))

            # Get number of profiles and levels
            n_prof = ds.sizes.get('N_PROF', 1)
            n_levels = ds.sizes.get('N_LEVELS', 0)

            for prof_idx in range(n_prof):
                # Profile-level data
                juld = self._extract_time_value(ds.get('JULD'), prof_idx)
                latitude = self._extract_value(ds.get('LATITUDE'), prof_idx)
                longitude = self._extract_value(ds.get('LONGITUDE'), prof_idx)

                # Level data
                for level_idx in range(n_levels):
                    record = {
                        'platform_number': platform_number,
                        'profile_index': prof_idx,
                        'level_index': level_idx,
                        'date_time': juld,
                        'latitude': latitude,
                        'longitude': longitude,
                        'pressure': self._extract_value(ds.get('PRES'), prof_idx, level_idx),
                        'temperature': self._extract_value(ds.get('TEMP'), prof_idx, level_idx),
                        'salinity': self._extract_value(ds.get('PSAL'), prof_idx, level_idx),
                        'pressure_qc': self._extract_qc_value(ds.get('PRES_QC'), prof_idx, level_idx),
                        'temperature_qc': self._extract_qc_value(ds.get('TEMP_QC'), prof_idx, level_idx),
                        'salinity_qc': self._extract_qc_value(ds.get('PSAL_QC'), prof_idx, level_idx),
                    }

                    # Add any additional biogeochemical parameters
                    for var_name in ds.data_vars:
                        if var_name not in ['PRES', 'TEMP', 'PSAL', 'JULD', 'LATITUDE', 'LONGITUDE'] and \
                                '_QC' not in var_name and \
                                ds[var_name].dims == ('N_PROF', 'N_LEVELS'):
                            record[var_name.lower()] = self._extract_value(ds.get(var_name), prof_idx, level_idx)

                    data_records.append(record)

            return pd.DataFrame(data_records)

        except Exception as e:
            logger.error(f"Error converting profile data: {e}")
            return None

    def _convert_technical_file(self, ds: xr.Dataset) -> Optional[pd.DataFrame]:
        """Convert technical NetCDF to DataFrame."""
        try:
            data_records = []

            platform_number = self._extract_scalar_value(ds.get('PLATFORM_NUMBER'))

            # Technical parameters are usually stored as arrays
            if 'TECHNICAL_PARAMETER_NAME' in ds.variables:
                param_names = ds['TECHNICAL_PARAMETER_NAME'].values
                param_values = ds.get('TECHNICAL_PARAMETER_VALUE', ds.get('TECHNICAL_PARAMETER_DATA'))

                if param_values is not None:
                    for i, param_name in enumerate(param_names):
                        if isinstance(param_name, bytes):
                            param_name = param_name.decode('utf-8').strip()
                        elif isinstance(param_name, np.ndarray):
                            param_name = ''.join([chr(x) for x in param_name if x != 0]).strip()

                        record = {
                            'platform_number': platform_number,
                            'parameter_index': i,
                            'parameter_name': param_name,
                            'parameter_value': self._extract_value(param_values, i) if hasattr(param_values,
                                                                                               'shape') else param_values,
                            'cycle_number': self._extract_value(ds.get('CYCLE_NUMBER'),
                                                                i) if 'CYCLE_NUMBER' in ds else None
                        }
                        data_records.append(record)

            return pd.DataFrame(data_records) if data_records else None

        except Exception as e:
            logger.error(f"Error converting technical data: {e}")
            return None

    def _convert_metadata_file(self, ds: xr.Dataset) -> Optional[pd.DataFrame]:
        """Convert metadata NetCDF to DataFrame."""
        try:
            record = {}

            # Extract metadata fields
            metadata_fields = {
                'platform_number': 'PLATFORM_NUMBER',
                'platform_type': 'PLATFORM_TYPE',
                'float_serial_no': 'FLOAT_SERIAL_NO',
                'firmware_version': 'FIRMWARE_VERSION',
                'wmo_inst_type': 'WMO_INST_TYPE',
                'project_name': 'PROJECT_NAME',
                'data_centre': 'DATA_CENTRE',
                'launch_date': 'LAUNCH_DATE',
                'launch_latitude': 'LAUNCH_LATITUDE',
                'launch_longitude': 'LAUNCH_LONGITUDE'
            }

            for field_name, var_name in metadata_fields.items():
                if var_name in ds.variables:
                    value = self._extract_scalar_value(ds[var_name])
                    record[field_name] = value

            # Add any additional metadata variables
            for var_name in ds.variables:
                if var_name not in metadata_fields.values() and ds[var_name].dims == ():
                    record[var_name.lower()] = self._extract_scalar_value(ds[var_name])

            return pd.DataFrame([record]) if record else None

        except Exception as e:
            logger.error(f"Error converting metadata: {e}")
            return None

    def _convert_trajectory_file(self, ds: xr.Dataset) -> Optional[pd.DataFrame]:
        """Convert trajectory NetCDF to DataFrame."""
        try:
            data_records = []

            platform_number = self._extract_scalar_value(ds.get('PLATFORM_NUMBER'))
            n_measurement = ds.sizes.get('N_MEASUREMENT', 0)

            for meas_idx in range(n_measurement):
                record = {
                    'platform_number': platform_number,
                    'measurement_index': meas_idx,
                    'date_time': self._extract_time_value(ds.get('JULD'), meas_idx),
                    'latitude': self._extract_value(ds.get('LATITUDE'), meas_idx),
                    'longitude': self._extract_value(ds.get('LONGITUDE'), meas_idx),
                    'cycle_number': self._extract_value(ds.get('CYCLE_NUMBER'), meas_idx),
                    'measurement_code': self._extract_value(ds.get('MEASUREMENT_CODE'), meas_idx),
                }

                # Add pressure, temperature, salinity if available
                for var_name in ['PRES', 'TEMP', 'PSAL']:
                    if var_name in ds.variables:
                        record[var_name.lower()] = self._extract_value(ds.get(var_name), meas_idx)

                data_records.append(record)

            return pd.DataFrame(data_records)

        except Exception as e:
            logger.error(f"Error converting trajectory data: {e}")
            return None

    def _extract_value(self, var, *indices) -> Any:
        """Safely extract value from xarray variable."""
        if var is None:
            return np.nan

        try:
            if len(indices) == 0:
                return var.values
            elif len(indices) == 1:
                return var.values[indices[0]] if var.values.size > indices[0] else np.nan
            elif len(indices) == 2:
                return var.values[indices[0], indices[1]] if var.values.shape[0] > indices[0] and var.values.shape[1] > \
                                                             indices[1] else np.nan
            else:
                return np.nan
        except:
            return np.nan

    def _extract_scalar_value(self, var) -> Any:
        """Extract scalar value, handling string arrays."""
        if var is None:
            return None

        try:
            value = var.values
            if isinstance(value, np.ndarray):
                if value.dtype.kind in ['S', 'U']:  # String types
                    if value.ndim == 0:
                        return str(value.item())
                    else:
                        # Handle character arrays
                        return ''.join([chr(x) for x in value.flatten() if x != 0]).strip()
                else:
                    return value.item() if value.size == 1 else value
            else:
                return value
        except:
            return None

    def _extract_qc_value(self, var, *indices) -> Any:
        """Extract quality control flag."""
        if var is None:
            return None

        try:
            value = self._extract_value(var, *indices)
            if isinstance(value, bytes):
                return value.decode('utf-8').strip()
            elif isinstance(value, np.ndarray) and value.dtype.kind in ['S', 'U']:
                return str(value).strip()
            else:
                return value
        except:
            return None

    def _extract_time_value(self, var, index=None) -> Optional[str]:
        """Extract and convert time value to ISO format."""
        if var is None:
            return None

        try:
            if index is not None:
                time_val = var.values[index]
            else:
                time_val = var.values

            if pd.isna(time_val) or time_val == var.attrs.get('_FillValue', np.nan):
                return None

            # ARGO time is days since 1950-01-01
            reference_date = pd.Timestamp('1950-01-01')
            actual_time = reference_date + pd.Timedelta(days=float(time_val))
            return actual_time.isoformat()

        except:
            return None

    def create_summary_report(self, converted_files: Dict[str, List[str]]) -> str:
        """Create a summary report of converted files."""
        report_path = self.output_dir / "conversion_summary.txt"

        with open(report_path, 'w') as f:
            f.write(f"ARGO NetCDF to CSV Conversion Summary\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")

            total_files = sum(len(files) for files in converted_files.values())
            f.write(f"Total files converted: {total_files}\n\n")

            for file_type, files in converted_files.items():
                if files:
                    f.write(f"{file_type.upper()}:\n")
                    f.write(f"  Count: {len(files)}\n")
                    f.write("  Files:\n")
                    for file_path in files:
                        f.write(f"    - {file_path}\n")
                    f.write("\n")

        logger.info(f"Summary report saved to: {report_path}")
        return str(report_path)


def main():
    """Main function to run the converter."""
    converter = ARGONetCDFConverter()

    print("Converting ARGO NetCDF files to CSV format...")
    converted_files = converter.convert_all_files()

    # Create summary report
    summary_path = converter.create_summary_report(converted_files)

    print(f"\nConversion complete! Summary saved to: {summary_path}")
    print("\nConverted files by type:")
    for file_type, files in converted_files.items():
        if files:
            print(f"  {file_type}: {len(files)} files")


if __name__ == "__main__":
    main()