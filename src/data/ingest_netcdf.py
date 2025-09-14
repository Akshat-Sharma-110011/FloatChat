"""
FloatChat ARGO Data Ingestion Module

This module handles the ingestion of ARGO oceanographic data from CSV files
into PostgreSQL database and Parquet files for analytics. It supports both
batch processing and streaming modes with comprehensive error handling,
logging, and data validation.

Author: FloatChat Team
Version: 1.0.0
"""

import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def setup_logging() -> logging.Logger:
    """
    Setup logging configuration with timestamp-based log files.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamp for log filename in DD-MM-YYYY_Hr-Min-Sec format
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_filename = logs_dir / f"{timestamp}.log"

    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters with status indicators
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler with detailed logging
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler with simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"[INIT] Logging initialized. Log file: {log_filename}")
    return logger


# Initialize logger
logger = setup_logging()


@dataclass
class ARGOProfile:
    """
    Data class representing a single ARGO profile measurement.

    Attributes:
        basin: Basin identifier (integer)
        timestamp: Measurement timestamp (ISO format)
        cycle_number: Float cycle number
        vertical_sampling_scheme: Description of sampling methodology
        longitude: Longitude coordinate (decimal degrees)
        latitude: Latitude coordinate (decimal degrees)
        pressure_decibar: Pressure measurement in decibars
        salinity_psu: Salinity in Practical Salinity Units
        temperature_degc: Temperature in degrees Celsius
    """
    basin: int
    timestamp: str
    cycle_number: int
    vertical_sampling_scheme: str
    longitude: float
    latitude: float
    pressure_decibar: float
    salinity_psu: float
    temperature_degc: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for database insertion."""
        return asdict(self)

    def validate(self) -> bool:
        """
        Validate profile data for completeness and basic range checks.

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check for required fields
            if not all([
                self.timestamp,
                isinstance(self.cycle_number, int),
                isinstance(self.basin, int)
            ]):
                logger.debug(f"[VALIDATION] Missing required fields in profile: basin={self.basin}, cycle={self.cycle_number}")
                return False

            # Basic range validation
            if not (-180 <= self.longitude <= 180):
                logger.debug(f"[VALIDATION] Invalid longitude: {self.longitude}")
                return False

            if not (-90 <= self.latitude <= 90):
                logger.debug(f"[VALIDATION] Invalid latitude: {self.latitude}")
                return False

            if self.pressure_decibar < 0:
                logger.debug(f"[VALIDATION] Invalid pressure: {self.pressure_decibar}")
                return False

            return True

        except Exception as e:
            logger.error(f"[ERROR] Validation error for profile: {e}")
            return False


class DatabaseManager:
    """
    Handles PostgreSQL database operations with connection pooling and error handling.
    """

    def __init__(self, db_url: str):
        """
        Initialize database manager.

        Args:
            db_url: PostgreSQL connection URL
        """
        self.db_url = db_url
        self.engine = None
        logger.info(f"[INIT] Initializing database manager with URL: {db_url.split('@')[1] if '@' in db_url else 'masked'}")
        self._setup_connection()

    def _setup_connection(self) -> None:
        """Setup SQLAlchemy engine for database operations."""
        try:
            logger.debug("[DB] Setting up database connection...")
            self.engine = create_engine(
                self.db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # Set to True for SQL query logging
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("[SUCCESS] Database connection established successfully")
        except Exception as e:
            logger.error(f"[CRITICAL] Failed to establish database connection: {e}")
            logger.error(traceback.format_exc())
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic cleanup.

        Yields:
            Connection object for database operations
        """
        conn = None
        try:
            logger.debug("[DB] Acquiring database connection...")
            conn = self.engine.connect()
            yield conn
            logger.debug("[DB] Database operation completed successfully")
        except Exception as e:
            if conn:
                conn.rollback()
                logger.debug("[DB] Database transaction rolled back")
            logger.error(f"[ERROR] Database operation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("[DB] Database connection closed")

    def create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        logger.info("[DB] Creating/verifying database tables...")

        create_profiles_table = """
        CREATE TABLE IF NOT EXISTS profiles (
            id SERIAL PRIMARY KEY,
            basin INTEGER NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            cycle_number INTEGER NOT NULL,
            vertical_sampling_scheme TEXT,
            longitude DECIMAL(12, 8) NOT NULL,
            latitude DECIMAL(12, 8) NOT NULL,
            pressure_decibar DECIMAL(10, 2) NOT NULL,
            salinity_psu DECIMAL(10, 6),
            temperature_degc DECIMAL(10, 6),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_profiles_timestamp ON profiles(timestamp);
        CREATE INDEX IF NOT EXISTS idx_profiles_location ON profiles(longitude, latitude);
        CREATE INDEX IF NOT EXISTS idx_profiles_cycle ON profiles(cycle_number);
        CREATE INDEX IF NOT EXISTS idx_profiles_basin ON profiles(basin);
        CREATE INDEX IF NOT EXISTS idx_profiles_pressure ON profiles(pressure_decibar);
        """

        try:
            with self.get_connection() as conn:
                logger.debug("[DB] Executing table creation SQL...")
                conn.execute(text(create_profiles_table))
                conn.commit()
                logger.info("[SUCCESS] Database tables and indexes created/verified successfully")
        except SQLAlchemyError as e:
            logger.error(f"[CRITICAL] Failed to create database tables: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_record_count(self) -> int:
        """
        Get total number of records in the profiles table.

        Returns:
            int: Total record count
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM profiles"))
                count = result.scalar()
                logger.debug(f"[DB] Current database record count: {count}")
                return count
        except Exception as e:
            logger.error(f"[ERROR] Failed to get record count: {e}")
            return 0


class ARGOIngestor:
    """
    Main ingestion class for ARGO CSV data processing.

    Handles reading CSV files, data validation, and storage in both
    PostgreSQL and Parquet formats with comprehensive error handling.
    """

    # Hardcoded database URL as requested
    DATABASE_URL = "postgresql://postgres:Strong.password177013@localhost:6000/floatchat"

    def __init__(self, config_path: str = "configs/intel.yaml"):
        """
        Initialize ARGO data ingestor.

        Args:
            config_path: Path to configuration YAML file
        """
        logger.info("[INIT] Initializing ARGOIngestor...")

        try:
            self.config = self._load_config(config_path)
        except Exception as e:
            logger.warning(f"[WARNING] Could not load configs file, using defaults: {e}")
            self.config = self._get_default_config()

        # Use hardcoded database URL
        self.db_manager = DatabaseManager(self.DATABASE_URL)
        self.batch_size = self.config.get('ingestion', {}).get('batch_size', 1000)
        self.processed_count = 0
        self.error_count = 0

        logger.info(f"[CONFIG] Batch size set to: {self.batch_size}")

        # Ensure output directories exist
        self._setup_directories()

        # Create database tables
        self.db_manager.create_tables()

        logger.info("[SUCCESS] ARGOIngestor initialization completed")

    def _get_processed_filename(self, raw_filename: str) -> str:
        """
        Generate processed filename from raw filename.
        Converts 'argo_data_2025-01-01_to_2025-02-01_raw.csv' to 'argo_data_2025-01-01_to_2025-02-01_processed.parquet'

        Args:
            raw_filename: Original raw CSV filename

        Returns:
            str: Processed parquet filename
        """
        # Remove .csv extension and replace 'raw' with 'processed', then add .parquet
        base_name = Path(raw_filename).stem  # Remove extension
        processed_name = base_name.replace('_raw', '_processed') + '.parquet'
        logger.debug(f"[FILENAME] Converted '{raw_filename}' to '{processed_name}'")
        return processed_name

    def convert_postgres_to_parquet(self) -> Dict[str, Any]:
        """
        Convert existing PostgreSQL data to Parquet format.
        This method reads all data from PostgreSQL and saves it to Parquet.

        Returns:
            Dict[str, Any]: Summary of the conversion process
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("[PROCESS] STARTING POSTGRES TO PARQUET CONVERSION")
        logger.info("="*60)

        try:
            logger.info("[PROCESS] Reading all data from PostgreSQL database...")

            # Use direct psycopg2 connection
            conn = psycopg2.connect(self.DATABASE_URL)

            try:
                # Read all data from the profiles table
                query = """
                SELECT basin, timestamp, cycle_number, vertical_sampling_scheme,
                       longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
                FROM profiles
                ORDER BY timestamp, cycle_number, pressure_decibar
                """

                df = pd.read_sql(query, conn)
                logger.info(f"[DB] Retrieved {len(df)} records from PostgreSQL")
            finally:
                conn.close()

            if df.empty:
                logger.warning("[WARNING] No data found in PostgreSQL database")
                return {"status": "no_data", "records_converted": 0}

            # Convert DataFrame records to ARGOProfile objects
            logger.info("[PROCESS] Converting database records to profile objects...")
            profiles = []

            for idx, row in df.iterrows():
                try:
                    profile = ARGOProfile(
                        basin=int(row['basin']),
                        timestamp=row['timestamp'].isoformat() if pd.notna(row['timestamp']) else '',
                        cycle_number=int(row['cycle_number']),
                        vertical_sampling_scheme=str(row['vertical_sampling_scheme']) if pd.notna(row['vertical_sampling_scheme']) else '',
                        longitude=float(row['longitude']),
                        latitude=float(row['latitude']),
                        pressure_decibar=float(row['pressure_decibar']),
                        salinity_psu=float(row['salinity_psu']) if pd.notna(row['salinity_psu']) else 0.0,
                        temperature_degc=float(row['temperature_degc']) if pd.notna(row['temperature_degc']) else 0.0
                    )
                    profiles.append(profile)
                except Exception as e:
                    logger.error(f"[ERROR] Error converting row {idx}: {e}")

            logger.info(f"[SUCCESS] Successfully converted {len(profiles)} database records to profile objects")

            # Save to Parquet with generic filename for conversion mode
            self.save_to_parquet(profiles, "argo_data_converted_processed.parquet")

            end_time = datetime.now()
            duration = end_time - start_time

            summary = {
                "status": "completed",
                "records_converted": len(profiles),
                "duration_seconds": duration.total_seconds(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

            logger.info("="*60)
            logger.info("[SUCCESS] POSTGRES TO PARQUET CONVERSION COMPLETED")
            logger.info("="*60)
            logger.info(f"[RESULT] Status: {summary['status']}")
            logger.info(f"[RESULT] Records converted: {summary['records_converted']}")
            logger.info(f"[RESULT] Duration: {duration.total_seconds():.2f} seconds")
            logger.info("="*60)

            return summary

        except Exception as e:
            logger.error("[CRITICAL] CRITICAL ERROR - Postgres to Parquet conversion failed")
            logger.error(f"[ERROR] Error: {e}")
            logger.error(traceback.format_exc())

            end_time = datetime.now()
            duration = end_time - start_time

            return {
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration.total_seconds(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

    def ingest_to_postgres_only(self) -> Dict[str, Any]:
        """
        Ingest CSV data to PostgreSQL database only (no Parquet files).

        Returns:
            Dict[str, Any]: Summary statistics of the ingestion process
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("[PROCESS] STARTING POSTGRES-ONLY DATA INGESTION PROCESS")
        logger.info("="*60)

        try:
            # Find CSV files
            csv_files = self.find_csv_files()

            if not csv_files:
                logger.warning("[WARNING] No CSV files found to process")
                return {"status": "no_files", "processed": 0, "errors": 0}

            logger.info(f"[PROCESS] Found {len(csv_files)} CSV file(s) to process")

            # Get initial database count
            initial_db_count = self.db_manager.get_record_count()
            logger.info(f"[DB] Initial database record count: {initial_db_count}")

            total_processed = 0
            total_errors = 0

            for file_idx, csv_file in enumerate(csv_files, 1):
                logger.info(f"[PROCESS] Processing file {file_idx}/{len(csv_files)}: {csv_file.name}")

                try:
                    # Load CSV data
                    df = self.load_csv(csv_file)
                    original_csv_count = len(df)
                    logger.info(f"[FILE] File {csv_file.name} contains {original_csv_count} records")

                    # Process in batches
                    num_batches = (len(df) + self.batch_size - 1) // self.batch_size
                    logger.info(f"[PROCESS] Processing {num_batches} batches with batch size {self.batch_size}")

                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * self.batch_size
                        end_idx = min(start_idx + self.batch_size, len(df))
                        batch_df = df.iloc[start_idx:end_idx]

                        logger.debug(f"[BATCH] Processing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx-1})")

                        # Process batch
                        profiles = self.process_batch(batch_df, batch_idx + 1)

                        if profiles:
                            # Save to PostgreSQL only
                            db_inserted = self.save_to_postgres(profiles, batch_idx + 1)
                            total_processed += db_inserted

                        logger.info(f"[BATCH] Batch {batch_idx + 1}/{num_batches} completed - {len(profiles)} profiles processed")

                    logger.info(f"[FILE] File {csv_file.name} processing completed")

                except Exception as e:
                    total_errors += 1
                    logger.error(f"[ERROR] Failed to process file {csv_file}: {e}")
                    logger.error(traceback.format_exc())

            # Final verification
            final_db_count = self.db_manager.get_record_count()

            # Calculate processing statistics
            end_time = datetime.now()
            duration = end_time - start_time
            records_added = final_db_count - initial_db_count

            summary = {
                "status": "completed",
                "files_processed": len(csv_files),
                "records_processed": total_processed,
                "database_records_added": records_added,
                "errors": total_errors,
                "duration_seconds": duration.total_seconds(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

            logger.info("="*60)
            logger.info("[SUCCESS] POSTGRES-ONLY INGESTION PROCESS COMPLETED")
            logger.info("="*60)
            logger.info(f"[RESULT] Status: {summary['status']}")
            logger.info(f"[RESULT] Files processed: {summary['files_processed']}")
            logger.info(f"[RESULT] Records processed: {summary['records_processed']}")
            logger.info(f"[RESULT] Database records added: {summary['database_records_added']}")
            logger.info(f"[RESULT] Errors: {summary['errors']}")
            logger.info(f"[RESULT] Duration: {duration.total_seconds():.2f} seconds")
            logger.info("="*60)

            return summary

        except Exception as e:
            logger.error("[CRITICAL] CRITICAL ERROR - Postgres-only ingestion process failed")
            logger.error(f"[ERROR] Error: {e}")
            logger.error(traceback.format_exc())

            end_time = datetime.now()
            duration = end_time - start_time

            return {
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration.total_seconds(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration when configs file is not available.

        Returns:
            Dict[str, Any]: Default configuration
        """
        logger.info("[CONFIG] Using default configuration")
        return {
            'data': {
                'raw_dir': 'data/raw/',
                'processed_dir': 'data/processed/'
            },
            'ingestion': {
                'batch_size': 1000
            }
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dict containing configuration parameters
        """
        logger.debug(f"[CONFIG] Loading configuration from: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[SUCCESS] Configuration loaded successfully from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"[WARNING] Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"[ERROR] Invalid YAML configuration: {e}")
            raise

    def _setup_directories(self) -> None:
        """Create necessary directories for data processing."""
        logger.debug("[SETUP] Setting up required directories...")

        directories = [
            Path('logs'),
            Path('data/raw'),
            Path('data/processed')
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[SETUP] Directory ensured: {directory}")

        logger.info("[SUCCESS] All required directories created/verified")

    def find_csv_files(self) -> List[Path]:
        """
        Find all CSV files in the raw data directory.

        Returns:
            List[Path]: List of CSV file paths
        """
        raw_dir = Path(self.config['data']['raw_dir'])
        logger.info(f"[SEARCH] Searching for CSV files in: {raw_dir}")

        if not raw_dir.exists():
            logger.error(f"[ERROR] Raw data directory does not exist: {raw_dir}")
            return []

        csv_files = list(raw_dir.glob("*.csv"))
        logger.info(f"[SEARCH] Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            logger.info(f"[FILE] Found CSV file: {csv_file.name} ({file_size_mb:.2f} MB)")

        return csv_files

    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load and validate CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            pandas.DataFrame: Loaded and validated data
        """
        logger.info(f"[LOAD] Loading CSV file: {filepath}")

        try:
            # Get file size for logging
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"[FILE] File size: {file_size_mb:.2f} MB")

            # Read CSV with proper data types and error handling
            df = pd.read_csv(
                filepath,
                dtype={
                    'basin': 'int32',
                    'cycle_number': 'int32',
                    'longitude': 'float64',
                    'latitude': 'float64',
                    'pressure_decibar': 'float64',
                    'salinity_psu': 'float64',
                    'temperature_degC': 'float64'
                },
                parse_dates=['timestamp'],
                na_values=['', 'nan', 'NaN', 'null', 'NULL'],
                keep_default_na=True
            )

            logger.info(f"[SUCCESS] Successfully loaded {len(df)} records from CSV")
            logger.debug(f"[DATA] DataFrame columns: {list(df.columns)}")
            logger.debug(f"[DATA] DataFrame shape: {df.shape}")

            # Log data quality metrics
            initial_count = len(df)
            logger.info(f"[DATA] Initial record count: {initial_count}")

            # Check for missing values in each column
            missing_data = df.isnull().sum()
            for column, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = (missing_count / initial_count) * 100
                    logger.info(f"[DATA] Column '{column}' has {missing_count} missing values ({percentage:.2f}%)")

            # Remove records with missing critical fields only
            critical_fields = ['basin', 'timestamp', 'cycle_number', 'longitude', 'latitude', 'pressure_decibar']
            before_cleanup = len(df)
            df = df.dropna(subset=critical_fields)
            after_cleanup = len(df)

            dropped_count = before_cleanup - after_cleanup
            if dropped_count > 0:
                logger.warning(f"[WARNING] Dropped {dropped_count} records due to missing critical fields")

            # Rename temperature column for consistency if needed
            if 'temperature_degC' in df.columns:
                df = df.rename(columns={'temperature_degC': 'temperature_degc'})
                logger.debug("[DATA] Renamed temperature_degC column to temperature_degc")

            # Log final statistics
            logger.info(f"[SUCCESS] Final record count after cleanup: {len(df)}")
            logger.info(f"[SUCCESS] Data loading completed successfully")

            return df

        except FileNotFoundError:
            logger.error(f"[ERROR] CSV file not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"[ERROR] CSV file is empty: {filepath}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"[ERROR] CSV parsing error for file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error loading CSV file {filepath}: {e}")
            logger.error(traceback.format_exc())
            raise

    def process_batch(self, batch_df: pd.DataFrame, batch_num: int) -> List[ARGOProfile]:
        """
        Process a batch of DataFrame rows into ARGOProfile objects.

        Args:
            batch_df: Batch of data to process
            batch_num: Batch number for logging

        Returns:
            List[ARGOProfile]: Validated profile objects
        """
        logger.debug(f"[BATCH] Processing batch {batch_num} with {len(batch_df)} records")

        profiles = []
        batch_errors = 0

        for idx, row in batch_df.iterrows():
            try:
                # Handle NaN values by converting to appropriate defaults
                salinity_psu = float(row.get('salinity_psu', 0.0)) if pd.notna(row.get('salinity_psu')) else 0.0
                temperature_degc = float(row.get('temperature_degc', 0.0)) if pd.notna(row.get('temperature_degc')) else 0.0
                vertical_sampling_scheme = str(row.get('vertical_sampling_scheme', '')) if pd.notna(row.get('vertical_sampling_scheme')) else ''

                profile = ARGOProfile(
                    basin=int(row['basin']),
                    timestamp=row['timestamp'].isoformat() if pd.notna(row['timestamp']) else '',
                    cycle_number=int(row['cycle_number']),
                    vertical_sampling_scheme=vertical_sampling_scheme,
                    longitude=float(row['longitude']),
                    latitude=float(row['latitude']),
                    pressure_decibar=float(row['pressure_decibar']),
                    salinity_psu=salinity_psu,
                    temperature_degc=temperature_degc
                )

                if profile.validate():
                    profiles.append(profile)
                else:
                    batch_errors += 1
                    logger.debug(f"[VALIDATION] Invalid profile at row {idx} in batch {batch_num}")

            except Exception as e:
                batch_errors += 1
                logger.error(f"[ERROR] Error processing row {idx} in batch {batch_num}: {e}")
                logger.debug(f"[DEBUG] Problematic row data: {dict(row)}")

        if batch_errors > 0:
            logger.warning(f"[WARNING] Batch {batch_num} processing completed with {batch_errors} errors out of {len(batch_df)} records")
        else:
            logger.info(f"[SUCCESS] Batch {batch_num} processed successfully with {len(profiles)} valid profiles")

        return profiles

    def save_to_postgres(self, profiles: List[ARGOProfile], batch_num: int) -> int:
        """
        Save profiles to PostgreSQL database using batch insert.

        Args:
            profiles: List of ARGOProfile objects to save
            batch_num: Batch number for logging

        Returns:
            int: Number of successfully inserted records
        """
        if not profiles:
            logger.debug(f"[DB] No profiles to save for batch {batch_num}")
            return 0

        logger.debug(f"[DB] Saving batch {batch_num} with {len(profiles)} profiles to PostgreSQL")

        insert_query = """
        INSERT INTO profiles (
            basin, timestamp, cycle_number, vertical_sampling_scheme,
            longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
        ) VALUES %s
        """

        try:
            # Use direct psycopg2 connection for better control
            conn = psycopg2.connect(self.DATABASE_URL)
            cursor = conn.cursor()

            try:
                # Prepare data for batch insert
                values = [
                    (
                        profile.basin,
                        profile.timestamp,
                        profile.cycle_number,
                        profile.vertical_sampling_scheme,
                        profile.longitude,
                        profile.latitude,
                        profile.pressure_decibar,
                        profile.salinity_psu,
                        profile.temperature_degc
                    )
                    for profile in profiles
                ]

                logger.debug(f"[DB] Prepared {len(values)} records for insertion")
                logger.debug(f"[DB] Sample record: {values[0] if values else 'No records'}")

                # Execute batch insert
                execute_values(
                    cursor,
                    insert_query,
                    values,
                    template=None,
                    page_size=min(self.batch_size, 1000)
                )

                # Commit the transaction
                conn.commit()

                logger.info(f"[SUCCESS] Successfully inserted {len(profiles)} profiles from batch {batch_num} to PostgreSQL")

                return len(profiles)

            finally:
                cursor.close()
                conn.close()

        except Exception as e:
            logger.error(f"[ERROR] Failed to insert batch {batch_num} to PostgreSQL: {e}")
            logger.error(traceback.format_exc())
            # Try to rollback if connection is still available
            try:
                if conn:
                    conn.rollback()
                    conn.close()
            except:
                pass
            raise

    def save_to_parquet(self, all_profiles: List[ARGOProfile], filename: Optional[str] = None) -> None:
        """
        Save all profiles to Parquet file for analytics.

        Args:
            all_profiles: Complete list of ARGOProfile objects to save
            filename: Optional filename, if not provided uses default naming
        """
        if not all_profiles:
            logger.warning("[WARNING] No profiles to save to Parquet")
            return

        logger.info(f"[PARQUET] Saving {len(all_profiles)} profiles to Parquet file")

        try:
            # Convert profiles to DataFrame
            data = [profile.to_dict() for profile in all_profiles]
            df = pd.DataFrame(data)

            logger.debug(f"[PARQUET] Created DataFrame with shape: {df.shape}")

            # Standardize timestamp format - handle various ISO formats with timezone info
            logger.debug("[PARQUET] Standardizing timestamp formats...")
            try:
                # Use format='ISO8601' to handle various ISO timestamp formats including microseconds and timezones
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
                logger.debug("[PARQUET] Successfully converted timestamps using ISO8601 format")
            except Exception as iso_error:
                logger.warning(f"[WARNING] ISO8601 conversion failed, trying mixed format: {iso_error}")
                try:
                    # Fallback to mixed format for inconsistent timestamp formats
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                    logger.debug("[PARQUET] Successfully converted timestamps using mixed format")
                except Exception as mixed_error:
                    logger.warning(f"[WARNING] Mixed format conversion failed, trying infer: {mixed_error}")
                    # Final fallback - let pandas infer the format
                    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, utc=True)
                    logger.debug("[PARQUET] Successfully converted timestamps using inferred format")

            # Determine output path
            processed_dir = Path('data/processed')
            processed_dir.mkdir(parents=True, exist_ok=True)

            if filename:
                parquet_path = processed_dir / filename
            else:
                # Use default naming convention
                parquet_path = processed_dir / 'argo_profiles_processed.parquet'

            logger.info(f"[PARQUET] Parquet file path: {parquet_path}")

            # Create PyArrow table and write to file
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path)

            # Log file information
            if parquet_path.exists():
                file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
                logger.info(f"[SUCCESS] Successfully saved {len(all_profiles)} profiles to Parquet file")
                logger.info(f"[FILE] Parquet file size: {file_size_mb:.2f} MB")

        except Exception as e:
            logger.error(f"[ERROR] Failed to save profiles to Parquet: {e}")
            logger.error(traceback.format_exc())
            raise

    def verify_data_integrity(self, original_count: int, db_count: int, parquet_count: int) -> bool:
        """
        Verify that all data has been correctly saved to both destinations.

        Args:
            original_count: Number of records in original CSV
            db_count: Number of records saved to database
            parquet_count: Number of records saved to Parquet

        Returns:
            bool: True if data integrity is maintained
        """
        logger.info("[VERIFY] Verifying data integrity...")
        logger.info(f"[VERIFY] Original CSV records: {original_count}")
        logger.info(f"[VERIFY] Database records: {db_count}")
        logger.info(f"[VERIFY] Parquet records: {parquet_count}")

        # Check if all records are preserved
        if db_count == original_count and parquet_count == original_count:
            logger.info("[SUCCESS] Data integrity verification PASSED - All records preserved")
            return True
        else:
            logger.error("[ERROR] Data integrity verification FAILED - Record count mismatch")
            if db_count != original_count:
                logger.error(f"[ERROR] Database missing {original_count - db_count} records")
            if parquet_count != original_count:
                logger.error(f"[ERROR] Parquet missing {original_count - parquet_count} records")
            return False

    def ingest_all(self) -> Dict[str, Any]:
        """
        Main ingestion method to process all CSV files in the raw directory.

        Returns:
            Dict[str, Any]: Summary statistics of the ingestion process
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("[PROCESS] STARTING ARGO DATA INGESTION PROCESS")
        logger.info("="*60)

        try:
            # Find CSV files
            csv_files = self.find_csv_files()

            if not csv_files:
                logger.warning("[WARNING] No CSV files found to process")
                return {"status": "no_files", "processed": 0, "errors": 0}

            logger.info(f"[PROCESS] Found {len(csv_files)} CSV file(s) to process")

            # Get initial database count
            initial_db_count = self.db_manager.get_record_count()
            logger.info(f"[DB] Initial database record count: {initial_db_count}")

            total_processed = 0
            total_errors = 0
            all_profiles_by_file = {}  # Store profiles by file for separate parquet files

            for file_idx, csv_file in enumerate(csv_files, 1):
                logger.info(f"[PROCESS] Processing file {file_idx}/{len(csv_files)}: {csv_file.name}")

                try:
                    # Load CSV data
                    df = self.load_csv(csv_file)
                    original_csv_count = len(df)
                    logger.info(f"[FILE] File {csv_file.name} contains {original_csv_count} records")

                    file_profiles = []

                    # Process in batches
                    num_batches = (len(df) + self.batch_size - 1) // self.batch_size
                    logger.info(f"[PROCESS] Processing {num_batches} batches with batch size {self.batch_size}")

                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * self.batch_size
                        end_idx = min(start_idx + self.batch_size, len(df))
                        batch_df = df.iloc[start_idx:end_idx]

                        logger.debug(f"[BATCH] Processing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx-1})")

                        # Process batch
                        profiles = self.process_batch(batch_df, batch_idx + 1)

                        if profiles:
                            # Save to PostgreSQL
                            db_inserted = self.save_to_postgres(profiles, batch_idx + 1)
                            total_processed += db_inserted

                            # Collect profiles for Parquet
                            file_profiles.extend(profiles)

                        logger.info(f"[BATCH] Batch {batch_idx + 1}/{num_batches} completed - {len(profiles)} profiles processed")

                    # Store profiles for this file
                    all_profiles_by_file[csv_file.name] = file_profiles
                    logger.info(f"[FILE] File {csv_file.name} processing completed - {len(file_profiles)} profiles collected")

                except Exception as e:
                    total_errors += 1
                    logger.error(f"[ERROR] Failed to process file {csv_file}: {e}")
                    logger.error(traceback.format_exc())

            # Save each file's profiles to separate Parquet files with proper naming
            for csv_filename, file_profiles in all_profiles_by_file.items():
                if file_profiles:
                    parquet_filename = self._get_processed_filename(csv_filename)
                    logger.info(f"[PARQUET] Saving {len(file_profiles)} profiles from {csv_filename} to {parquet_filename}")
                    self.save_to_parquet(file_profiles, parquet_filename)

            # Calculate totals for all files
            total_parquet_records = sum(len(profiles) for profiles in all_profiles_by_file.values())

            # Final verification
            final_db_count = self.db_manager.get_record_count()

            # Calculate processing statistics
            end_time = datetime.now()
            duration = end_time - start_time
            records_added = final_db_count - initial_db_count

            # Verify data integrity
            integrity_check = self.verify_data_integrity(
                total_processed, records_added, total_parquet_records
            )

            summary = {
                "status": "completed",
                "files_processed": len(csv_files),
                "records_processed": total_processed,
                "database_records_added": records_added,
                "parquet_records": total_parquet_records,
                "errors": total_errors,
                "duration_seconds": duration.total_seconds(),
                "data_integrity": integrity_check,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "parquet_files_created": list(all_profiles_by_file.keys())
            }

            logger.info("="*60)
            logger.info("[SUCCESS] INGESTION PROCESS COMPLETED")
            logger.info("="*60)
            logger.info(f"[RESULT] Status: {summary['status']}")
            logger.info(f"[RESULT] Files processed: {summary['files_processed']}")
            logger.info(f"[RESULT] Records processed: {summary['records_processed']}")
            logger.info(f"[RESULT] Database records added: {summary['database_records_added']}")
            logger.info(f"[RESULT] Parquet records: {summary['parquet_records']}")
            logger.info(f"[RESULT] Parquet files created: {len(all_profiles_by_file)}")
            logger.info(f"[RESULT] Errors: {summary['errors']}")
            logger.info(f"[RESULT] Duration: {duration.total_seconds():.2f} seconds")
            logger.info(f"[RESULT] Data integrity: {'PASSED' if integrity_check else 'FAILED'}")
            logger.info("="*60)

            return summary

        except Exception as e:
            logger.error("[CRITICAL] CRITICAL ERROR - Ingestion process failed")
            logger.error(f"[ERROR] Error: {e}")
            logger.error(traceback.format_exc())

            end_time = datetime.now()
            duration = end_time - start_time

            return {
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration.total_seconds(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }


def main():
    """
    Main entry point for the ingestion script.
    Can be run standalone for batch processing, Parquet conversion only, or postgres-only mode.
    """
    import argparse

    parser = argparse.ArgumentParser(description='ARGO Data Ingestion Tool')
    parser.add_argument('--mode', choices=['full', 'parquet-only', 'postgres-only'], default='full',
                       help='Run mode: full ingestion, parquet conversion only, or postgres-only mode')

    args = parser.parse_args()

    logger.info(f"[INIT] Starting ARGO Data Ingestion Script - Mode: {args.mode}")

    try:
        ingestor = ARGOIngestor()

        if args.mode == 'parquet-only':
            logger.info("[PROCESS] Running in Parquet-only mode - converting PostgreSQL data to Parquet")
            result = ingestor.convert_postgres_to_parquet()

            if result["status"] == "completed":
                print(f"[SUCCESS] Parquet conversion successful: {result['records_converted']} records converted")
            elif result["status"] == "no_data":
                print("[WARNING] No data found in PostgreSQL database to convert")
            else:
                print(f"[ERROR] Parquet conversion failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        elif args.mode == 'postgres-only':
            logger.info("[PROCESS] Running in Postgres-only mode - ingesting CSV data to PostgreSQL only")
            result = ingestor.ingest_to_postgres_only()

            if result["status"] == "completed":
                print(f"[SUCCESS] Postgres-only ingestion successful: {result['records_processed']} records processed")
                print(f"[INFO] Database records added: {result['database_records_added']}")
                if result["errors"] > 0:
                    print(f"[WARNING] {result['errors']} errors encountered")
            elif result["status"] == "no_files":
                print("[WARNING] No CSV files found in data/raw/ directory")
                print("[INFO] Please add CSV files to the data/raw/ directory and try again")
            else:
                print(f"[ERROR] Postgres-only ingestion failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        else:
            logger.info("[PROCESS] Running in full ingestion mode")
            result = ingestor.ingest_all()

            if result["status"] == "completed":
                print(f"[SUCCESS] Ingestion successful: {result['records_processed']} records processed")
                print(f"[INFO] Database records added: {result['database_records_added']}")
                print(f"[INFO] Parquet files created: {len(result.get('parquet_files_created', []))}")
                if result["errors"] > 0:
                    print(f"[WARNING] {result['errors']} errors encountered")
                if not result.get("data_integrity", False):
                    print("[WARNING] Data integrity check failed")
            elif result["status"] == "no_files":
                print("[WARNING] No CSV files found in data/raw/ directory")
                print("[INFO] Please add CSV files to the data/raw/ directory and try again")
            else:
                print(f"[ERROR] Ingestion failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"[CRITICAL] Main execution failed: {e}")
        logger.error(traceback.format_exc())
        print(f"[CRITICAL] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()