"""
FloatChat Data Splitter for RAG Preparation

This module splits ARGO oceanographic data into chunks suitable for embedding.
Each chunk represents a single timestamp with all its associated measurements,
preserving variations in vertical_sampling_scheme, pressure, temperature, and salinity
while deduplicating constant metadata fields.

Author: FloatChat Team
Created: 2025-01-XX
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import yaml


class DataSplitterError(Exception):
    """Custom exception for DataSplitter operations."""
    pass


class DataSplitter:
    """
    Splits structured ARGO data into smaller chunks for embedding generation.

    Groups data by timestamp to create coherent chunks that maintain the relationship
    between oceanographic measurements while removing redundant metadata duplication.
    Each chunk contains all vertical profile data for a specific timestamp.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary loaded from intel.yaml
        logger (logging.Logger): Logger instance for operation tracking
        parquet_path (Path): Path to the processed parquet file
        db_url (str): PostgreSQL database connection URL
        chunk_size_limit (int): Maximum size limit for individual chunks
        overlap_strategy (str): Strategy for handling overlapping data
    """

    def __init__(self, config_path: str = "configs/intel.yaml", chunk_size_limit: int = 2000):
        """
        Initialize the DataSplitter with configuration and setup logging.

        Args:
            config_path (str): Path to the configuration YAML file
            chunk_size_limit (int): Maximum character limit for chunks

        Raises:
            DataSplitterError: If configuration loading or initialization fails
        """
        self.chunk_size_limit = chunk_size_limit
        self.overlap_strategy = "timestamp_based"

        try:
            # Setup logging first
            self._setup_logging()
            self.logger.info("[INIT] Initializing DataSplitter")

            # Load configuration
            self.config = self._load_config(config_path)
            self.logger.info(f"[CONFIG] Configuration loaded from {config_path}")

            # Set paths and database URL
            self.parquet_path = Path(self.config.get('data', {}).get('processed_parquet',
                                                                     'data/processed/argo_profiles.parquet'))
            self.db_url = 'postgresql://postgres:Strong.password177013@localhost:6000/floatchat'

            # Validate paths
            self._validate_setup()

            self.logger.info("[INIT] DataSplitter initialization completed successfully")

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"[ERROR] Failed to initialize DataSplitter: {str(e)}")
            raise DataSplitterError(f"Initialization failed: {str(e)}") from e

    def _setup_logging(self) -> None:
        """
        Setup logging configuration with timestamped log files.

        Creates logs directory if it doesn't exist and configures logger
        to write to both console and timestamped log file.
        """
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        log_filename = logs_dir / f"{timestamp}.log"

        # Configure logger
        self.logger = logging.getLogger(f"DataSplitter_{timestamp}")
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"[LOGGING] Logging initialized - Log file: {log_filename}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            DataSplitterError: If configs file cannot be loaded or parsed
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                raise ValueError("Configuration file is empty or invalid")

            return config

        except Exception as e:
            raise DataSplitterError(f"Failed to load configuration: {str(e)}") from e

    def _validate_setup(self) -> None:
        """
        Validate that required files and database connections are available.

        Raises:
            DataSplitterError: If validation fails
        """
        try:
            # Check parquet file
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

            self.logger.info(f"[VALIDATION] Parquet file validated: {self.parquet_path}")

            # Test database connection
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            self.logger.info("[VALIDATION] Database connection validated")

        except Exception as e:
            raise DataSplitterError(f"Setup validation failed: {str(e)}") from e

    def load_data_from_parquet(self) -> pd.DataFrame:
        """
        Load ARGO data from parquet file.

        Returns:
            pd.DataFrame: Loaded dataframe with ARGO data

        Raises:
            DataSplitterError: If data loading fails
        """
        try:
            self.logger.info(f"[DATA_LOAD] Loading data from parquet: {self.parquet_path}")

            df = pd.read_parquet(self.parquet_path)

            # Validate required columns
            required_columns = [
                'basin', 'timestamp', 'cycle_number', 'vertical_sampling_scheme',
                'longitude', 'latitude', 'pressure_decibar', 'salinity_psu', 'temperature_degc'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            self.logger.info(f"[DATA_LOAD] Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            self.logger.info(f"[DATA_LOAD] Unique timestamps: {df['timestamp'].nunique()}")

            return df

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load data from parquet: {str(e)}")
            raise DataSplitterError(f"Data loading failed: {str(e)}") from e

    def load_data_from_postgres(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load ARGO data from PostgreSQL database.

        Args:
            limit (Optional[int]): Limit number of rows to load (for testing)

        Returns:
            pd.DataFrame: Loaded dataframe with ARGO data

        Raises:
            DataSplitterError: If data loading fails
        """
        try:
            self.logger.info("[DATA_LOAD] Loading data from PostgreSQL database")

            engine = create_engine(self.db_url)

            # Construct query
            query = """
                    SELECT basin, timestamp, cycle_number, vertical_sampling_scheme, longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
                    FROM profiles
                    ORDER BY timestamp, pressure_decibar \
                    """

            if limit:
                query += f" LIMIT {limit}"
                self.logger.info(f"[DATA_LOAD] Applying row limit: {limit}")

            df = pd.read_sql(query, engine)

            self.logger.info(f"[DATA_LOAD] Successfully loaded {len(df)} rows from database")
            self.logger.info(f"[DATA_LOAD] Unique timestamps: {df['timestamp'].nunique()}")

            return df

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load data from PostgreSQL: {str(e)}")
            raise DataSplitterError(f"Database loading failed: {str(e)}") from e

    def create_timestamp_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Split dataframe into chunks based on timestamp grouping.

        Each chunk contains all measurements for a specific timestamp,
        with constant metadata deduplicated and variable measurements preserved.

        Args:
            df (pd.DataFrame): Input dataframe with ARGO data

        Returns:
            List[Dict[str, Any]]: List of chunks, each containing:
                - metadata: Common fields for the timestamp
                - measurements: List of depth profile measurements
                - chunk_id: Unique identifier
                - source: Data source identifier
        """
        try:
            self.logger.info("[CHUNKING] Starting timestamp-based chunking")

            chunks = []
            grouped = df.groupby('timestamp')

            self.logger.info(f"[CHUNKING] Processing {len(grouped)} unique timestamps")

            for timestamp, group in grouped:
                try:
                    chunk = self._create_single_timestamp_chunk(timestamp, group, len(chunks))

                    # Validate chunk size
                    chunk_text = self._chunk_to_text(chunk)
                    if len(chunk_text) > self.chunk_size_limit:
                        self.logger.warning(
                            f"[CHUNKING] Chunk {chunk['chunk_id']} exceeds size limit: "
                            f"{len(chunk_text)} > {self.chunk_size_limit}"
                        )
                        # Split large chunks if needed
                        sub_chunks = self._split_large_chunk(chunk)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk)

                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to process timestamp {timestamp}: {str(e)}")
                    continue

            self.logger.info(f"[CHUNKING] Successfully created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"[ERROR] Chunking process failed: {str(e)}")
            raise DataSplitterError(f"Chunking failed: {str(e)}") from e

    def _create_single_timestamp_chunk(self, timestamp: Any, group: pd.DataFrame,
                                       chunk_index: int) -> Dict[str, Any]:
        """
        Create a single chunk for a specific timestamp.

        Args:
            timestamp: The timestamp value
            group (pd.DataFrame): All rows for this timestamp
            chunk_index (int): Index for chunk ID generation

        Returns:
            Dict[str, Any]: Formatted chunk dictionary
        """
        # Extract constant metadata (should be same for all rows in group)
        first_row = group.iloc[0]

        metadata = {
            'timestamp': str(timestamp),
            'basin': first_row['basin'],
            'cycle_number': first_row['cycle_number'],
            'vertical_sampling_scheme': first_row['vertical_sampling_scheme'],
            'longitude': first_row['longitude'],
            'latitude': first_row['latitude'],
            'profile_count': len(group)
        }

        # Extract variable measurements (pressure, temperature, salinity)
        measurements = []
        for _, row in group.iterrows():
            measurement = {
                'pressure_decibar': row['pressure_decibar'],
                'temperature_degc': row['temperature_degc'],
                'salinity_psu': row['salinity_psu']
            }
            measurements.append(measurement)

        # Sort measurements by pressure for consistent ordering
        measurements = sorted(measurements, key=lambda x: x['pressure_decibar'])

        chunk = {
            'chunk_id': f"argo_timestamp_{chunk_index:06d}",
            'timestamp': str(timestamp),
            'metadata': metadata,
            'measurements': measurements,
            'source': 'argo_floats',
            'chunk_type': 'temporal_profile'
        }

        return chunk

    def _chunk_to_text(self, chunk: Dict[str, Any]) -> str:
        """
        Convert chunk dictionary to text representation for size estimation.

        Args:
            chunk (Dict[str, Any]): Chunk dictionary

        Returns:
            str: Text representation of chunk
        """
        try:
            # Create human-readable text representation
            metadata = chunk['metadata']
            measurements = chunk['measurements']

            text_parts = [
                f"ARGO Float Profile - Timestamp: {metadata['timestamp']}",
                f"Location: {metadata['latitude']:.6f}°N, {metadata['longitude']:.6f}°E",
                f"Basin: {metadata['basin']}, Cycle: {metadata['cycle_number']}",
                f"Sampling Scheme: {metadata['vertical_sampling_scheme']}",
                f"Profile Depth Points: {len(measurements)}",
                ""
            ]

            # Add measurement summary
            text_parts.append("Depth Profile Measurements:")
            for i, measurement in enumerate(measurements):
                pressure = measurement['pressure_decibar']
                temp = measurement['temperature_degc']
                salinity = measurement['salinity_psu']
                text_parts.append(f"  {i + 1:3d}. Pressure: {pressure:6.1f} dbar, "
                                  f"Temperature: {temp:6.3f}°C, "
                                  f"Salinity: {salinity:6.3f} PSU")

            return "\n".join(text_parts)

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to convert chunk to text: {str(e)}")
            return json.dumps(chunk, default=str)

    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a large chunk into smaller sub-chunks.

        Args:
            chunk (Dict[str, Any]): Large chunk to split

        Returns:
            List[Dict[str, Any]]: List of smaller chunks
        """
        self.logger.info(f"[SPLIT] Splitting large chunk {chunk['chunk_id']}")

        measurements = chunk['measurements']
        metadata = chunk['metadata'].copy()

        # Calculate measurements per sub-chunk
        estimated_chars_per_measurement = 80  # Rough estimate
        max_measurements = max(1, self.chunk_size_limit // estimated_chars_per_measurement - 10)

        sub_chunks = []
        for i in range(0, len(measurements), max_measurements):
            sub_measurements = measurements[i:i + max_measurements]

            # Update metadata for sub-chunk
            sub_metadata = metadata.copy()
            sub_metadata['profile_count'] = len(sub_measurements)
            sub_metadata['sub_chunk_info'] = f"Part {len(sub_chunks) + 1}, depths {i + 1}-{i + len(sub_measurements)}"

            sub_chunk = {
                'chunk_id': f"{chunk['chunk_id']}_part_{len(sub_chunks):02d}",
                'timestamp': chunk['timestamp'],
                'metadata': sub_metadata,
                'measurements': sub_measurements,
                'source': chunk['source'],
                'chunk_type': 'temporal_profile_split'
            }

            sub_chunks.append(sub_chunk)

        self.logger.info(f"[SPLIT] Created {len(sub_chunks)} sub-chunks")
        return sub_chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics about the created chunks.

        Args:
            chunks (List[Dict[str, Any]]): List of chunks to analyze

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        try:
            if not chunks:
                return {'error': 'No chunks provided'}

            chunk_sizes = [len(self._chunk_to_text(chunk)) for chunk in chunks]
            measurement_counts = [len(chunk['measurements']) for chunk in chunks]

            stats = {
                'total_chunks': len(chunks),
                'chunk_size_stats': {
                    'min': min(chunk_sizes),
                    'max': max(chunk_sizes),
                    'mean': sum(chunk_sizes) / len(chunk_sizes),
                    'median': sorted(chunk_sizes)[len(chunk_sizes) // 2]
                },
                'measurements_per_chunk_stats': {
                    'min': min(measurement_counts),
                    'max': max(measurement_counts),
                    'mean': sum(measurement_counts) / len(measurement_counts),
                    'total_measurements': sum(measurement_counts)
                },
                'chunk_types': {chunk_type: sum(1 for c in chunks if c.get('chunk_type') == chunk_type)
                                for chunk_type in set(c.get('chunk_type', 'unknown') for c in chunks)},
                'oversized_chunks': sum(1 for size in chunk_sizes if size > self.chunk_size_limit)
            }

            self.logger.info(f"[STATS] Generated statistics for {len(chunks)} chunks")
            return stats

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to generate statistics: {str(e)}")
            return {'error': str(e)}

    def get_chunk_text_representation(self, chunk: Dict[str, Any]) -> str:
        """
        Get text representation of a chunk for embedding generation.

        This is the primary method used by embedding ingestors to convert
        structured chunks into text suitable for embedding models.

        Args:
            chunk (Dict[str, Any]): Chunk dictionary

        Returns:
            str: Clean text representation optimized for embeddings
        """
        return self._chunk_to_text(chunk)

    def run_chunking_pipeline(self, data_source: str = "parquet",
                              limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the data chunking pipeline without saving.

        This method is designed to be called by other modules (like ingest_embeddings.py)
        that need chunked data for further processing.

        Args:
            data_source (str): Data source ("parquet" or "postgres")
            limit (Optional[int]): Limit number of rows (for testing)

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: Chunks and statistics
        """
        try:
            self.logger.info("[PIPELINE] Starting data chunking pipeline")

            # Load data
            if data_source.lower() == "parquet":
                df = self.load_data_from_parquet()
            elif data_source.lower() == "postgres":
                df = self.load_data_from_postgres(limit=limit)
            else:
                raise ValueError(f"Invalid data source: {data_source}")

            # Apply row limit if specified (for parquet source)
            if limit and data_source.lower() == "parquet":
                df = df.head(limit)
                self.logger.info(f"[PIPELINE] Applied row limit: {limit}")

            # Create chunks
            chunks = self.create_timestamp_chunks(df)

            # Generate statistics
            stats = self.get_chunk_statistics(chunks)

            # Log pipeline completion
            self.logger.info(f"[SUCCESS] Chunking pipeline completed successfully")
            self.logger.info(f"[SUMMARY] Processed {len(df)} rows into {len(chunks)} chunks")

            return chunks, stats

        except Exception as e:
            self.logger.error(f"[ERROR] Chunking pipeline execution failed: {str(e)}")
            raise DataSplitterError(f"Chunking pipeline failed: {str(e)}") from e


def main():
    """
    Main function for command-line usage of DataSplitter.
    """
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="FloatChat Data Splitter - Split ARGO oceanographic data into chunks for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/rag/splitter.py                                    # Use defaults (parquet source)
  python src/rag/splitter.py --source postgres                 # Use PostgreSQL source
  python src/rag/splitter.py --output data/custom_chunks.json  # Custom output path
  python src/rag/splitter.py --limit 1000                      # Process only 1000 rows
  python src/rag/splitter.py --chunk-size 1500                 # Custom chunk size limit
  python src/rag/splitter.py --configs configs/custom.yaml      # Custom configs file
        """
    )

    parser.add_argument(
        "--source", "-s",
        choices=["parquet", "postgres"],
        default="parquet",
        help="Data source to use (default: parquet)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed/argo_chunks.json",
        help="Output path for processed chunks (default: data/processed/argo_chunks.json)"
    )

    parser.add_argument(
        "--configs", "-c",
        type=str,
        default="configs/intel.yaml",
        help="Path to configuration file (default: configs/intel.yaml)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Maximum chunk size in characters (default: 2000)"
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of rows to process (useful for testing)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Generate and display statistics only, don't save chunks"
    )

    # Parse arguments
    args = parser.parse_args()

    # Display startup banner
    print("=" * 60)
    print("FloatChat Data Splitter")
    print("Production-Ready RAG Data Processing Pipeline")
    print("=" * 60)
    print(f"Data Source: {args.source}")
    print(f"Output Path: {args.output}")
    print(f"Config File: {args.config}")
    print(f"Chunk Size Limit: {args.chunk_size} characters")
    if args.limit:
        print(f"Row Limit: {args.limit}")
    print("-" * 60)

    try:
        # Initialize splitter with custom parameters
        splitter = DataSplitter(
            config_path=args.config,
            chunk_size_limit=args.chunk_size
        )

        # Run pipeline for standalone usage (testing/development)
        print("[INFO] Starting data chunking pipeline...")
        chunks, stats = splitter.run_chunking_pipeline(
            data_source=args.source,
            limit=args.limit
        )

        # Save chunks if not in stats-only mode (for standalone testing)
        if not args.stats_only:
            # Import json for standalone saving
            import json
            from pathlib import Path

            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Create summary for standalone usage
            summary = {
                'total_chunks': len(chunks),
                'creation_timestamp': datetime.now().isoformat(),
                'chunk_size_limit': args.chunk_size,
                'chunks': chunks
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

            print(f"[INFO] Chunks saved to {args.output} for testing purposes")

        # Display results
        print("\n" + "=" * 60)
        print("PROCESSING RESULTS")
        print("=" * 60)
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Total measurements processed: {stats['measurements_per_chunk_stats']['total_measurements']}")

        print(f"\nChunk Size Statistics:")
        print(f"  - Average: {stats['chunk_size_stats']['mean']:.1f} characters")
        print(f"  - Minimum: {stats['chunk_size_stats']['min']} characters")
        print(f"  - Maximum: {stats['chunk_size_stats']['max']} characters")
        print(f"  - Oversized chunks: {stats['oversized_chunks']}")

        print(f"\nMeasurements per Chunk:")
        print(f"  - Average: {stats['measurements_per_chunk_stats']['mean']:.1f}")
        print(f"  - Minimum: {stats['measurements_per_chunk_stats']['min']}")
        print(f"  - Maximum: {stats['measurements_per_chunk_stats']['max']}")

        if stats['chunk_types']:
            print(f"\nChunk Types:")
            for chunk_type, count in stats['chunk_types'].items():
                print(f"  - {chunk_type}: {count}")

        if not args.stats_only:
            print(f"\nOutput saved to: {args.output}")
        else:
            print("\n[INFO] Statistics-only mode - chunks not saved")

        # Verbose output
        if args.verbose:
            print(f"\nDetailed Statistics:")
            print(f"  - Median chunk size: {stats['chunk_size_stats']['median']} characters")
            print(f"  - Chunk size limit: {args.chunk_size} characters")
            print(f"  - Data source: {args.source}")
            if args.limit:
                print(f"  - Row processing limit: {args.limit}")

        print("\n" + "=" * 60)
        print("[SUCCESS] Data splitting pipeline completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Operation cancelled by user")
        sys.exit(130)  # Standard exit code for Ctrl+C

    except DataSplitterError as e:
        print(f"\n[ERROR] Data splitter error: {str(e)}")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()