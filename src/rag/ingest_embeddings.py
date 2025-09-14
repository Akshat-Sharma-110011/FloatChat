# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\MLOps-Projects\FloatChat\src\rag\ingest_embeddings.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2025-09-13 15:40:43 UTC (1757778043)

"""
FloatChat Embedding Ingestor

Generates embeddings for ARGO oceanographic data chunks and stores them in FAISS vector database.
Uses the DataSplitter to chunk data, then creates embeddings and builds the searchable index.

This module follows the RAG architecture pattern: Load -> Split -> Embed -> Store

Author: FloatChat Team
Created: 2025-01-XX
"""
import logging
import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
import numpy as np
import faiss
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.rag.splitter import DataSplitter, DataSplitterError

class EmbeddingIngestorError(Exception):
    """Custom exception for EmbeddingIngestor operations."""

class EmbeddingIngestor:
    """
    Generates embeddings for all document chunks and saves them to a FAISS vector database.

    This class orchestrates the complete embedding pipeline:
    1. Uses DataSplitter to chunk ARGO data by timestamp
    2. Generates embeddings using sentence-transformers
    3. Builds FAISS index for similarity search
    4. Saves index and metadata for retrieval

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        splitter (DataSplitter): Data splitting utility
        embedding_model (SentenceTransformer): Embedding generation model
        index_dir (Path): Directory for storing FAISS index
        vector_store (faiss.Index): FAISS index for vector storage
        dimension (int): Embedding vector dimension
    """

    def __init__(self, config_path: str='configs/intel.yaml'):
        """
        Initialize the EmbeddingIngestor with configuration and setup components.

        Args:
            config_path (str): Path to the configuration YAML file

        Raises:
            EmbeddingIngestorError: If initialization fails
        """
        try:
            self._setup_logging()
            self.logger.info('[INIT] Initializing EmbeddingIngestor')
            self.config = self._load_config(config_path)
            self.logger.info(f'[CONFIG] Configuration loaded from {config_path}')
            self.index_dir = Path(self.config.get('vector_store', {}).get('path', 'data/index/faiss_index'))
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.splitter = DataSplitter(config_path=config_path)
            self.logger.info('[INIT] DataSplitter initialized')
            model_name = self.config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
            self._load_embedding_model(model_name)
            self.vector_store = None
            self.document_metadata = []
            self.logger.info('[INIT] EmbeddingIngestor initialization completed successfully')
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f'[ERROR] Failed to initialize EmbeddingIngestor: {str(e)}')
            raise EmbeddingIngestorError(f'Initialization failed: {str(e)}') from e

    def _setup_logging(self) -> None:
        """
        Setup logging configuration with timestamped log files.
        """
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        log_filename = logs_dir / f'embedding_ingestor_{timestamp}.log'
        self.logger = logging.getLogger(f'EmbeddingIngestor_{timestamp}')
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.info(f'[LOGGING] Embedding ingestor logging initialized - Log file: {log_filename}')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f'Configuration file not found: {config_path}')
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError('Configuration file is empty or invalid')
            return config
        except Exception as e:
            raise EmbeddingIngestorError(f'Failed to load configuration: {str(e)}') from e

    def _load_embedding_model(self, model_name: str) -> None:
        """
        Load the sentence transformer model for embedding generation.

        Args:
            model_name (str): Name or path of the sentence transformer model
        """
        try:
            self.logger.info(f'[MODEL] Loading embedding model: {model_name}')
            warnings.filterwarnings('ignore', message='.*resume_download.*')
            warnings.filterwarnings('ignore', message='.*torch.utils.checkpoint.*')
            self.embedding_model = SentenceTransformer(model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f'[MODEL] Model loaded successfully - Dimension: {self.dimension}')
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to load embedding model: {str(e)}')
            raise EmbeddingIngestorError(f'Model loading failed: {str(e)}') from e

    def _prepare_chunks_for_embedding(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Convert chunks to text representations suitable for embedding.

        Args:
            chunks (List[Dict[str, Any]]): List of document chunks

        Returns:
            List[str]: List of text representations
        """
        try:
            self.logger.info(f'[PREPARE] Preparing {len(chunks)} chunks for embedding')
            text_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    text = self.splitter.get_chunk_text_representation(chunk)
                    text_chunks.append(text)
                    metadata = {'chunk_id': chunk.get('chunk_id', f'chunk_{i}'), 'timestamp': chunk.get('timestamp'), 'source': chunk.get('source', 'argo_floats'), 'chunk_type': chunk.get('chunk_type', 'temporal_profile'), 'metadata': chunk.get('metadata', {}), 'measurement_count': len(chunk.get('measurements', [])), 'original_chunk': chunk}
                    self.document_metadata.append(metadata)
                except Exception as e:
                    self.logger.warning(f'[WARNING] Failed to process chunk {i}: {str(e)}')
            self.logger.info(f'[PREPARE] Successfully prepared {len(text_chunks)} text chunks')
            return text_chunks
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to prepare chunks for embedding: {str(e)}')
            raise EmbeddingIngestorError(f'Chunk preparation failed: {str(e)}') from e

    def _generate_embeddings(self, text_chunks: List[str], batch_size: int=32) -> np.ndarray:
        """
        Generate embeddings for text chunks using the sentence transformer model.

        Args:
            text_chunks (List[str]): List of text chunks to embed
            batch_size (int): Batch size for embedding generation

        Returns:
            np.ndarray: Array of embeddings with shape (n_chunks, embedding_dimension)
        """
        try:
            self.logger.info(f'[EMBED] Generating embeddings for {len(text_chunks)} chunks')
            self.logger.info(f'[EMBED] Using batch size: {batch_size}')
            embeddings = []
            for i in tqdm(range(0, len(text_chunks), batch_size), desc='Generating embeddings', unit='batch'):
                batch = text_chunks[i:i + batch_size]
                try:
                    batch_embeddings = self.embedding_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    self.logger.error(f'[ERROR] Failed to process batch {i // batch_size}: {str(e)}')
                    raise
            all_embeddings = np.vstack(embeddings)
            self.logger.info(f'[EMBED] Successfully generated embeddings - Shape: {all_embeddings.shape}')
            return all_embeddings
        except Exception as e:
            self.logger.error(f'[ERROR] Embedding generation failed: {str(e)}')
            raise EmbeddingIngestorError(f'Embedding generation failed: {str(e)}') from e

    def _create_faiss_index(self, embeddings: np.ndarray, index_type: str='IndexFlatIP') -> faiss.Index:
        """
        Create and populate FAISS index with embeddings.

        Args:
            embeddings (np.ndarray): Array of embeddings
            index_type (str): Type of FAISS index to create

        Returns:
            faiss.Index: Populated FAISS index
        """
        try:
            self.logger.info(f'[INDEX] Creating FAISS index - Type: {index_type}')
            self.logger.info(f'[INDEX] Embedding shape: {embeddings.shape}')
            if index_type == 'IndexFlatIP':
                index = faiss.IndexFlatIP(self.dimension)
            elif index_type == 'IndexFlatL2':
                index = faiss.IndexFlatL2(self.dimension)
            elif index_type == 'IndexHNSWFlat':
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                self.logger.warning(f'[WARNING] Unknown index type {index_type}, using IndexFlatIP')
                index = faiss.IndexFlatIP(self.dimension)
            self.logger.info('[INDEX] Adding embeddings to FAISS index...')
            index.add(embeddings.astype(np.float32))
            self.logger.info(f'[INDEX] FAISS index created successfully - Total vectors: {index.ntotal}')
            return index
        except Exception as e:
            self.logger.error(f'[ERROR] FAISS index creation failed: {str(e)}')
            raise EmbeddingIngestorError(f'Index creation failed: {str(e)}') from e

    def _save_index_and_metadata(self, index: faiss.Index) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            index (faiss.Index): FAISS index to save
        """
        try:
            self.logger.info(f'[SAVE] Saving index and metadata to {self.index_dir}')
            index_file = self.index_dir / 'faiss.index'
            faiss.write_index(index, str(index_file))
            self.logger.info(f'[SAVE] FAISS index saved: {index_file}')
            metadata_file = self.index_dir / 'metadata.pkl'
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            self.logger.info(f'[SAVE] Metadata saved: {metadata_file}')
            config_file = self.index_dir / 'config_snapshot.yaml'
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f'[SAVE] Configuration snapshot saved: {config_file}')
            info_file = self.index_dir / 'index_info.json'
            index_info = {'creation_timestamp': datetime.now().isoformat(), 'total_vectors': int(index.ntotal), 'vector_dimension': self.dimension, 'index_type': type(index).__name__, 'embedding_model': self.config.get('embeddings', {}).get('model_name'), 'total_chunks': len(self.document_metadata), 'index_size_bytes': os.path.getsize(metadata_file) if metadata_file.exists() else 0, 'metadata_size_bytes': os.path.getsize(self.document_metadata) if self.document_metadata else 0}
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(index_info, f, indent=2)
            self.logger.info(f'[SAVE] Index info saved: {info_file}')
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to save index and metadata: {str(e)}')
            raise EmbeddingIngestorError(f'Saving failed: {str(e)}') from e
    pass
    pass
    pass
    pass

    def run_embedding_pipeline(self, data_source: str='parquet', limit: Optional[int]=None, batch_size: int=32, index_type: str='IndexFlatIP') -> Dict[str, Any]:
        """
        Run the complete embedding ingestion pipeline.

        Args:
            data_source (str): Data source ("parquet" or "postgres")
            limit (Optional[int]): Limit number of rows to process
            batch_size (int): Batch size for embedding generation
            index_type (str): Type of FAISS index to create

        Returns:
            Dict[str, Any]: Pipeline execution statistics
        """
        try:
            pipeline_start = datetime.now()
            self.logger.info('[PIPELINE] Starting complete embedding ingestion pipeline')
            self.logger.info('[STEP 1] Chunking data...')
            chunks, chunk_stats = self.splitter.run_chunking_pipeline(data_source=data_source, limit=limit)
            if not chunks:
                raise ValueError('No chunks generated by splitter')
            self.logger.info('[STEP 2] Preparing chunks for embedding...')
            text_chunks = self._prepare_chunks_for_embedding(chunks)
            if not text_chunks:
                raise ValueError('No valid text chunks prepared')
            self.logger.info('[STEP 3] Generating embeddings...')
            embeddings = self._generate_embeddings(text_chunks, batch_size=batch_size)
            self.logger.info('[STEP 4] Creating FAISS index...')
            self.vector_store = self._create_faiss_index(embeddings, index_type=index_type)
            self.logger.info('[STEP 5] Saving index and metadata...')
            self._save_index_and_metadata(self.vector_store)
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()
            pipeline_stats = {'execution_time_seconds': execution_time, 'total_chunks_processed': len(chunks), 'total_embeddings_generated': embeddings.shape[0], 'embedding_dimension': self.dimension, 'index_type': index_type, 'faiss_index_size': int(self.vector_store.ntotal), 'data_source': data_source, 'batch_size': batch_size, 'chunk_statistics': chunk_stats, 'index_directory': str(self.index_dir), 'pipeline_completion_time': pipeline_end.isoformat()}
            self.logger.info('[SUCCESS] Embedding ingestion pipeline completed successfully')
            self.logger.info(f'[SUMMARY] Processed {len(chunks)} chunks in {execution_time:.2f} seconds')
            self.logger.info(f'[SUMMARY] Generated {embeddings.shape[0]} embeddings of dimension {self.dimension}')
            self.logger.info(f'[SUMMARY] FAISS index saved to {self.index_dir}')
            return pipeline_stats
        except Exception as e:
            self.logger.error(f'[ERROR] Embedding pipeline execution failed: {str(e)}')
            raise EmbeddingIngestorError(f'Pipeline failed: {str(e)}') from e

    def load_existing_index(self) -> bool:
        """
        Load existing FAISS index and metadata from disk.

        Returns:
            bool: True if index loaded successfully, False otherwise
        """
        try:
            index_file = self.index_dir / 'faiss.index'
            metadata_file = self.index_dir / 'metadata.pkl'
            if not index_file.exists() or not metadata_file.exists():
                self.logger.info('[LOAD] No existing index found')
                return False
            self.vector_store = faiss.read_index(str(index_file))
            with open(metadata_file, 'rb') as f:
                self.document_metadata = pickle.load(f)
            self.logger.info(f'[LOAD] Successfully loaded existing index with {self.vector_store.ntotal} vectors')
            return True
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to load existing index: {str(e)}')
            return False

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.

        Returns:
            Dict[str, Any]: Index information
        """
        try:
            info_file = self.index_dir / 'index_info.json'
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    return
            return {'total_vectors': int(self.vector_store.ntotal) if self.vector_store else 0, 'vector_dimension': self.dimension, 'total_metadata_entries': len(self.document_metadata), 'index_directory': str(self.index_dir)}
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to get index info: {str(e)}')
            return {'error': str(e)}

def main():
    """
    Main function for command-line usage of EmbeddingIngestor.
    """
    import argparse
    parser = argparse.ArgumentParser(description='FloatChat Embedding Ingestor - Generate and store embeddings for ARGO data', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python src/rag/ingest_embeddings.py                           # Use defaults\n  python src/rag/ingest_embeddings.py --source postgres         # Use PostgreSQL source\n  python src/rag/ingest_embeddings.py --limit 1000              # Process limited data\n  python src/rag/ingest_embeddings.py --batch-size 64           # Custom batch size\n  python src/rag/ingest_embeddings.py --index-type IndexHNSWFlat # HNSW index\n  python src/rag/ingest_embeddings.py --load-existing           # Load existing index info\n        ')
    parser.add_argument('--source', '-s', choices=['parquet', 'postgres'], default='parquet', help='Data source to use (default: parquet)')
    parser.add_argument('--configs', '-c', type=str, default='configs/intel.yaml', help='Path to configuration file (default: configs/intel.yaml)')
    parser.add_argument('--limit', '-l', type=int, default=None, help='Limit number of rows to process (useful for testing)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size for embedding generation (default: 32)')
    parser.add_argument('--index-type', choices=['IndexFlatIP', 'IndexFlatL2', 'IndexHNSWFlat'], default='IndexFlatIP', help='FAISS index type (default: IndexFlatIP)')
    parser.add_argument('--load-existing', action='store_true', help='Load and display info about existing index')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    print('======================================================================')
    print('FloatChat Embedding Ingestor')
    print('RAG Vector Database Generation Pipeline')
    print('======================================================================')
    if args.load_existing:
        print('Loading existing index information...')
    else:
        print(f'Data Source: {args.source}')
        print(f'Config File: {args.config}')
        print(f'Batch Size: {args.batch_size}')
        print(f'Index Type: {args.index_type}')
        if args.limit:
            print(f'Row Limit: {args.limit}')
    print('----------------------------------------------------------------------')
    try:
        ingestor = EmbeddingIngestor(config_path=args.config)
        if args.load_existing:
            success = ingestor.load_existing_index()
            if success:
                info = ingestor.get_index_info()
                print('\n======================================================================')
                print('EXISTING INDEX INFORMATION')
                print('======================================================================')
                print(f"Total vectors: {info.get('total_vectors', 'Unknown')}")
                print(f"Vector dimension: {info.get('vector_dimension', 'Unknown')}")
                print(f"Index type: {info.get('index_type', 'Unknown')}")
                print(f"Embedding model: {info.get('embedding_model', 'Unknown')}")
                print(f"Creation time: {info.get('creation_timestamp', 'Unknown')}")
                print(f"Index directory: {info.get('index_directory', ingestor.index_dir)}")
                if 'index_size_bytes' in info:
                    size_mb = info['index_size_bytes'] / 1048576
                    print(f'Index file size: {size_mb:.2f} MB')
            else:
                print('[INFO] No existing index found or failed to load')
        else:
            print('[INFO] Starting embedding generation pipeline...')
            pipeline_stats = ingestor.run_embedding_pipeline(data_source=args.source, limit=args.limit, batch_size=args.batch_size, index_type=args.index_type)
            print('\n======================================================================')
            print('EMBEDDING PIPELINE RESULTS')
            print('======================================================================')
            print(f"Execution time: {pipeline_stats['execution_time_seconds']:.2f} seconds")
            print(f"Total chunks processed: {pipeline_stats['total_chunks_processed']}")
            print(f"Total embeddings generated: {pipeline_stats['total_embeddings_generated']}")
            print(f"Embedding dimension: {pipeline_stats['embedding_dimension']}")
            print(f"FAISS index size: {pipeline_stats['faiss_index_size']} vectors")
            print(f"Index type: {pipeline_stats['index_type']}")
            print(f"Data source: {pipeline_stats['data_source']}")
            print(f"Index directory: {pipeline_stats['index_directory']}")
            chunk_stats = pipeline_stats.get('chunk_statistics', {})
            if chunk_stats:
                print('\nChunk Statistics:')
                print(f"  - Total chunks: {chunk_stats.get('total_chunks', 'Unknown')}")
                if 'chunk_size_stats' in chunk_stats:
                    size_stats = chunk_stats['chunk_size_stats']
                    print(f"  - Average chunk size: {size_stats.get('mean', 0):.1f} characters")
                    print(f"  - Max chunk size: {size_stats.get('max', 0)} characters")
                if 'measurements_per_chunk_stats' in chunk_stats:
                    measure_stats = chunk_stats['measurements_per_chunk_stats']
                    print(f"  - Average measurements per chunk: {measure_stats.get('mean', 0):.1f}")
                    print(f"  - Total measurements: {measure_stats.get('total_measurements', 0)}")
            chunks_per_second = pipeline_stats['total_chunks_processed'] / pipeline_stats['execution_time_seconds']
            embeddings_per_second = pipeline_stats['total_embeddings_generated'] / pipeline_stats['execution_time_seconds']
            print('\nPerformance Metrics:')
            print(f'  - Chunks processed per second: {chunks_per_second:.2f}')
            print(f'  - Embeddings generated per second: {embeddings_per_second:.2f}')
            if args.verbose:
                print('\nDetailed Configuration:')
                print(f"  - Batch size: {pipeline_stats['batch_size']}")
                print(f"  - Pipeline completion: {pipeline_stats['pipeline_completion_time']}")
        print('\n======================================================================')
        print('[SUCCESS] Embedding ingestor completed successfully!')
        print('======================================================================')
    except KeyboardInterrupt:
        print('\n[INTERRUPT] Operation cancelled by user')
        sys.exit(130)
    except EmbeddingIngestorError as e:
        print(f'\n[ERROR] Embedding ingestor error: {str(e)}')
        sys.exit(1)
    except Exception as e:
        print(f'\n[ERROR] Unexpected error: {str(e)}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()