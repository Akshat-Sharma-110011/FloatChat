# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: D:\MLOps-Projects\FloatChat\src\rag\retriever.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2025-09-13 22:56:35 UTC (1757804195)

"""
FloatChat Retriever

Provides similarity search over the pre-computed FAISS vector store for ARGO oceanographic data.
This module implements the retrieval component of the RAG (Retrieval-Augmented Generation) pipeline,
enabling semantic search over embedded ARGO float profiles and measurements.

The retriever supports various search strategies, filtering capabilities, and result ranking
to provide the most relevant oceanographic data chunks for LLM context augmentation.

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
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
import numpy as np
import faiss
import yaml
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from collections import defaultdict

class RetrieverError(Exception):
    """Custom exception for Retriever operations."""

@dataclass
class DocumentChunk:
    """
    Represents a retrieved document chunk with metadata and relevance scoring.

    Attributes:
        chunk_id (str): Unique identifier for the chunk
        text (str): Text content of the chunk
        metadata (Dict[str, Any]): Associated metadata (timestamp, location, etc.)
        score (float): Similarity/relevance score
        measurements (List[Dict]): Associated oceanographic measurements
        source (str): Data source identifier
        chunk_type (str): Type of chunk (temporal_profile, etc.)
        retrieval_context (Dict[str, Any]): Context about how this chunk was retrieved
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    measurements: List[Dict[str, Any]]
    source: str = 'argo_floats'
    chunk_type: str = 'temporal_profile'
    retrieval_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentChunk to dictionary representation."""
        return asdict(self)

    def get_location(self) -> Tuple[float, float]:
        """Get latitude and longitude from metadata."""
        lat = self.metadata.get('latitude', 0.0)
        lon = self.metadata.get('longitude', 0.0)
        return (lat, lon)

    def get_timestamp(self) -> str:
        """Get timestamp from metadata."""
        return self.metadata.get('timestamp', '')

    def get_measurement_count(self) -> int:
        """Get number of measurements in this chunk."""
        return len(self.measurements)

class Retriever:
    """
    Provides similarity search over the pre-computed FAISS vector store.

    This class handles all retrieval operations including:
    - Loading pre-computed FAISS indices and metadata
    - Embedding query text using the same model used for indexing
    - Performing similarity search with various strategies
    - Post-processing and ranking results
    - Filtering results by metadata criteria

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        index_dir (Path): Directory containing FAISS index
        embedding_model (SentenceTransformer): Model for query embedding
        vector_store (faiss.Index): FAISS index for similarity search
        document_metadata (List[Dict]): Metadata for all indexed documents
        dimension (int): Embedding vector dimension
        is_loaded (bool): Whether index and metadata are loaded
    """

    def __init__(self, config_path: str='configs/intel.yaml'):
        """
        Initialize the Retriever with configuration and load the vector store.

        Args:
            config_path (str): Path to the configuration YAML file

        Raises:
            RetrieverError: If initialization or index loading fails
        """
        try:
            self._setup_logging()
            self.logger.info('[INIT] Initializing Retriever')
            self.config = self._load_config(config_path)
            self.logger.info(f'[CONFIG] Configuration loaded from {config_path}')
            self.index_dir = Path(self.config.get('vector_store', {}).get('path', 'data/index/faiss_index'))
            self.embedding_model = None
            self.vector_store = None
            self.document_metadata = []
            self.dimension = 0
            self.is_loaded = False
            model_name = self.config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
            self._load_embedding_model(model_name)
            self.load_index()
            self.logger.info('[INIT] Retriever initialization completed successfully')
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f'[ERROR] Failed to initialize Retriever: {str(e)}')
            raise RetrieverError(f'Initialization failed: {str(e)}') from e

    def _setup_logging(self) -> None:
        """
        Setup logging configuration with timestamped log files.
        """
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        log_filename = logs_dir / f'retriever_{timestamp}.log'
        self.logger = logging.getLogger(f'Retriever_{timestamp}')
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
        self.logger.info(f'[LOGGING] Retriever logging initialized - Log file: {log_filename}')

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
            raise RetrieverError(f'Failed to load configuration: {str(e)}') from e

    def _load_embedding_model(self, model_name: str) -> None:
        """
        Load the sentence transformer model for query embedding.

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
            raise RetrieverError(f'Model loading failed: {str(e)}') from e

    def load_index(self) -> bool:
        """
        Load FAISS index and document metadata from disk.

        Returns:
            bool: True if index loaded successfully, False otherwise

        Raises:
            RetrieverError: If index loading fails critically
        """
        try:
            self.logger.info(f'[LOAD] Loading index from {self.index_dir}')
            index_file = self.index_dir / 'faiss.index'
            metadata_file = self.index_dir / 'metadata.pkl'
            info_file = self.index_dir / 'index_info.json'
            if not index_file.exists():
                raise FileNotFoundError(f'FAISS index file not found: {index_file}')
            if not metadata_file.exists():
                raise FileNotFoundError(f'Metadata file not found: {metadata_file}')
            self.logger.info('[LOAD] Loading FAISS index...')
            self.vector_store = faiss.read_index(str(index_file))
            self.logger.info('[LOAD] Loading document metadata...')
            with open(metadata_file, 'rb') as f:
                self.document_metadata = pickle.load(f)
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    index_info = json.load(f)
                expected_vectors = index_info.get('total_vectors', 0)
                actual_vectors = int(self.vector_store.ntotal)
                expected_metadata = len(self.document_metadata)
                if actual_vectors != expected_vectors:
                    self.logger.warning(f'[WARNING] Vector count mismatch: {actual_vectors} vs {expected_vectors}')
                if actual_vectors != expected_metadata:
                    self.logger.warning(f'[WARNING] Metadata count mismatch: {expected_metadata} vs {actual_vectors}')
                self.logger.info(f"[LOAD] Index info: {expected_vectors} vectors, dimension {index_info.get('vector_dimension')}")
            self.is_loaded = True
            self.logger.info(f'[SUCCESS] Index loaded successfully - {self.vector_store.ntotal} vectors, {len(self.document_metadata)} metadata entries')
            return True
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to load index: {str(e)}')
            self.is_loaded = False
            raise RetrieverError(f'Index loading failed: {str(e)}') from e

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string using the loaded embedding model.

        Args:
            query (str): Query text to embed

        Returns:
            np.ndarray: Query embedding vector
        """
        try:
            if not self.embedding_model:
                raise ValueError('Embedding model not loaded')
            embedding = self.embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return embedding.astype(np.float32)
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to embed query: {str(e)}')
            raise RetrieverError(f'Query embedding failed: {str(e)}') from e

    def _validate_retrieval_params(self, k: int, score_threshold: Optional[float]=None) -> None:
        """
        Validate retrieval parameters.

        Args:
            k (int): Number of results to retrieve
            score_threshold (Optional[float]): Minimum similarity score threshold
        """
        if not self.is_loaded:
            raise RetrieverError('Index not loaded. Call load_index() first.')
        if k <= 0:
            raise ValueError('k must be positive')
        if k > self.vector_store.ntotal:
            self.logger.warning(f'[WARNING] k={k} is larger than index size {self.vector_store.ntotal}')
        if score_threshold is not None and (score_threshold < 0 or score_threshold > 1):
            raise ValueError('score_threshold must be between 0 and 1')
    pass
    pass
    pass
    pass

    def retrieve_topk(self, query: str, k: int=5, score_threshold: Optional[float]=None, metadata_filters: Optional[Dict[str, Any]]=None, search_strategy: str='similarity') -> List[DocumentChunk]:
        """
        Retrieve top-k most similar document chunks for a query.

        Args:
            query (str): Query text
            k (int): Number of results to retrieve (default: 5)
            score_threshold (Optional[float]): Minimum similarity score threshold
            metadata_filters (Optional[Dict[str, Any]]): Filters to apply based on metadata
            search_strategy (str): Search strategy ("similarity", "mmr", "hybrid")

        Returns:
            List[DocumentChunk]: List of retrieved document chunks with relevance scores

        Raises:
            RetrieverError: If retrieval fails
        """
        try:
            self.logger.info(f'[RETRIEVE] Starting retrieval - Query length: {len(query)}, k={k}')
            self._validate_retrieval_params(k, score_threshold)
            query_embedding = self._embed_query(query)
            search_k = min(k * 2, self.vector_store.ntotal)
            scores, indices = self.vector_store.search(query_embedding.reshape(1, -1), search_k)
            scores = scores[0]
            indices = indices[0]
            chunks = []
            for i, (score, idx) in enumerate(zip(scores, indices)):
                if idx == -1:
                    continue
                try:
                    if idx >= len(self.document_metadata):
                        self.logger.warning(f'[WARNING] Index {idx} out of metadata range')
                        continue
                    metadata_entry = self.document_metadata[idx]
                    original_chunk = metadata_entry.get('original_chunk', {})
                    if metadata_filters and (not self._passes_metadata_filters(metadata_entry, metadata_filters)):
                        continue
                    if score_threshold is not None and score < score_threshold:
                        continue
                    chunk = DocumentChunk(chunk_id=metadata_entry.get('chunk_id', f'chunk_{idx}'), text=self._reconstruct_chunk_text(original_chunk), metadata=metadata_entry.get('metadata', {}), score=float(score), measurements=original_chunk.get('measurements', []), source=metadata_entry.get('source', 'argo_floats'), chunk_type=query, retrieval_context={'query': i + 1, 'rank': search_strategy, 'search_strategy': search_strategy, 'retrieval_timestamp': datetime.now().isoformat()})
                    chunks.append(chunk)
                except Exception as e:
                    self.logger.warning(f'[WARNING] Failed to process result {idx}: {str(e)}')
            if search_strategy == 'mmr':
                chunks = self._apply_mmr_ranking(chunks, query_embedding, k)
            elif search_strategy == 'hybrid':
                chunks = self._apply_hybrid_ranking(chunks, query, k)
            chunks = chunks[:k]
            self.logger.info(f'[SUCCESS] Retrieved {len(chunks)} chunks')
            if chunks:
                avg_score = sum((chunk.score for chunk in chunks)) / len(chunks)
                score_range = (min((chunk.score for chunk in chunks)), max((chunk.score for chunk in chunks)))
                self.logger.info(f'[STATS] Average score: {avg_score:.4f}, Score range: {score_range[0]:.4f}-{score_range[1]:.4f}')
            return chunks
        except Exception as e:
            self.logger.error(f'[ERROR] Retrieval failed: {str(e)}')
            raise RetrieverError(f'Retrieval failed: {str(e)}') from e

    def _reconstruct_chunk_text(self, original_chunk: Dict[str, Any]) -> str:
        """
        Reconstruct text representation of a chunk from its original data.

        Args:
            original_chunk (Dict[str, Any]): Original chunk data

        Returns:
            str: Reconstructed text representation
        """
        try:
            if not original_chunk:
                return ''
            metadata = original_chunk.get('metadata', {})
            measurements = original_chunk.get('measurements', [])
            text_parts = [f"ARGO Float Profile - Timestamp: {metadata.get('timestamp', 'Unknown')}", f"Location: {metadata.get('latitude', 0.0):.6f}°N, {metadata.get('longitude', 0.0):.6f}°E", f"Basin: {metadata.get('basin', 'Unknown')}, Cycle: {metadata.get('cycle_number', 'Unknown')}", f"Sampling Scheme: {metadata.get('vertical_sampling_scheme', 'Unknown')}", f'Profile Depth Points: {len(measurements)}', '']
            if measurements:
                text_parts.append('Depth Profile Measurements:')
                for i, measurement in enumerate(measurements[:10]):
                    pressure = measurement.get('pressure_decibar', 0.0)
                    temp = measurement.get('temperature_degc', 0.0)
                    salinity = measurement.get('salinity_psu', 0.0)
                    text_parts.append(f'  {i + 1:3d}. Pressure: {pressure:6.1f} dbar, Temperature: {temp:6.3f}°C, Salinity: {salinity:6.3f} PSU')
                if len(measurements) > 10:
                    text_parts.append(f'  ... and {len(measurements) - 10} more measurements')
            return '\n'.join(text_parts)
        except Exception as e:
            self.logger.warning(f'[WARNING] Failed to reconstruct chunk text: {str(e)}')
            return json.dumps(original_chunk, default=str)

    def _passes_metadata_filters(self, metadata_entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if a metadata entry passes the specified filters.

        Args:
            metadata_entry (Dict[str, Any]): Metadata entry to check
            filters (Dict[str, Any]): Filter criteria

        Returns:
            bool: True if entry passes all filters
        """
        try:
            metadata = metadata_entry.get('metadata', {})
            for key, expected_value in filters.items():
                if key not in metadata:
                    return False
                actual_value = metadata[key]
                if isinstance(expected_value, dict):
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        return False
                    if 'max' in expected_value and actual_value > expected_value['max']:
                        return False
                elif isinstance(expected_value, (list, tuple)):
                    if actual_value not in expected_value:
                        return False
                elif actual_value != expected_value:
                    return False
            else:
                return True
        except Exception as e:
            self.logger.warning(f'[WARNING] Filter evaluation failed: {str(e)}')
            return False
    pass

    def _apply_mmr_ranking(self, chunks: List[DocumentChunk], query_embedding: np.ndarray, k: int, lambda_param: float=0.7) -> List[DocumentChunk]:
        """
        Apply Maximal Marginal Relevance (MMR) ranking to diversify results.

        Args:
            chunks (List[DocumentChunk]): Initial chunks sorted by similarity
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of results to select
            lambda_param (float): Balance between relevance and diversity (0-1)

        Returns:
            List[DocumentChunk]: Re-ranked chunks using MMR
        """
        if len(chunks) <= k:
            return chunks
        try:
            self.logger.info(f'[MMR] Applying MMR ranking with lambda={lambda_param}')
            selected = [chunks[0]]
            remaining = chunks[1:]
            while len(selected) < k and remaining:
                best_idx = 0
                best_score = -float('inf')
                for i, chunk in enumerate(remaining):
                    relevance = chunk.score
                    max_similarity = 0.0
                    for selected_chunk in selected:
                        similarity = self._compute_chunk_similarity(chunk, selected_chunk)
                        max_similarity = max(max_similarity, similarity)
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                selected.append(remaining.pop(best_idx))
            self.logger.info(f'[MMR] Selected {len(selected)} diverse chunks')
            return selected
        except Exception as e:
            self.logger.warning(f'[WARNING] MMR ranking failed, using original order: {str(e)}')
            return chunks[:k]

    def _compute_chunk_similarity(self, chunk1: DocumentChunk, chunk2: DocumentChunk) -> float:
        """
        Compute similarity between two chunks based on metadata and content.

        Args:
            chunk1 (DocumentChunk): First chunk
            chunk2 (DocumentChunk): Second chunk

        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            similarity = 0.0
            if chunk1.get_timestamp() == chunk2.get_timestamp():
                similarity += 0.4
            lat1, lon1 = chunk1.get_location()
            lat2, lon2 = chunk2.get_location()
            distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            spatial_sim = max(0, 1 - distance / 10.0)
            similarity += 0.3 * spatial_sim
            if chunk1.metadata.get('basin') == chunk2.metadata.get('basin'):
                similarity += 0.3
            return min(similarity, 1.0)
        except Exception:
            return 0.0

    def _apply_hybrid_ranking(self, chunks: List[DocumentChunk], query: str, k: int) -> List[DocumentChunk]:
        """
        Apply hybrid ranking combining semantic similarity with metadata relevance.

        Args:
            chunks (List[DocumentChunk]): Initial chunks
            query (str): Original query text
            k (int): Number of results to select

        Returns:
            List[DocumentChunk]: Re-ranked chunks
        """
        try:
            self.logger.info('[HYBRID] Applying hybrid ranking')
            query_lower = query.lower()
            oceanographic_terms = ['temperature', 'salinity', 'pressure', 'depth', 'ocean', 'water', 'profile']
            for chunk in chunks:
                boost = 0.0
                for term in oceanographic_terms:
                    if term in query_lower and term in chunk.text.lower():
                        boost += 0.1
                if 'temperature' in query_lower and chunk.measurements:
                    temps = [m.get('temperature_degc', 0) for m in chunk.measurements]
                    if any((t > 20 for t in temps)):
                        boost += 0.05
                chunk.score = min(1.0, chunk.score + boost)
            chunks.sort(key=lambda x: x.score, reverse=True)
            return chunks[:k]
        except Exception as e:
            self.logger.warning(f'[WARNING] Hybrid ranking failed: {str(e)}')
            return chunks[:k]
    pass

    def retrieve_by_location(self, latitude: float, longitude: float, radius_km: float=100.0, k: int=10) -> List[DocumentChunk]:
        """
        Retrieve chunks based on geographic proximity.

        Args:
            latitude (float): Target latitude
            longitude (float): Target longitude
            radius_km (float): Search radius in kilometers
            k (int): Maximum number of results

        Returns:
            List[DocumentChunk]: Location-based retrieved chunks
        """
        try:
            self.logger.info(f'[LOCATION] Retrieving by location: {latitude:.4f}, {longitude:.4f} (radius: {radius_km}km)')
            if not self.is_loaded:
                raise RetrieverError('Index not loaded')
            location_chunks = []
            for i, metadata_entry in enumerate(self.document_metadata):
                chunk_metadata = metadata_entry.get('metadata', {})
                chunk_lat = chunk_metadata.get('latitude', 0.0)
                chunk_lon = chunk_metadata.get('longitude', 0.0)
                lat_diff = latitude - chunk_lat
                lon_diff = longitude - chunk_lon
                distance_deg = (lat_diff ** 2 + lon_diff ** 2) ** 0.5
                distance_km = distance_deg * 111.0
                if distance_km <= radius_km:
                    original_chunk = metadata_entry.get('original_chunk', {})
                    chunk = DocumentChunk(chunk_id=metadata_entry.get('chunk_id', f'chunk_{i}'), text=self._reconstruct_chunk_text(original_chunk), metadata=chunk_metadata, score=1.0 - distance_km / radius_km, measurements=original_chunk.get('measurements', []), source=metadata_entry.get('source', 'argo_floats'), chunk_type=metadata_entry.get('chunk_type', 'temporal_profile'), retrieval_context={'retrieval_type': 'location_based', 'target_location': (latitude, longitude), 'distance_km': distance_km, 'retrieval_timestamp': datetime.now().isoformat()})
                    location_chunks.append(chunk)
            location_chunks.sort(key=lambda x: x.score, reverse=True)
            result = location_chunks[:k]
            self.logger.info(f'[LOCATION] Found {len(result)} chunks within {radius_km}km')
            return result
        except Exception as e:
            self.logger.error(f'[ERROR] Location-based retrieval failed: {str(e)}')
            raise RetrieverError(f'Location retrieval failed: {str(e)}') from e
    pass

    def retrieve_by_time_range(self, start_time: str, end_time: str, k: int=10) -> List[DocumentChunk]:
        """
        Retrieve chunks within a specific time range.

        Args:
            start_time (str): Start timestamp (ISO format)
            end_time (str): End timestamp (ISO format)
            k (int): Maximum number of results

        Returns:
            List[DocumentChunk]: Time-based retrieved chunks
        """
        try:
            self.logger.info(f'[TIME] Retrieving by time range: {start_time} to {end_time}')
            if not self.is_loaded:
                raise RetrieverError('Index not loaded')
            time_chunks = []
            for i, metadata_entry in enumerate(self.document_metadata):
                chunk_metadata = metadata_entry.get('metadata', {})
                chunk_timestamp = chunk_metadata.get('timestamp', '')
                if start_time <= chunk_timestamp <= end_time:
                    original_chunk = metadata_entry.get('original_chunk', {})
                    chunk = DocumentChunk(chunk_id=metadata_entry.get('chunk_id', f'chunk_{i}'), text=self._reconstruct_chunk_text(original_chunk), metadata=chunk_metadata, score=1.0, measurements=original_chunk.get('measurements', []), source=metadata_entry.get('source', 'argo_floats'), chunk_type=metadata_entry.get('chunk_type', 'temporal_profile'), retrieval_context={'retrieval_type': 'time_based', 'time_range': (start_time, end_time), 'retrieval_timestamp': datetime.now().isoformat()})
                    time_chunks.append(chunk)
            time_chunks.sort(key=lambda x: x.get_timestamp())
            result = time_chunks[:k]
            self.logger.info(f'[TIME] Found {len(result)} chunks in time range')
            return result
        except Exception as e:
            self.logger.error(f'[ERROR] Time-based retrieval failed: {str(e)}')
            raise RetrieverError(f'Time retrieval failed: {str(e)}') from e

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded index.

        Returns:
            Dict[str, Any]: Index statistics and metadata
        """
        try:
            if not self.is_loaded:
                return {'error': 'Index not loaded'}
            stats = {'total_vectors': int(self.vector_store.ntotal), 'vector_dimension': self.dimension, 'total_metadata_entries': len(self.document_metadata), 'index_type': type(self.vector_store).__name__, 'is_trained': getattr(self.vector_store, 'is_trained', True), 'index_directory': str(self.index_dir)}
            if self.document_metadata:
                source_counts = defaultdict(int)
                chunk_type_counts = defaultdict(int)
                basin_counts = defaultdict(int)
                for metadata in self.document_metadata:
                    source_counts[metadata.get('source', 'unknown')] += 1
                    chunk_type_counts[metadata.get('chunk_type', 'unknown')] += 1
                    chunk_metadata = metadata.get('metadata', {})
                    basin_counts[chunk_metadata.get('basin', 'unknown')] += 1
                stats['source_distribution'] = dict(source_counts)
                stats['chunk_type_distribution'] = dict(chunk_type_counts)
                stats['basin_distribution'] = dict(basin_counts)
                timestamps = []
                for metadata in self.document_metadata:
                    chunk_metadata = metadata.get('metadata', {})
                    timestamp = chunk_metadata.get('timestamp')
                    if timestamp:
                        timestamps.append(timestamp)
                if timestamps:
                    timestamps.sort()
                    stats['time_range'] = {'earliest': timestamps[0], 'latest': timestamps[-1], 'total_timestamps': len(set(timestamps))}
                total_measurements = sum((metadata.get('measurement_count', 0) for metadata in self.document_metadata))
                stats['total_measurements'] = total_measurements
            return stats
        except Exception as e:
            self.logger.error(f'[ERROR] Failed to get index statistics: {str(e)}')
            return {'error': str(e)}
    pass

    def search_similar_profiles(self, reference_measurements: List[Dict[str, float]], k: int=5) -> List[DocumentChunk]:
        """
        Find profiles similar to a reference set of measurements.

        Args:
            reference_measurements (List[Dict[str, float]]): Reference measurements
            k (int): Number of similar profiles to return

        Returns:
            List[DocumentChunk]: Similar oceanographic profiles
        """
        try:
            self.logger.info(f'[PROFILE] Searching for profiles similar to {len(reference_measurements)} measurements')
            if not self.is_loaded:
                raise RetrieverError('Index not loaded')
            query_parts = []
            for i, measurement in enumerate(reference_measurements[:5]):
                pressure = measurement.get('pressure_decibar', 0)
                temp = measurement.get('temperature_degc', 0)
                salinity = measurement.get('salinity_psu', 0)
                query_parts.append(f'Depth {pressure:.1f}m: temperature {temp:.2f}°C, salinity {salinity:.2f}')
            synthetic_query = 'Oceanographic profile with measurements: ' + '; '.join(query_parts)
            results = self.retrieve_topk(synthetic_query, k=k, search_strategy='similarity')
            for result in results:
                result.retrieval_context['retrieval_type'] = 'profile_similarity'
                result.retrieval_context['reference_measurement_count'] = len(reference_measurements)
            self.logger.info(f'[PROFILE] Found {len(results)} similar profiles')
            return results
        except Exception as e:
            self.logger.error(f'[ERROR] Profile similarity search failed: {str(e)}')
            raise RetrieverError(f'Profile search failed: {str(e)}') from e
    pass
    pass

    def batch_retrieve(self, queries: List[str], k: int=5, batch_size: int=16) -> List[List[DocumentChunk]]:
        """
        Perform batch retrieval for multiple queries efficiently.

        Args:
            queries (List[str]): List of query strings
            k (int): Number of results per query
            batch_size (int): Processing batch size

        Returns:
            List[List[DocumentChunk]]: Results for each query
        """
        try:
            self.logger.info(f'[BATCH] Processing {len(queries)} queries in batches of {batch_size}')
            if not self.is_loaded:
                raise RetrieverError('Index not loaded')
            all_results = []
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_results = []
                for query in batch_queries:
                    try:
                        results = self.retrieve_topk(query, k=k)
                        batch_results.append(results)
                    except Exception as e:
                        self.logger.warning(f'[WARNING] Batch query failed: {str(e)}')
                        batch_results.append([])
                all_results.extend(batch_results)
                if i + batch_size < len(queries):
                    self.logger.info(f'[BATCH] Processed {i + batch_size}/{len(queries)} queries')
            self.logger.info(f'[BATCH] Completed batch retrieval for {len(queries)} queries')
            return all_results
        except Exception as e:
            self.logger.error(f'[ERROR] Batch retrieval failed: {str(e)}')
            raise RetrieverError(f'Batch retrieval failed: {str(e)}') from e
    pass

    def export_results(self, chunks: List[DocumentChunk], output_path: str, format: str='json') -> None:
        """
        Export retrieval results to file.

        Args:
            chunks (List[DocumentChunk]): Retrieved chunks to export
            output_path (str): Output file path
            format (str): Export format ("json", "csv")
        """
        try:
            self.logger.info(f'[EXPORT] Exporting {len(chunks)} chunks to {output_path}')
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if format.lower() == 'json':
                export_data = {'export_timestamp': datetime.now().isoformat(), 'total_chunks': len(chunks), 'chunks': [chunk.to_dict() for chunk in chunks]}
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            elif format.lower() == 'csv':
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if not chunks:
                        return
                    fieldnames = ['chunk_id', 'score', 'timestamp', 'latitude', 'longitude', 'basin', 'measurement_count', 'source', 'chunk_type']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for chunk in chunks:
                        row = {'chunk_id': chunk.chunk_id, 'score': chunk.score, 'timestamp': chunk.get_timestamp(), 'latitude': chunk.get_location()[0], 'longitude': chunk.get_location()[1], 'basin': chunk.metadata.get('basin', ''), 'measurement_count': chunk.get_measurement_count(), 'source': chunk.source, 'chunk_type': chunk.chunk_type}
                        writer.writerow(row)
            else:
                raise ValueError(f'Unsupported export format: {format}')
            self.logger.info(f'[SUCCESS] Results exported to {output_path}')
        except Exception as e:
            self.logger.error(f'[ERROR] Export failed: {str(e)}')
            raise RetrieverError(f'Export failed: {str(e)}') from e

def main():
    """
    Main function for command-line usage of Retriever.
    """
    import argparse
    parser = argparse.ArgumentParser(description='FloatChat Retriever - Search ARGO oceanographic data using vector similarity', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python src/rag/retriever.py --query "temperature profile Atlantic Ocean"\n  python src/rag/retriever.py --query "salinity measurements" --k 10\n  python src/rag/retriever.py --location 40.7 -74.0 --radius 50\n  python src/rag/retriever.py --time-range "2024-01-01" "2024-12-31"\n  python src/rag/retriever.py --stats                    # Show index statistics\n  python src/rag/retriever.py --query "ocean data" --export results.json\n        ')
    parser.add_argument('--query', '-q', type=str, help='Query text for similarity search')
    parser.add_argument('--k', type=int, default=5, help='Number of results to retrieve (default: 5)')
    parser.add_argument('--score-threshold', type=float, help='Minimum similarity score threshold (0-1)')
    parser.add_argument('--location', nargs=2, type=float, metavar=('LAT', 'LON'), help='Location-based search: latitude longitude')
    parser.add_argument('--radius', type=float, default=100.0, help='Search radius in kilometers (default: 100)')
    parser.add_argument('--time-range', nargs=2, type=str, metavar=('START', 'END'), help='Time range search: start_time end_time (ISO format)')
    parser.add_argument('--configs', '-c', type=str, default='configs/intel.yaml', help='Path to configuration file (default: configs/intel.yaml)')
    parser.add_argument('--strategy', choices=['similarity', 'mmr', 'hybrid'], default='similarity', help='Search strategy (default: similarity)')
    parser.add_argument('--export', type=str, help='Export results to file (JSON or CSV format)')
    parser.add_argument('--stats', action='store_true', help='Show index statistics and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    print('======================================================================')
    print('FloatChat Retriever')
    print('Semantic Search for ARGO Oceanographic Data')
    print('======================================================================')
    try:
        retriever = Retriever(config_path=args.config)
        if args.stats:
            stats = retriever.get_index_statistics()
            print('INDEX STATISTICS')
            print('----------------------------------------------------------------------')
            print(f"Total vectors: {stats.get('total_vectors', 'Unknown')}")
            print(f"Vector dimension: {stats.get('vector_dimension', 'Unknown')}")
            print(f"Index type: {stats.get('index_type', 'Unknown')}")
            print(f"Total metadata entries: {stats.get('total_metadata_entries', 'Unknown')}")
            print(f"Total measurements: {stats.get('total_measurements', 'Unknown')}")
            if 'time_range' in stats:
                time_range = stats['time_range']
                print('\nTime Range:')
                print(f"  - Earliest: {time_range.get('earliest', 'Unknown')}")
                print(f"  - Latest: {time_range.get('latest', 'Unknown')}")
                print(f"  - Unique timestamps: {time_range.get('total_timestamps', 'Unknown')}")
            if 'source_distribution' in stats:
                print('\nData Sources:')
                for source, count in stats['source_distribution'].items():
                    print(f'  - {source}: {count}')
            if 'basin_distribution' in stats:
                print('\nBasin Distribution:')
                for basin, count in stats['basin_distribution'].items():
                    print(f'  - Basin {basin}: {count}')
            return None
        else:
            results = []
            if args.query:
                print(f"Searching for: '{args.query}'")
                print(f'Strategy: {args.strategy}, k={args.k}')
                if args.score_threshold:
                    print(f'Score threshold: {args.score_threshold}')
                print('----------------------------------------------------------------------')
                results = retriever.retrieve_topk(query=args.query, k=args.k, score_threshold=args.score_threshold, search_strategy=args.strategy)
            elif args.location:
                lat, lon = args.location
                print(f'Location search: {lat:.4f}°, {lon:.4f}° (radius: {args.radius}km)')
                print('----------------------------------------------------------------------')
                results = retriever.retrieve_by_location(latitude=lat, longitude=lon, radius_km=args.radius, k=args.k)
            elif args.time_range:
                start_time, end_time = args.time_range
                print(f'Time range search: {start_time} to {end_time}')
                print('----------------------------------------------------------------------')
                results = retriever.retrieve_by_time_range(start_time=start_time, end_time=end_time, k=args.k)
            else:
                print('No search criteria specified. Use --query, --location, --time-range, or --stats')
                return
            if results:
                print(f'\nFOUND {len(results)} RESULTS')
                print('======================================================================')
                for i, chunk in enumerate(results, 1):
                    print(f'\nResult {i}: {chunk.chunk_id}')
                    print(f'Score: {chunk.score:.4f}')
                    print(f'Timestamp: {chunk.get_timestamp()}')
                    print(f'Location: {chunk.get_location()[0]:.4f}°N, {chunk.get_location()[1]:.4f}°E')
                    print(f"Basin: {chunk.metadata.get('basin', 'Unknown')}")
                    print(f'Measurements: {chunk.get_measurement_count()}')
                    if args.verbose:
                        print(f'Text preview: {chunk.text[:200]}...')
                    print('----------------------------------------')
                if args.export:
                    format_type = 'json' if args.export.endswith('.json') else 'csv'
                    retriever.export_results(results, args.export, format_type)
                    print(f'\nResults exported to: {args.export}')
            else:
                print('No results found matching your criteria.')
            print('\n======================================================================')
            print('[SUCCESS] Retrieval completed successfully!')
            print('======================================================================')
    except KeyboardInterrupt:
        print('\n[INTERRUPT] Operation cancelled by user')
        sys.exit(130)
    except RetrieverError as e:
        print(f'\n[ERROR] Retriever error: {str(e)}')
        sys.exit(1)
    except Exception as e:
        print(f'\n[ERROR] Unexpected error: {str(e)}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()