"""
FloatChat Retrieve Documents Tool

This tool provides semantic search capabilities over ARGO oceanographic data.
It uses the FloatChat retriever to find relevant oceanographic profiles and measurements
based on user queries, returning contextually relevant information for LLM processing.

Author: FloatChat Team
Created: 2025-01-XX
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.rag.retriever import Retriever, RetrieverError, DocumentChunk


class RetrieveDocsError(Exception):
    """Custom exception for retrieve_docs tool operations."""
    pass


def retrieve_docs_tool(query: str, k: int = 5,
                       score_threshold: Optional[float] = None,
                       metadata_filters: Optional[Dict[str, Any]] = None,
                       search_strategy: str = "similarity",
                       retriever: Optional[Retriever] = None,
                       config_path: str = "configs/intel.yaml") -> Dict[str, Any]:
    """
    Tool: Return top-k relevant document chunks for a query using semantic similarity.

    This tool searches through the ARGO oceanographic database using vector similarity
    to find the most relevant profiles and measurements for the given query.

    Args:
        query (str): Query text for semantic search
        k (int): Number of results to retrieve (default: 5)
        score_threshold (Optional[float]): Minimum similarity score threshold
        metadata_filters (Optional[Dict[str, Any]]): Filters based on metadata
        search_strategy (str): Search strategy ("similarity", "mmr", "hybrid")
        retriever (Optional[Retriever]): Pre-initialized retriever instance
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Dictionary containing retrieved chunks and metadata

    Raises:
        RetrieveDocsError: If retrieval operation fails
    """
    # Setup logging
    logger = logging.getLogger("RetrieveDocsTool")

    try:
        logger.info(f"[RETRIEVE] Starting document retrieval for query: '{query}' (k={k})")

        # Initialize retriever if not provided
        if retriever is None:
            logger.info("[RETRIEVE] Initializing new retriever instance")
            retriever = Retriever(config_path=config_path)

        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        if k > 50:  # Reasonable limit
            logger.warning(f"[WARNING] Large k value ({k}), limiting to 50")
            k = 50

        # Perform retrieval
        logger.info(f"[RETRIEVE] Executing search with strategy: {search_strategy}")
        chunks = retriever.retrieve_topk(
            query=query,
            k=k,
            score_threshold=score_threshold,
            metadata_filters=metadata_filters,
            search_strategy=search_strategy
        )

        # Process and format results
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": round(chunk.score, 4),
                "metadata": {
                    "timestamp": chunk.get_timestamp(),
                    "location": {
                        "latitude": chunk.get_location()[0],
                        "longitude": chunk.get_location()[1]
                    },
                    "basin": chunk.metadata.get("basin", "Unknown"),
                    "cycle_number": chunk.metadata.get("cycle_number", "Unknown"),
                    "vertical_sampling_scheme": chunk.metadata.get("vertical_sampling_scheme", "Unknown"),
                    "measurement_count": chunk.get_measurement_count()
                },
                "measurements_summary": _summarize_measurements(chunk.measurements),
                "source": chunk.source,
                "chunk_type": chunk.chunk_type,
                "retrieval_context": chunk.retrieval_context
            }
            processed_chunks.append(processed_chunk)

        # Calculate retrieval statistics
        stats = _calculate_retrieval_stats(chunks, query)

        # Prepare response
        response = {
            "query": query,
            "chunks": processed_chunks,
            "total_found": len(processed_chunks),
            "search_parameters": {
                "k": k,
                "score_threshold": score_threshold,
                "search_strategy": search_strategy,
                "metadata_filters": metadata_filters
            },
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        logger.info(f"[SUCCESS] Retrieved {len(processed_chunks)} documents")

        # Log summary statistics
        if chunks:
            avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
            score_range = (min(chunk.score for chunk in chunks), max(chunk.score for chunk in chunks))
            logger.info(f"[STATS] Average score: {avg_score:.4f}, Range: {score_range[0]:.4f}-{score_range[1]:.4f}")

        return response

    except RetrieverError as e:
        logger.error(f"[ERROR] Retriever error: {str(e)}")
        return _error_response(query, f"Retrieval failed: {str(e)}", "retriever_error")

    except ValueError as e:
        logger.error(f"[ERROR] Validation error: {str(e)}")
        return _error_response(query, f"Invalid input: {str(e)}", "validation_error")

    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        return _error_response(query, f"Unexpected error: {str(e)}", "unexpected_error")


def _summarize_measurements(measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of oceanographic measurements for a chunk.

    Args:
        measurements (List[Dict[str, Any]]): List of measurement dictionaries

    Returns:
        Dict[str, Any]: Summary statistics of measurements
    """
    if not measurements:
        return {
            "count": 0,
            "depth_range": {"min": 0, "max": 0},
            "temperature_range": {"min": 0, "max": 0},
            "salinity_range": {"min": 0, "max": 0}
        }

    try:
        # Extract values
        pressures = [m.get('pressure_decibar', 0) for m in measurements]
        temperatures = [m.get('temperature_degc', 0) for m in measurements]
        salinities = [m.get('salinity_psu', 0) for m in measurements]

        # Calculate statistics
        summary = {
            "count": len(measurements),
            "depth_range": {
                "min": round(min(pressures), 1) if pressures else 0,
                "max": round(max(pressures), 1) if pressures else 0,
                "mean": round(sum(pressures) / len(pressures), 1) if pressures else 0
            },
            "temperature_range": {
                "min": round(min(temperatures), 3) if temperatures else 0,
                "max": round(max(temperatures), 3) if temperatures else 0,
                "mean": round(sum(temperatures) / len(temperatures), 3) if temperatures else 0
            },
            "salinity_range": {
                "min": round(min(salinities), 3) if salinities else 0,
                "max": round(max(salinities), 3) if salinities else 0,
                "mean": round(sum(salinities) / len(salinities), 3) if salinities else 0
            }
        }

        return summary

    except Exception as e:
        return {
            "count": len(measurements),
            "error": f"Failed to calculate summary: {str(e)}"
        }


def _calculate_retrieval_stats(chunks: List[DocumentChunk], query: str) -> Dict[str, Any]:
    """
    Calculate statistics about the retrieved chunks.

    Args:
        chunks (List[DocumentChunk]): Retrieved document chunks
        query (str): Original query

    Returns:
        Dict[str, Any]: Retrieval statistics
    """
    try:
        if not chunks:
            return {
                "total_chunks": 0,
                "query_length": len(query),
                "average_score": 0.0,
                "score_distribution": {},
                "geographic_distribution": {},
                "temporal_distribution": {},
                "basin_distribution": {}
            }

        # Basic statistics
        scores = [chunk.score for chunk in chunks]
        stats = {
            "total_chunks": len(chunks),
            "query_length": len(query),
            "average_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "median_score": round(sorted(scores)[len(scores) // 2], 4)
        }

        # Score distribution
        score_ranges = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for score in scores:
            if score < 0.2:
                score_ranges["0.0-0.2"] += 1
            elif score < 0.4:
                score_ranges["0.2-0.4"] += 1
            elif score < 0.6:
                score_ranges["0.4-0.6"] += 1
            elif score < 0.8:
                score_ranges["0.6-0.8"] += 1
            else:
                score_ranges["0.8-1.0"] += 1
        stats["score_distribution"] = score_ranges

        # Geographic distribution
        lat_ranges = {"North (>30°)": 0, "Tropical (30°S-30°N)": 0, "South (<-30°)": 0}
        for chunk in chunks:
            lat, _ = chunk.get_location()
            if lat > 30:
                lat_ranges["North (>30°)"] += 1
            elif lat < -30:
                lat_ranges["South (<-30°)"] += 1
            else:
                lat_ranges["Tropical (30°S-30°N)"] += 1
        stats["geographic_distribution"] = lat_ranges

        # Basin distribution
        basin_counts = {}
        for chunk in chunks:
            basin = chunk.metadata.get("basin", "Unknown")
            basin_counts[f"Basin {basin}"] = basin_counts.get(f"Basin {basin}", 0) + 1
        stats["basin_distribution"] = basin_counts

        # Temporal distribution (simplified by year)
        year_counts = {}
        for chunk in chunks:
            timestamp = chunk.get_timestamp()
            if timestamp:
                try:
                    year = timestamp[:4]  # Extract year from ISO format
                    year_counts[year] = year_counts.get(year, 0) + 1
                except:
                    year_counts["Unknown"] = year_counts.get("Unknown", 0) + 1
        stats["temporal_distribution"] = year_counts

        return stats

    except Exception as e:
        return {
            "error": f"Failed to calculate statistics: {str(e)}",
            "total_chunks": len(chunks) if chunks else 0
        }


def _error_response(query: str, error_message: str, error_type: str) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        query (str): Original query
        error_message (str): Error description
        error_type (str): Type of error

    Returns:
        Dict[str, Any]: Error response dictionary
    """
    return {
        "query": query,
        "chunks": [],
        "total_found": 0,
        "error": error_message,
        "error_type": error_type,
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "suggestions": _get_error_suggestions(error_type)
    }


def _get_error_suggestions(error_type: str) -> List[str]:
    """
    Get suggestions for resolving errors.

    Args:
        error_type (str): Type of error encountered

    Returns:
        List[str]: List of suggestions
    """
    suggestions = {
        "retriever_error": [
            "Check if the FAISS index is properly loaded",
            "Verify the configuration file path",
            "Ensure the embedding model is accessible",
            "Try a simpler query"
        ],
        "validation_error": [
            "Ensure the query is not empty",
            "Check that k is a positive integer",
            "Verify score_threshold is between 0 and 1",
            "Review metadata filter format"
        ],
        "unexpected_error": [
            "Check system resources and dependencies",
            "Review server logs for detailed error information",
            "Try again with a simpler query",
            "Contact system administrator if problem persists"
        ]
    }

    return suggestions.get(error_type, ["Check logs for more information", "Try again"])


def test_retrieve_docs_tool():
    """
    Test function for the retrieve_docs tool.
    """
    print("Testing retrieve_docs tool...")

    # Test cases
    test_queries = [
        "temperature profile Atlantic Ocean",
        "salinity measurements North Pacific",
        "deep water properties Mediterranean",
        "ARGO float cycle 100",
        ""  # Empty query test
    ]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            result = retrieve_docs_tool(query=query, k=3)

            if result["status"] == "success":
                print(f"✓ Success: Found {result['total_found']} chunks")
                if result["chunks"]:
                    chunk = result["chunks"][0]
                    print(f"  Top result: {chunk['chunk_id']} (score: {chunk['score']})")
                    print(
                        f"  Location: {chunk['metadata']['location']['latitude']:.2f}°N, {chunk['metadata']['location']['longitude']:.2f}°E")
            else:
                print(f"✗ Error: {result['error']}")

        except Exception as e:
            print(f"✗ Exception: {str(e)}")

    print("\nTest completed.")


if __name__ == "__main__":
    # Run tests if executed directly
    test_retrieve_docs_tool()