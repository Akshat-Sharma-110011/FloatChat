"""
FloatChat MCP Server - Fixed Version

Model Context Protocol (MCP) server that exposes oceanographic data tools for LLM interaction.
This server provides three main tools:
- retrieve_docs: Semantic search over ARGO oceanographic data
- query_sql: SQL queries against the ARGO database
- plot_timeseries: Generate visualizations from oceanographic data

The server includes intelligent intent detection to automatically determine which tools
to use based on user queries, using a lightweight HuggingFace LLM for classification.

Author: FloatChat Team
Created: 2025-01-XX
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import warnings
import traceback

# MCP imports - Updated for current SDK
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# FloatChat imports
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from src.rag.retriever import Retriever, RetrieverError
    from servers.mcp_float.tools.retrieve_docs import retrieve_docs_tool
    from servers.mcp_float.tools.query_sql import query_sql_tool
    from servers.mcp_float.tools.plotter import plot_timeseries_tool
except ImportError as e:
    print(f"[ERROR] Failed to import FloatChat modules: {e}")
    print("Please ensure you're running from the project root directory")
    print("Current working directory:", Path.cwd())
    sys.exit(1)

# Intent detection imports
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import yaml


class MCPServerError(Exception):
    """Custom exception for MCP server operations."""
    pass


class IntentClassifier:
    """
    Intent classification system to determine which tools to use based on user queries.
    Uses a lightweight HuggingFace model for zero-shot classification.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize intent classifier with HuggingFace model.

        Args:
            model_name (str): HuggingFace model for zero-shot classification
        """
        self.logger = logging.getLogger("IntentClassifier")
        self.model_name = model_name
        self.classifier = None
        self._load_model()

        # Define intent categories and keywords
        self.intent_categories = {
            "retrieve_docs": {
                "description": "Search for oceanographic information, context, and similar profiles",
                "keywords": ["what", "show", "find", "about", "information", "context", "similar",
                             "conditions", "profiles", "explain", "describe", "ocean", "atlantic",
                             "pacific", "indian", "mediterranean", "basin", "region", "area"],
                "patterns": ["what are", "show me", "find", "explain", "describe", "information about"]
            },
            "query_sql": {
                "description": "Execute precise database queries for numerical data and statistics",
                "keywords": ["how many", "count", "average", "maximum", "minimum", "between",
                             "statistics", "data", "number", "total", "sum", "range", "filter",
                             "where", "measurements", "records", "depth", "temperature", "salinity"],
                "patterns": ["how many", "count", "average", "max", "min", "statistics", "data for"]
            },
            "plot_timeseries": {
                "description": "Generate charts, graphs, and visualizations",
                "keywords": ["plot", "chart", "graph", "visualize", "show", "display", "create",
                             "time series", "trends", "versus", "vs", "over time", "profile",
                             "visualization", "draw", "generate"],
                "patterns": ["plot", "chart", "graph", "visualize", "show graph", "create plot"]
            }
        }

    def _load_model(self) -> None:
        """Load the HuggingFace model for intent classification."""
        try:
            self.logger.info(f"[INTENT] Loading classification model: {self.model_name}")

            # Suppress transformers warnings
            warnings.filterwarnings("ignore", message=".*resume_download.*")
            warnings.filterwarnings("ignore", message=".*torch.*")

            # Initialize zero-shot classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )

            self.logger.info("[INTENT] Classification model loaded successfully")

        except Exception as e:
            self.logger.warning(f"[WARNING] Failed to load classification model: {str(e)}")
            self.classifier = None

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user intent to determine appropriate tool usage.

        Args:
            query (str): User query text

        Returns:
            Dict[str, Any]: Classification results with recommended tools
        """
        try:
            query_lower = query.lower()

            # Rule-based classification (fallback and primary)
            rule_scores = self._rule_based_classification(query_lower)

            # ML-based classification (if model available)
            ml_scores = {}
            if self.classifier:
                try:
                    candidate_labels = list(self.intent_categories.keys())
                    result = self.classifier(query, candidate_labels)
                    ml_scores = {
                        label: score
                        for label, score in zip(result['labels'], result['scores'])
                    }
                except Exception as e:
                    self.logger.warning(f"[WARNING] ML classification failed: {str(e)}")

            # Combine rule-based and ML scores
            combined_scores = {}
            for tool in self.intent_categories.keys():
                rule_score = rule_scores.get(tool, 0.0)
                ml_score = ml_scores.get(tool, 0.0)
                # Weight rule-based higher as it's more reliable for our domain
                combined_scores[tool] = (0.7 * rule_score) + (0.3 * ml_score)

            # Determine recommended tools
            max_score = max(combined_scores.values()) if combined_scores else 0.0
            threshold = 0.3  # Minimum confidence threshold

            recommended_tools = []
            if max_score > threshold:
                # Primary tool
                primary_tool = max(combined_scores.items(), key=lambda x: x[1])[0]
                recommended_tools.append(primary_tool)

                # Secondary tools if scores are close
                for tool, score in combined_scores.items():
                    if tool != primary_tool and score > (max_score * 0.8):
                        recommended_tools.append(tool)
            else:
                # Default to retrieve_docs for general queries
                recommended_tools = ["retrieve_docs"]

            classification_result = {
                "query": query,
                "recommended_tools": recommended_tools,
                "confidence_scores": combined_scores,
                "max_confidence": max_score,
                "classification_method": "hybrid" if self.classifier else "rule_based",
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"[INTENT] Classified query -> Tools: {recommended_tools}, Confidence: {max_score:.3f}")

            return classification_result

        except Exception as e:
            self.logger.error(f"[ERROR] Intent classification failed: {str(e)}")
            # Fallback to retrieve_docs
            return {
                "query": query,
                "recommended_tools": ["retrieve_docs"],
                "confidence_scores": {"retrieve_docs": 0.5},
                "max_confidence": 0.5,
                "classification_method": "fallback",
                "error": str(e)
            }

    def _rule_based_classification(self, query_lower: str) -> Dict[str, float]:
        """
        Rule-based intent classification using keywords and patterns.

        Args:
            query_lower (str): Lowercase query text

        Returns:
            Dict[str, float]: Scores for each tool category
        """
        scores = {tool: 0.0 for tool in self.intent_categories.keys()}

        for tool, config in self.intent_categories.items():
            score = 0.0

            # Keyword matching
            keywords = config.get("keywords", [])
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1.0

            # Pattern matching
            patterns = config.get("patterns", [])
            for pattern in patterns:
                if pattern in query_lower:
                    score += 2.0  # Patterns are more specific

            # Normalize by total possible score
            max_possible = len(keywords) + (len(patterns) * 2)
            if max_possible > 0:
                scores[tool] = score / max_possible

            # Special rules for disambiguation
            if tool == "query_sql":
                # Boost for numerical/statistical queries
                if any(word in query_lower for word in ["how many", "count", "average", "total", "statistics"]):
                    scores[tool] += 0.3
                if any(word in query_lower for word in ["between", "range", "from", "to", "where"]):
                    scores[tool] += 0.2

            elif tool == "plot_timeseries":
                # Boost for visualization requests
                if any(word in query_lower for word in ["plot", "chart", "graph", "visualize"]):
                    scores[tool] += 0.4
                if any(word in query_lower for word in ["time series", "over time", "trend"]):
                    scores[tool] += 0.3

            elif tool == "retrieve_docs":
                # Boost for general information requests
                if any(word in query_lower for word in ["what", "explain", "about", "information"]):
                    scores[tool] += 0.2

        return scores


# Global variables for server components
intent_classifier = None
retriever = None
config = {}
logger = None


def _setup_logging(config_path: str = "configs/intel.yaml") -> logging.Logger:
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_filename = logs_dir / f"mcp_server_{timestamp}.log"

    logger = logging.getLogger("FloatChatMCPServer")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler for errors only in STDIO mode
    if os.environ.get('MCP_TRANSPORT') != 'stdio':
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.info(f"[LOGGING] MCP Server logging initialized - Log file: {log_filename}")
    return logger


def _load_config(config_path: str = "configs/intel.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"[WARNING] Config file not found: {config_path}, using defaults", file=sys.stderr)
            return _get_default_config()

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """Get default configuration when config file is not available."""
    return {
        'database': {
            'url': 'postgresql://postgres:Strong.password177013@localhost:6000/floatchat'
        },
        'data': {
            'index_dir': 'data/index'
        },
        'retriever': {
            'model_name': 'all-MiniLM-L6-v2'
        }
    }


def _initialize_components(config_path: str = "configs/intel.yaml"):
    """Initialize all server components."""
    global intent_classifier, retriever, config, logger

    try:
        # Setup logging
        logger = _setup_logging(config_path)
        logger.info("[INIT] Initializing FloatChat MCP Server")

        # Load configs
        config = _load_config(config_path)

        # Initialize intent classifier (make it optional)
        try:
            intent_classifier = IntentClassifier()
        except Exception as e:
            logger.warning(f"[WARNING] Failed to initialize intent classifier: {e}")
            intent_classifier = None

        # Initialize retriever (make it optional)
        try:
            retriever = Retriever(config_path=config_path)
            logger.info("[INIT] Retriever initialized successfully")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to initialize retriever: {str(e)}")
            retriever = None

        logger.info("[INIT] FloatChat MCP Server initialized successfully")

    except Exception as e:
        print(f"[CRITICAL] Server initialization failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Don't exit, allow server to start with limited functionality


# Initialize the FastMCP app at module level - MUST be named 'app', 'server', or 'mcp'
mcp = FastMCP("FloatChat-MCP")


# Register MCP tools
@mcp.tool()
def retrieve_docs(query: str, k: int = 5, score_threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Retrieve relevant oceanographic data chunks using semantic similarity search.

    This tool searches through ARGO float profiles and oceanographic measurements
    to find information relevant to the user's query.

    Args:
        query: Query text for semantic search
        k: Number of results to retrieve (default: 5)
        score_threshold: Minimum similarity score (optional)

    Returns:
        Dictionary containing search results with retrieved documents
    """
    try:
        if logger:
            logger.info(f"[TOOL] retrieve_docs called with query: '{query}'")

        if not retriever:
            return {
                "error": "Retriever not available - please check server initialization",
                "chunks": [],
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }

        # Use the retrieve_docs tool function
        result = retrieve_docs_tool(
            query=query,
            k=k,
            score_threshold=score_threshold,
            retriever=retriever
        )

        if logger:
            logger.info(f"[SUCCESS] Retrieved {len(result.get('chunks', []))} documents")
        return result

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] retrieve_docs failed: {str(e)}")
        return {
            "error": str(e),
            "chunks": [],
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


@mcp.tool()
def query_sql(sql: str, limit: Optional[int] = 1000) -> Dict[str, Any]:
    """
    Execute SQL queries against the ARGO oceanographic database.

    This tool allows precise data filtering and statistical analysis
    of oceanographic measurements.

    Args:
        sql: SQL query to execute
        limit: Maximum number of rows to return (default: 1000)

    Returns:
        Dictionary containing query results with rows and columns
    """
    try:
        if logger:
            logger.info(f"[TOOL] query_sql called with query: {sql[:100]}...")

        # Use the query_sql tool function
        result = query_sql_tool(
            sql=sql,
            limit=limit,
            config=config
        )

        row_count = len(result.get('rows', []))
        if logger:
            logger.info(f"[SUCCESS] SQL query returned {row_count} rows")
        return result

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] query_sql failed: {str(e)}")
        return {
            "error": str(e),
            "rows": [],
            "columns": [],
            "query": sql,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


@mcp.tool()
def plot_timeseries(
        data_query: str,
        plot_type: str = "line",
        title: Optional[str] = None,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate visualizations from oceanographic data.

    This tool creates charts, graphs, and plots to visualize
    temperature, salinity, pressure, and other oceanographic measurements.

    Args:
        data_query: Query or SQL to get data for plotting
        plot_type: Type of plot (line, scatter, histogram) (default: "line")
        title: Plot title (optional)
        x_column: Column for X-axis (optional)
        y_column: Column for Y-axis (optional)

    Returns:
        Dictionary containing plot information and file path
    """
    try:
        if logger:
            logger.info(f"[TOOL] plot_timeseries called with data_query: {data_query[:100]}...")

        # Use the plot_timeseries tool function
        result = plot_timeseries_tool(
            data_query=data_query,
            plot_type=plot_type,
            title=title,
            x_column=x_column,
            y_column=y_column,
            config=config,
            retriever=retriever
        )

        if logger:
            logger.info(f"[SUCCESS] Generated plot: {result.get('plot_path', 'N/A')}")
        return result

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] plot_timeseries failed: {str(e)}")
        return {
            "error": str(e),
            "plot_path": None,
            "data_query": data_query,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


@mcp.tool()
def classify_intent(query: str) -> Dict[str, Any]:
    """
    Classify user intent to determine appropriate tool usage.

    This tool analyzes user queries to recommend which tools should be used
    to best answer their questions about oceanographic data.

    Args:
        query: User query to classify

    Returns:
        Dictionary containing intent classification results and tool recommendations
    """
    try:
        if logger:
            logger.info(f"[TOOL] classify_intent called with query: '{query}'")

        if not intent_classifier:
            return {
                "error": "Intent classifier not available",
                "query": query,
                "recommended_tools": ["retrieve_docs"],
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }

        result = intent_classifier.classify_intent(query)

        if logger:
            logger.info(f"[SUCCESS] Intent classified - Tools: {result.get('recommended_tools', [])}")
        return result

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] classify_intent failed: {str(e)}")
        return {
            "error": str(e),
            "query": query,
            "recommended_tools": ["retrieve_docs"],
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """
    Get information about the MCP server and its capabilities.

    Returns:
        Dictionary containing server information and capabilities
    """
    try:
        return {
            "name": "FloatChat MCP Server",
            "version": "1.0.0",
            "description": "Oceanographic data analysis tools with intelligent intent detection",
            "tools": [
                {
                    "name": "retrieve_docs",
                    "description": "Search oceanographic data using semantic similarity",
                    "available": retriever is not None
                },
                {
                    "name": "query_sql",
                    "description": "Execute SQL queries against ARGO database",
                    "available": True
                },
                {
                    "name": "plot_timeseries",
                    "description": "Generate visualizations from oceanographic data",
                    "available": True
                },
                {
                    "name": "classify_intent",
                    "description": "Classify user intent for tool recommendation",
                    "available": intent_classifier is not None
                },
                {
                    "name": "get_server_info",
                    "description": "Get server information and capabilities",
                    "available": True
                }
            ],
            "capabilities": {
                "semantic_search": retriever is not None and retriever.is_loaded if retriever else False,
                "sql_queries": True,
                "data_visualization": True,
                "intent_classification": intent_classifier is not None,
                "vector_search": retriever is not None and retriever.is_loaded if retriever else False
            },
            "data_sources": {
                "argo_profiles": True,
                "total_vectors": retriever.vector_store.ntotal if retriever and retriever.is_loaded else 0
            },
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        if logger:
            logger.error(f"[ERROR] get_server_info failed: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main function to run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="FloatChat MCP Server")
    parser.add_argument("--config", "-c", default="configs/intel.yaml", help="Configuration file path")
    parser.add_argument("--info", action="store_true", help="Show server info and exit")

    args = parser.parse_args()

    try:
        # Initialize server components
        _initialize_components(config_path=args.config)

        if args.info:
            # Show server info
            info_tool = get_server_info()
            print(json.dumps(info_tool, indent=2), file=sys.stderr)
            return

        # Run the server - FastMCP will automatically detect transport
        # The 'mcp' variable will be discovered by the fastmcp CLI
        print("[INFO] FloatChat MCP Server ready", file=sys.stderr)
        mcp.run()

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Server failed: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# Initialize components when module is imported
# This allows the server to be ready when imported by fastmcp CLI
_initialize_components()

if __name__ == "__main__":
    main()