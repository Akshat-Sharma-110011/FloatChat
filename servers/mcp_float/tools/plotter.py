"""
FloatChat Plotter Tool

This tool generates visualizations from oceanographic data including time series plots,
depth profiles, scatter plots, and statistical distributions. It supports both
SQL-based data queries and semantic search results for flexible visualization.

Author: FloatChat Team
Created: 2025-01-XX
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from servers.mcp_float.tools.query_sql import query_sql_tool
from servers.mcp_float.tools.retrieve_docs import retrieve_docs_tool


class PlotterError(Exception):
    """Custom exception for plotter tool operations."""
    pass


class OceanographicPlotter:
    """
    Specialized plotter for oceanographic data visualization.
    """

    def __init__(self, output_dir: str = "data/index"):
        """
        Initialize the oceanographic plotter.

        Args:
            output_dir (str): Directory to save plot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("OceanographicPlotter")

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")

    def create_timeseries_plot(self, data: pd.DataFrame,
                               x_column: str, y_column: str,
                               title: Optional[str] = None,
                               plot_type: str = "line") -> Dict[str, Any]:
        """
        Create a time series plot from oceanographic data.

        Args:
            data (pd.DataFrame): Data to plot
            x_column (str): Column for X-axis (typically time)
            y_column (str): Column for Y-axis (measurement)
            title (Optional[str]): Plot title
            plot_type (str): Type of plot (line, scatter, histogram)

        Returns:
            Dict[str, Any]: Plot information and file path
        """
        try:
            self.logger.info(f"[PLOT] Creating {plot_type} plot: {x_column} vs {y_column}")

            if data.empty:
                raise ValueError("No data provided for plotting")

            if x_column not in data.columns:
                raise ValueError(f"X-axis column '{x_column}' not found in data")

            if y_column not in data.columns:
                raise ValueError(f"Y-axis column '{y_column}' not found in data")

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Prepare data
            x_data = data[x_column].dropna()
            y_data = data[y_column].dropna()

            if len(x_data) == 0 or len(y_data) == 0:
                raise ValueError("No valid data points for plotting")

            # Create plot based on type
            if plot_type == "line":
                ax.plot(x_data, y_data, linewidth=2, marker='o', markersize=4, alpha=0.7)
            elif plot_type == "scatter":
                ax.scatter(x_data, y_data, alpha=0.6, s=50)
            elif plot_type == "histogram":
                ax.hist(y_data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel(y_column)
                ax.set_ylabel('Frequency')
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

            # Set labels and title
            if plot_type != "histogram":
                ax.set_xlabel(self._format_axis_label(x_column))
                ax.set_ylabel(self._format_axis_label(y_column))

            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"{self._format_axis_label(y_column)} vs {self._format_axis_label(x_column)}",
                             fontsize=14, fontweight='bold')

            # Format axes
            ax.grid(True, alpha=0.3)

            # Handle time axis formatting
            if 'time' in x_column.lower() or 'date' in x_column.lower():
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(x_data):
                        x_data = pd.to_datetime(x_data)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    plt.xticks(rotation=45)
                except:
                    pass  # If datetime conversion fails, use default formatting

            # Adjust layout
            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oceanographic_plot_{plot_type}_{timestamp}.png"
            filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            # Calculate plot statistics
            stats = self._calculate_plot_stats(data, x_column, y_column, plot_type)

            plot_info = {
                "plot_path": str(filepath),
                "plot_type": plot_type,
                "x_column": x_column,
                "y_column": y_column,
                "title": title or f"{self._format_axis_label(y_column)} vs {self._format_axis_label(x_column)}",
                "data_points": len(data),
                "statistics": stats,
                "creation_time": datetime.now().isoformat()
            }

            self.logger.info(f"[SUCCESS] Plot saved: {filepath}")

            return plot_info

        except Exception as e:
            self.logger.error(f"[ERROR] Plot creation failed: {str(e)}")
            raise PlotterError(f"Plot creation failed: {str(e)}")

    def create_depth_profile_plot(self, data: pd.DataFrame,
                                  depth_column: str = "pressure_decibar",
                                  value_columns: List[str] = None,
                                  title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a depth profile plot (typical for oceanographic data).

        Args:
            data (pd.DataFrame): Data to plot
            depth_column (str): Column representing depth/pressure
            value_columns (List[str]): Columns to plot against depth
            title (Optional[str]): Plot title

        Returns:
            Dict[str, Any]: Plot information and file path
        """
        try:
            self.logger.info("[PLOT] Creating depth profile plot")

            if data.empty:
                raise ValueError("No data provided for plotting")

            if depth_column not in data.columns:
                raise ValueError(f"Depth column '{depth_column}' not found in data")

            # Default value columns if not specified
            if value_columns is None:
                potential_columns = ['temperature_degc', 'salinity_psu']
                value_columns = [col for col in potential_columns if col in data.columns]

            if not value_columns:
                raise ValueError("No valid value columns found for depth profile")

            # Create subplots
            n_cols = len(value_columns)
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 8), sharey=True)

            if n_cols == 1:
                axes = [axes]

            # Create depth profiles
            for i, value_col in enumerate(value_columns):
                if value_col not in data.columns:
                    continue

                ax = axes[i]

                # Filter valid data
                valid_data = data[[depth_column, value_col]].dropna()

                if len(valid_data) == 0:
                    continue

                # Plot profile (depth on y-axis, values on x-axis)
                ax.plot(valid_data[value_col], valid_data[depth_column],
                        linewidth=2, marker='o', markersize=3)

                ax.set_xlabel(self._format_axis_label(value_col))
                if i == 0:  # Only label y-axis on first subplot
                    ax.set_ylabel(self._format_axis_label(depth_column))

                ax.invert_yaxis()  # Invert y-axis for depth
                ax.grid(True, alpha=0.3)

            # Set overall title
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold')
            else:
                fig.suptitle("Oceanographic Depth Profile", fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"depth_profile_{timestamp}.png"
            filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            plot_info = {
                "plot_path": str(filepath),
                "plot_type": "depth_profile",
                "depth_column": depth_column,
                "value_columns": value_columns,
                "title": title or "Oceanographic Depth Profile",
                "data_points": len(data),
                "creation_time": datetime.now().isoformat()
            }

            self.logger.info(f"[SUCCESS] Depth profile saved: {filepath}")

            return plot_info

        except Exception as e:
            self.logger.error(f"[ERROR] Depth profile creation failed: {str(e)}")
            raise PlotterError(f"Depth profile creation failed: {str(e)}")

    def create_scatter_matrix(self, data: pd.DataFrame,
                              columns: List[str] = None,
                              title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a scatter plot matrix for oceanographic variables.

        Args:
            data (pd.DataFrame): Data to plot
            columns (List[str]): Columns to include in matrix
            title (Optional[str]): Plot title

        Returns:
            Dict[str, Any]: Plot information and file path
        """
        try:
            self.logger.info("[PLOT] Creating scatter matrix")

            if data.empty:
                raise ValueError("No data provided for plotting")

            # Default columns if not specified
            if columns is None:
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                # Prioritize oceanographic variables
                preferred = ['temperature_degc', 'salinity_psu', 'pressure_decibar', 'latitude', 'longitude']
                columns = [col for col in preferred if col in numeric_columns]
                if not columns:
                    columns = numeric_columns[:4]  # Limit to 4 columns for readability

            if len(columns) < 2:
                raise ValueError("Need at least 2 numeric columns for scatter matrix")

            # Filter data to selected columns
            plot_data = data[columns].dropna()

            if len(plot_data) == 0:
                raise ValueError("No valid data points after filtering")

            # Create scatter matrix
            fig = plt.figure(figsize=(12, 12))
            axes = pd.plotting.scatter_matrix(plot_data, alpha=0.6, figsize=(12, 12),
                                              diagonal='hist', grid=True)

            # Format axis labels
            for i, ax_row in enumerate(axes):
                for j, ax in enumerate(ax_row):
                    if i == len(axes) - 1:  # Bottom row
                        ax.set_xlabel(self._format_axis_label(columns[j]))
                    if j == 0:  # Left column
                        ax.set_ylabel(self._format_axis_label(columns[i]))

            # Set title
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold')
            else:
                fig.suptitle("Oceanographic Variables Scatter Matrix", fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scatter_matrix_{timestamp}.png"
            filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            plot_info = {
                "plot_path": str(filepath),
                "plot_type": "scatter_matrix",
                "columns": columns,
                "title": title or "Oceanographic Variables Scatter Matrix",
                "data_points": len(plot_data),
                "creation_time": datetime.now().isoformat()
            }

            self.logger.info(f"[SUCCESS] Scatter matrix saved: {filepath}")

            return plot_info

        except Exception as e:
            self.logger.error(f"[ERROR] Scatter matrix creation failed: {str(e)}")
            raise PlotterError(f"Scatter matrix creation failed: {str(e)}")

    def _format_axis_label(self, column_name: str) -> str:
        """
        Format column names for axis labels.

        Args:
            column_name (str): Raw column name

        Returns:
            str: Formatted label
        """
        label_mapping = {
            'temperature_degc': 'Temperature (°C)',
            'salinity_psu': 'Salinity (PSU)',
            'pressure_decibar': 'Pressure (dbar)',
            'latitude': 'Latitude (°N)',
            'longitude': 'Longitude (°E)',
            'cycle_number': 'Cycle Number',
            'basin': 'Basin',
            'timestamp': 'Time'
        }

        return label_mapping.get(column_name, column_name.replace('_', ' ').title())

    def _calculate_plot_stats(self, data: pd.DataFrame,
                              x_column: str, y_column: str,
                              plot_type: str) -> Dict[str, Any]:
        """
        Calculate statistics for the plotted data.

        Args:
            data (pd.DataFrame): Plotted data
            x_column (str): X-axis column
            y_column (str): Y-axis column
            plot_type (str): Type of plot

        Returns:
            Dict[str, Any]: Plot statistics
        """
        try:
            stats = {
                "data_points": len(data),
                "plot_type": plot_type
            }

            # X-axis statistics
            if x_column in data.columns:
                x_data = data[x_column].dropna()
                if len(x_data) > 0 and pd.api.types.is_numeric_dtype(x_data):
                    stats["x_axis"] = {
                        "column": x_column,
                        "min": float(x_data.min()),
                        "max": float(x_data.max()),
                        "mean": float(x_data.mean()),
                        "count": len(x_data)
                    }

            # Y-axis statistics
            if y_column in data.columns:
                y_data = data[y_column].dropna()
                if len(y_data) > 0 and pd.api.types.is_numeric_dtype(y_data):
                    stats["y_axis"] = {
                        "column": y_column,
                        "min": float(y_data.min()),
                        "max": float(y_data.max()),
                        "mean": float(y_data.mean()),
                        "count": len(y_data)
                    }

            return stats

        except Exception as e:
            return {"error": f"Failed to calculate statistics: {str(e)}"}


def plot_timeseries_tool(data_query: str,
                         plot_type: str = "line",
                         title: Optional[str] = None,
                         x_column: Optional[str] = None,
                         y_column: Optional[str] = None,
                         config: Optional[Dict[str, Any]] = None,
                         retriever: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tool: Generate a time series plot from oceanographic data and return the image path.

    This tool creates visualizations from ARGO data using either SQL queries or
    semantic search results, supporting various plot types for oceanographic analysis.

    Args:
        data_query (str): SQL query or semantic query to get data for plotting
        plot_type (str): Type of plot (line, scatter, histogram, depth_profile, scatter_matrix)
        title (Optional[str]): Plot title
        x_column (Optional[str]): Column for X-axis
        y_column (Optional[str]): Column for Y-axis
        config (Optional[Dict[str, Any]]): Configuration dictionary
        retriever (Optional[Any]): Retriever instance for semantic search

    Returns:
        Dict[str, Any]: Plot information including file path
    """
    # Setup logging
    logger = logging.getLogger("PlotTimeseresTool")

    try:
        logger.info(f"[PLOT] Starting plot generation: {plot_type}")
        logger.info(f"[PLOT] Data query: {data_query[:100]}...")

        # Initialize plotter
        output_dir = config.get('data', {}).get('index_dir', 'data/index') if config else 'data/index'
        plotter = OceanographicPlotter(output_dir=output_dir)

        # Get data based on query type
        data = _get_data_for_plotting(data_query, config, retriever, logger)

        if data is None or data.empty:
            return _error_response(data_query, "No data retrieved for plotting", "no_data")

        logger.info(f"[PLOT] Retrieved {len(data)} rows, {len(data.columns)} columns")

        # Determine columns if not specified
        if not x_column or not y_column:
            x_column, y_column = _determine_plot_columns(data, plot_type, x_column, y_column)

        # Create plot based on type
        if plot_type == "depth_profile":
            depth_col = x_column if 'pressure' in str(x_column).lower() else 'pressure_decibar'
            value_cols = [y_column] if y_column else None
            plot_info = plotter.create_depth_profile_plot(
                data=data,
                depth_column=depth_col,
                value_columns=value_cols,
                title=title
            )
        elif plot_type == "scatter_matrix":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            plot_info = plotter.create_scatter_matrix(
                data=data,
                columns=numeric_cols[:5],  # Limit to 5 columns
                title=title
            )
        else:
            plot_info = plotter.create_timeseries_plot(
                data=data,
                x_column=x_column,
                y_column=y_column,
                title=title,
                plot_type=plot_type
            )

        # Prepare response
        response = {
            "data_query": data_query,
            "plot_info": plot_info,
            "plot_path": plot_info["plot_path"],
            "plot_type": plot_type,
            "data_summary": {
                "rows": len(data),
                "columns": len(data.columns),
                "column_names": list(data.columns)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        logger.info(f"[SUCCESS] Plot created: {plot_info['plot_path']}")

        return response

    except PlotterError as e:
        logger.error(f"[ERROR] Plotter error: {str(e)}")
        return _error_response(data_query, str(e), "plotter_error")

    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        return _error_response(data_query, f"Unexpected error: {str(e)}", "unexpected_error")


def _get_data_for_plotting(data_query: str, config: Optional[Dict[str, Any]],
                           retriever: Optional[Any], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Get data for plotting from either SQL query or semantic search.

    Args:
        data_query (str): Query string
        config (Optional[Dict[str, Any]]): Configuration
        retriever (Optional[Any]): Retriever instance
        logger (logging.Logger): Logger instance

    Returns:
        Optional[pd.DataFrame]: Data for plotting
    """
    try:
        # Check if it's an SQL query
        if data_query.strip().lower().startswith('select'):
            logger.info("[PLOT] Executing SQL query for data")
            sql_result = query_sql_tool(sql=data_query, limit=10000, config=config)

            if sql_result["status"] == "success" and sql_result["rows"]:
                return pd.DataFrame(sql_result["rows"])
            else:
                logger.warning(f"[WARNING] SQL query failed: {sql_result.get('error', 'Unknown error')}")
                return None

        else:
            # Use semantic search
            logger.info("[PLOT] Using semantic search for data")
            if not retriever:
                logger.error("[ERROR] Retriever not available for semantic search")
                return None

            search_result = retrieve_docs_tool(
                query=data_query,
                k=20,  # Get more chunks for better data coverage
                retriever=retriever
            )

            if search_result["status"] == "success" and search_result["chunks"]:
                # Convert chunks to DataFrame
                rows = []
                for chunk in search_result["chunks"]:
                    for measurement in chunk.get("measurements", []):
                        row = {
                            "chunk_id": chunk["chunk_id"],
                            "timestamp": chunk["metadata"]["timestamp"],
                            "latitude": chunk["metadata"]["location"]["latitude"],
                            "longitude": chunk["metadata"]["location"]["longitude"],
                            "basin": chunk["metadata"]["basin"],
                            "cycle_number": chunk["metadata"]["cycle_number"],
                            **measurement
                        }
                        rows.append(row)

                if rows:
                    return pd.DataFrame(rows)
                else:
                    logger.warning("[WARNING] No measurements found in search results")
                    return None
            else:
                logger.warning(f"[WARNING] Semantic search failed: {search_result.get('error', 'No results')}")
                return None

    except Exception as e:
        logger.error(f"[ERROR] Failed to get data for plotting: {str(e)}")
        return None


def _determine_plot_columns(data: pd.DataFrame, plot_type: str,
                            x_column: Optional[str], y_column: Optional[str]) -> Tuple[str, str]:
    """
    Determine appropriate columns for plotting based on data and plot type.

    Args:
        data (pd.DataFrame): Available data
        plot_type (str): Type of plot
        x_column (Optional[str]): Specified X column
        y_column (Optional[str]): Specified Y column

    Returns:
        Tuple[str, str]: (x_column, y_column)
    """
    columns = data.columns.tolist()

    # If both columns specified, return them
    if x_column and y_column and x_column in columns and y_column in columns:
        return x_column, y_column

    # Default mappings based on available columns
    time_columns = ['timestamp', 'date', 'time']
    depth_columns = ['pressure_decibar', 'depth', 'pressure']
    measurement_columns = ['temperature_degc', 'salinity_psu']

    # Find available columns
    available_time = [col for col in time_columns if col in columns]
    available_depth = [col for col in depth_columns if col in columns]
    available_measurements = [col for col in measurement_columns if col in columns]

    # Determine columns based on plot type and available data
    if plot_type == "depth_profile":
        x_col = available_depth[0] if available_depth else columns[0]
        y_col = available_measurements[0] if available_measurements else columns[1] if len(columns) > 1 else columns[0]
    elif available_time and available_measurements:
        x_col = available_time[0]
        y_col = available_measurements[0]
    elif len(columns) >= 2:
        # Use first two numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
        else:
            x_col, y_col = columns[0], columns[1]
    else:
        # Fallback
        x_col = y_col = columns[0]

    return x_col, y_col


def _error_response(data_query: str, error_message: str, error_type: str) -> Dict[str, Any]:
    """
    Create standardized error response for plotting operations.

    Args:
        data_query (str): Original data query
        error_message (str): Error description
        error_type (str): Type of error

    Returns:
        Dict[str, Any]: Error response dictionary
    """
    return {
        "data_query": data_query,
        "plot_path": None,
        "plot_info": None,
        "error": error_message,
        "error_type": error_type,
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "suggestions": _get_plot_error_suggestions(error_type)
    }


def _get_plot_error_suggestions(error_type: str) -> List[str]:
    """
    Get suggestions for resolving plotting errors.

    Args:
        error_type (str): Type of error

    Returns:
        List[str]: List of suggestions
    """
    suggestions = {
        "no_data": [
            "Check if your query returns valid data",
            "Verify column names and table references",
            "Try a broader search query",
            "Ensure the database contains relevant data"
        ],
        "plotter_error": [
            "Verify column names exist in the data",
            "Check data types (numeric columns for numerical plots)",
            "Ensure sufficient data points for visualization",
            "Try a different plot type"
        ],
        "unexpected_error": [
            "Check system resources and dependencies",
            "Review data query format",
            "Try with simpler data or plot type",
            "Contact administrator if problem persists"
        ]
    }

    return suggestions.get(error_type, ["Check query and try again"])


def test_plotter_tool():
    """
    Test function for the plotter tool.
    """
    print("Testing plotter tool...")

    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
        'temperature_degc': np.random.normal(15, 3, 100),
        'salinity_psu': np.random.normal(35, 2, 100),
        'pressure_decibar': np.random.uniform(0, 1000, 100),
        'latitude': np.random.uniform(-60, 60, 100),
        'longitude': np.random.uniform(-180, 180, 100)
    })

    # Test different plot types
    test_cases = [
        {
            "query": "SELECT * FROM profiles LIMIT 100",
            "plot_type": "line",
            "x_column": "timestamp",
            "y_column": "temperature_degc",
            "title": "Temperature Time Series"
        },
        {
            "query": "SELECT * FROM profiles WHERE pressure_decibar < 500",
            "plot_type": "scatter",
            "x_column": "salinity_psu",
            "y_column": "temperature_degc",
            "title": "Temperature vs Salinity"
        },
        {
            "query": "temperature profile ocean data",
            "plot_type": "depth_profile",
            "title": "Ocean Depth Profile"
        },
        {
            "query": "SELECT * FROM profiles LIMIT 50",
            "plot_type": "scatter_matrix",
            "title": "Oceanographic Variables Matrix"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['plot_type']} plot")
        try:
            result = plot_timeseries_tool(
                data_query=test_case["query"],
                plot_type=test_case["plot_type"],
                title=test_case.get("title"),
                x_column=test_case.get("x_column"),
                y_column=test_case.get("y_column")
            )

            if result["status"] == "success":
                print(f"✓ Success: Plot saved to {result['plot_path']}")
                print(f"  Data points: {result['data_summary']['rows']}")
            else:
                print(f"✗ Error: {result['error']}")

        except Exception as e:
            print(f"✗ Exception: {str(e)}")

    print("\nPlotter test completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FloatChat Plotter Tool")
    parser.add_argument("--test", action="store_true", help="Run test plots")
    parser.add_argument("--query", type=str, help="Data query for plotting")
    parser.add_argument("--type", type=str, default="line",
                        choices=["line", "scatter", "histogram", "depth_profile", "scatter_matrix"],
                        help="Plot type")
    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--x-column", type=str, help="X-axis column")
    parser.add_argument("--y-column", type=str, help="Y-axis column")
    parser.add_argument("--output", type=str, default="data/index", help="Output directory")

    args = parser.parse_args()

    if args.test:
        test_plotter_tool()
    elif args.query:
        result = plot_timeseries_tool(
            data_query=args.query,
            plot_type=args.type,
            title=args.title,
            x_column=args.x_column,
            y_column=args.y_column
        )

        if result["status"] == "success":
            print(f"Plot created successfully: {result['plot_path']}")
            print(f"Data points: {result['data_summary']['rows']}")
            if result["plot_info"].get("statistics"):
                stats = result["plot_info"]["statistics"]
                print(f"Plot statistics: {stats}")
        else:
            print(f"Plot creation failed: {result['error']}")
            if result.get("suggestions"):
                print("Suggestions:")
                for suggestion in result["suggestions"]:
                    print(f"  - {suggestion}")
    else:
        print("Use --test to run tests or --query to create a plot. See --help for options.")