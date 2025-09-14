"""
FloatChat SQL Query Tool

This tool provides SQL query capabilities against the ARGO oceanographic database.
It allows precise data filtering, statistical analysis, and numerical computations
on oceanographic measurements with built-in security and validation.

Author: FloatChat Team
Created: 2025-01-XX
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import yaml


class QuerySQLError(Exception):
    """Custom exception for query_sql tool operations."""
    pass


class SQLValidator:
    """
    SQL query validator to ensure safe execution and prevent harmful operations.
    """

    # Allowed SQL keywords and operations
    ALLOWED_KEYWORDS = {
        'select', 'from', 'where', 'group', 'by', 'order', 'limit', 'offset',
        'having', 'as', 'and', 'or', 'not', 'in', 'like', 'between', 'is',
        'null', 'distinct', 'count', 'sum', 'avg', 'max', 'min', 'round',
        'case', 'when', 'then', 'else', 'end', 'cast', 'extract', 'date',
        'year', 'month', 'day', 'inner', 'left', 'right', 'join', 'on',
        'union', 'all', 'exists', 'with', 'asc', 'desc', 'stddev', 'variance'
    }

    # Forbidden keywords and patterns
    FORBIDDEN_KEYWORDS = {
        'drop', 'delete', 'insert', 'update', 'create', 'alter', 'truncate',
        'grant', 'revoke', 'exec', 'execute', 'sp_', 'xp_', 'cmdshell',
        'bulk', 'openrowset', 'opendatasource', 'shutdown', 'backup',
        'restore', 'dbcc', 'merge', 'call', 'procedure', 'function'
    }

    # Allowed table names
    ALLOWED_TABLES = {
        'profiles', 'trajectory', 'measurements'
    }

    def __init__(self):
        self.logger = logging.getLogger("SQLValidator")

    def validate_query(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety and compliance.

        Args:
            sql (str): SQL query to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if not sql or not sql.strip():
                return False, "SQL query cannot be empty"

            sql_lower = sql.lower().strip()

            # Check for forbidden keywords
            for keyword in self.FORBIDDEN_KEYWORDS:
                if keyword in sql_lower:
                    return False, f"Forbidden keyword detected: {keyword}"

            # Check if it's a SELECT statement
            if not sql_lower.startswith('select'):
                return False, "Only SELECT queries are allowed"

            # Check for multiple statements (basic protection against SQL injection)
            if ';' in sql and not sql.rstrip().endswith(';'):
                return False, "Multiple SQL statements are not allowed"

            # Check for comments that might hide malicious code
            if '--' in sql or '/*' in sql:
                return False, "SQL comments are not allowed"

            # Validate table references (basic check)
            tables_in_query = self._extract_table_names(sql_lower)
            for table in tables_in_query:
                if table not in self.ALLOWED_TABLES:
                    return False, f"Access to table '{table}' is not allowed"

            # Check query complexity (prevent resource exhaustion)
            if sql_lower.count('join') > 5:
                return False, "Too many JOIN operations (maximum 5 allowed)"

            if sql_lower.count('union') > 3:
                return False, "Too many UNION operations (maximum 3 allowed)"

            # Check for nested subqueries (prevent complexity)
            paren_count = sql.count('(')
            if paren_count > 10:
                return False, "Query too complex (too many nested operations)"

            return True, "Query validated successfully"

        except Exception as e:
            self.logger.error(f"[ERROR] Query validation failed: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def _extract_table_names(self, sql_lower: str) -> List[str]:
        """
        Extract table names from SQL query (basic implementation).

        Args:
            sql_lower (str): Lowercase SQL query

        Returns:
            List[str]: List of table names found in query
        """
        tables = []

        # Simple regex to find table names after FROM and JOIN
        from_pattern = r'from\s+(\w+)'
        join_pattern = r'join\s+(\w+)'

        from_matches = re.findall(from_pattern, sql_lower)
        join_matches = re.findall(join_pattern, sql_lower)

        tables.extend(from_matches)
        tables.extend(join_matches)

        return list(set(tables))  # Remove duplicates


def query_sql_tool(sql: str,
                   limit: Optional[int] = 1000,
                   config: Optional[Dict[str, Any]] = None,
                   config_path: str = "configs/intel.yaml") -> Dict[str, Any]:
    """
    Tool: Execute a read-only SQL query against the ARGO database and return results.

    This tool provides secure access to the ARGO oceanographic database for precise
    data filtering, statistical analysis, and numerical computations.

    Args:
        sql (str): SQL query to execute (SELECT only)
        limit (Optional[int]): Maximum number of rows to return
        config (Optional[Dict[str, Any]]): Configuration dictionary
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Query results with rows, columns, and metadata

    Raises:
        QuerySQLError: If query execution fails
    """
    # Setup logging
    logger = logging.getLogger("QuerySQLTool")

    try:
        logger.info(f"[SQL] Executing SQL query (length: {len(sql)} chars)")

        # Load configuration if not provided
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"[WARNING] Failed to load configs: {e}")
                config = {}

        # Get database URL
        db_url = config.get('database', {}).get('url',
                                                'postgresql://postgres:Strong.password177013@localhost:6000/floatchat')

        # Validate SQL query
        validator = SQLValidator()
        is_valid, validation_message = validator.validate_query(sql)

        if not is_valid:
            logger.error(f"[ERROR] SQL validation failed: {validation_message}")
            return _error_response(sql, f"Query validation failed: {validation_message}", "validation_error")

        logger.info("[SQL] Query validation passed")

        # Apply limit if not already present and limit is specified
        if limit and limit > 0:
            if 'limit' not in sql.lower():
                sql = sql.rstrip().rstrip(';') + f' LIMIT {limit}'
                logger.info(f"[SQL] Applied row limit: {limit}")

        # Execute query
        engine = create_engine(db_url,
                               pool_timeout=30,
                               pool_recycle=3600,
                               echo=False)

        start_time = datetime.now()

        with engine.connect() as conn:
            logger.info("[SQL] Executing query against database")
            result = conn.execute(text(sql))

            # Fetch results
            rows = result.fetchall()
            columns = list(result.keys()) if hasattr(result, 'keys') else []

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[SQL] Query executed successfully in {execution_time:.3f}s")

        # Convert results to serializable format
        processed_rows = []
        for row in rows:
            if hasattr(row, '_asdict'):
                processed_rows.append(row._asdict())
            elif hasattr(row, '_mapping'):
                processed_rows.append(dict(row._mapping))
            else:
                processed_rows.append(dict(zip(columns, row)))

        # Calculate query statistics
        stats = _calculate_query_stats(processed_rows, sql, execution_time)

        # Prepare response
        response = {
            "sql": sql,
            "rows": processed_rows,
            "columns": columns,
            "row_count": len(processed_rows),
            "column_count": len(columns),
            "execution_time_seconds": round(execution_time, 3),
            "statistics": stats,
            "query_metadata": {
                "applied_limit": limit,
                "validation_passed": True,
                "database_engine": "postgresql",
                "query_hash": hash(sql.strip().lower())
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        logger.info(f"[SUCCESS] Query returned {len(processed_rows)} rows, {len(columns)} columns")

        return response

    except SQLAlchemyError as e:
        logger.error(f"[ERROR] Database error: {str(e)}")
        return _error_response(sql, f"Database error: {str(e)}", "database_error")

    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        return _error_response(sql, f"Unexpected error: {str(e)}", "unexpected_error")


def _calculate_query_stats(rows: List[Dict[str, Any]], sql: str, execution_time: float) -> Dict[str, Any]:
    """
    Calculate statistics about the query results.

    Args:
        rows (List[Dict[str, Any]]): Query result rows
        sql (str): Original SQL query
        execution_time (float): Query execution time in seconds

    Returns:
        Dict[str, Any]: Query statistics
    """
    try:
        if not rows:
            return {
                "row_count": 0,
                "column_count": 0,
                "execution_time": execution_time,
                "query_type": "select",
                "data_summary": {}
            }

        stats = {
            "row_count": len(rows),
            "column_count": len(rows[0]) if rows else 0,
            "execution_time": execution_time,
            "query_type": _detect_query_type(sql),
            "data_summary": {}
        }

        # Generate data summary for numeric columns
        if rows:
            sample_row = rows[0]
            numeric_summaries = {}

            for column, value in sample_row.items():
                try:
                    # Check if column contains numeric data
                    numeric_values = []
                    for row in rows:
                        val = row.get(column)
                        if isinstance(val, (int, float)) and val is not None:
                            numeric_values.append(val)

                    if len(numeric_values) > 0:
                        numeric_summaries[column] = {
                            "count": len(numeric_values),
                            "min": round(min(numeric_values), 3),
                            "max": round(max(numeric_values), 3),
                            "mean": round(sum(numeric_values) / len(numeric_values), 3),
                            "non_null_percentage": round((len(numeric_values) / len(rows)) * 100, 1)
                        }

                except Exception:
                    continue  # Skip columns that can't be analyzed

            stats["data_summary"]["numeric_columns"] = numeric_summaries

            # Count unique values for key columns
            key_columns = ["basin", "cycle_number", "timestamp"]
            unique_counts = {}
            for column in key_columns:
                if column in sample_row:
                    unique_values = set(row.get(column) for row in rows if row.get(column) is not None)
                    unique_counts[column] = len(unique_values)

            stats["data_summary"]["unique_counts"] = unique_counts

        return stats

    except Exception as e:
        return {
            "error": f"Failed to calculate statistics: {str(e)}",
            "row_count": len(rows) if rows else 0,
            "execution_time": execution_time
        }


def _detect_query_type(sql: str) -> str:
    """
    Detect the type of SQL query.

    Args:
        sql (str): SQL query string

    Returns:
        str: Query type
    """
    sql_lower = sql.lower().strip()

    if 'group by' in sql_lower or 'count(' in sql_lower or 'sum(' in sql_lower or 'avg(' in sql_lower:
        return "aggregate"
    elif 'join' in sql_lower:
        return "join"
    elif 'where' in sql_lower:
        return "filtered_select"
    else:
        return "simple_select"


def _error_response(sql: str, error_message: str, error_type: str) -> Dict[str, Any]:
    """
    Create a standardized error response for SQL queries.

    Args:
        sql (str): Original SQL query
        error_message (str): Error description
        error_type (str): Type of error

    Returns:
        Dict[str, Any]: Error response dictionary
    """
    return {
        "sql": sql,
        "rows": [],
        "columns": [],
        "row_count": 0,
        "column_count": 0,
        "error": error_message,
        "error_type": error_type,
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "suggestions": _get_sql_error_suggestions(error_type)
    }


def _get_sql_error_suggestions(error_type: str) -> List[str]:
    """
    Get suggestions for resolving SQL errors.

    Args:
        error_type (str): Type of error encountered

    Returns:
        List[str]: List of suggestions
    """
    suggestions = {
        "validation_error": [
            "Use only SELECT statements",
            "Remove any DROP, INSERT, UPDATE, or DELETE keywords",
            "Ensure table names are valid (profiles, trajectory, measurements)",
            "Remove SQL comments (-- or /* */)",
            "Check for forbidden keywords in your query"
        ],
        "database_error": [
            "Check column names and table references",
            "Verify data types in WHERE clauses",
            "Ensure proper SQL syntax",
            "Check if the database connection is available"
        ],
        "unexpected_error": [
            "Simplify your query and try again",
            "Check for special characters or formatting issues",
            "Review the query structure",
            "Contact administrator if problem persists"
        ]
    }

    return suggestions.get(error_type, ["Check query syntax and try again"])


def get_schema_info() -> Dict[str, Any]:
    """
    Get information about the available database schema.

    Returns:
        Dict[str, Any]: Schema information
    """
    return {
        "tables": {
            "profiles": {
                "description": "ARGO float profile measurements",
                "primary_table": True,
                "estimated_rows": "~2M profiles",
                "columns": [
                    {"name": "profile_id", "type": "integer", "description": "Unique profile identifier",
                     "indexed": True},
                    {"name": "basin", "type": "integer",
                     "description": "Ocean basin identifier (1=Atlantic, 2=Pacific, 3=Indian)", "indexed": True},
                    {"name": "timestamp", "type": "timestamp", "description": "Measurement timestamp (UTC)",
                     "indexed": True},
                    {"name": "cycle_number", "type": "integer", "description": "Float cycle number", "range": "1-999"},
                    {"name": "vertical_sampling_scheme", "type": "text", "description": "Sampling methodology",
                     "nullable": True},
                    {"name": "longitude", "type": "float", "description": "Longitude coordinate (-180 to 180)",
                     "unit": "degrees", "indexed": True},
                    {"name": "latitude", "type": "float", "description": "Latitude coordinate (-90 to 90)",
                     "unit": "degrees", "indexed": True},
                    {"name": "pressure_decibar", "type": "float", "description": "Pressure measurement",
                     "unit": "decibar", "range": "0-6000"},
                    {"name": "temperature_degc", "type": "float", "description": "Temperature measurement",
                     "unit": "degrees Celsius", "range": "-2 to 35"},
                    {"name": "salinity_psu", "type": "float", "description": "Salinity measurement", "unit": "PSU",
                     "range": "30-40"},
                    {"name": "platform_number", "type": "text", "description": "Float platform identifier"},
                    {"name": "data_mode", "type": "text", "description": "Data processing mode (R/A/D)"},
                    {"name": "position_qc", "type": "integer", "description": "Position quality control flag"}
                ]
            },
            "trajectory": {
                "description": "Float trajectory and surface data",
                "primary_table": False,
                "estimated_rows": "~200K trajectories",
                "columns": [
                    {"name": "platform_number", "type": "text", "description": "Float platform identifier",
                     "indexed": True},
                    {"name": "cycle_number", "type": "integer", "description": "Cycle number"},
                    {"name": "longitude", "type": "float", "description": "Surface longitude", "unit": "degrees"},
                    {"name": "latitude", "type": "float", "description": "Surface latitude", "unit": "degrees"},
                    {"name": "juld", "type": "float", "description": "Julian day"},
                    {"name": "juld_qc", "type": "integer", "description": "Date quality flag"}
                ]
            },
            "measurements": {
                "description": "Individual measurement records",
                "primary_table": False,
                "estimated_rows": "~50M measurements",
                "columns": [
                    {"name": "profile_id", "type": "integer", "description": "Reference to profiles table",
                     "indexed": True},
                    {"name": "depth_meters", "type": "float", "description": "Measurement depth", "unit": "meters"},
                    {"name": "parameter_name", "type": "text", "description": "Measured parameter"},
                    {"name": "parameter_value", "type": "float", "description": "Parameter value"},
                    {"name": "qc_flag", "type": "integer", "description": "Quality control flag"}
                ]
            }
        },
        "basins": {
            "1": {"name": "Atlantic Ocean", "description": "Atlantic basin profiles"},
            "2": {"name": "Pacific Ocean", "description": "Pacific basin profiles"},
            "3": {"name": "Indian Ocean", "description": "Indian basin profiles"},
            "4": {"name": "Southern Ocean", "description": "Southern Ocean profiles"},
            "5": {"name": "Arctic Ocean", "description": "Arctic basin profiles"}
        },
        "sample_queries": {
            "basic_counts": [
                "SELECT COUNT(*) as total_profiles FROM profiles",
                "SELECT basin, COUNT(*) as profile_count FROM profiles GROUP BY basin",
                "SELECT COUNT(DISTINCT platform_number) as unique_floats FROM profiles"
            ],
            "statistical_analysis": [
                "SELECT AVG(temperature_degc) as avg_temp FROM profiles WHERE pressure_decibar BETWEEN 0 AND 100",
                "SELECT basin, AVG(salinity_psu) as avg_salinity FROM profiles GROUP BY basin",
                "SELECT MIN(temperature_degc), MAX(temperature_degc) FROM profiles WHERE basin = 1"
            ],
            "geographic_filtering": [
                "SELECT * FROM profiles WHERE latitude BETWEEN 40 AND 50 AND longitude BETWEEN -10 AND 0 LIMIT 10",
                "SELECT COUNT(*) FROM profiles WHERE latitude > 60",  # Arctic profiles
                "SELECT basin, latitude, longitude FROM profiles WHERE temperature_degc > 25"
            ],
            "temporal_analysis": [
                "SELECT DATE_TRUNC('year', timestamp) as year, COUNT(*) FROM profiles GROUP BY year ORDER BY year",
                "SELECT * FROM profiles WHERE timestamp > '2020-01-01' LIMIT 100",
                "SELECT AVG(temperature_degc) FROM profiles WHERE EXTRACT(month FROM timestamp) IN (6,7,8)"  # Summer
            ],
            "depth_profiles": [
                "SELECT pressure_decibar, AVG(temperature_degc) FROM profiles WHERE pressure_decibar < 1000 GROUP BY pressure_decibar ORDER BY pressure_decibar",
                "SELECT COUNT(*) FROM profiles WHERE pressure_decibar > 2000",  # Deep profiles
                "SELECT platform_number, MAX(pressure_decibar) as max_depth FROM profiles GROUP BY platform_number ORDER BY max_depth DESC LIMIT 10"
            ]
        },
        "query_guidelines": [
            "Use only SELECT statements for data retrieval",
            "Reference valid table names: profiles, trajectory, measurements",
            "Include appropriate WHERE clauses for efficient filtering",
            "Use LIMIT to control result size (automatic limit of 1000 applied)",
            "Aggregate functions supported: COUNT, AVG, SUM, MIN, MAX, STDDEV",
            "Date functions: EXTRACT, DATE_TRUNC are available",
            "Geographic filtering: Use latitude/longitude ranges",
            "Basin filtering: Use basin IN (1,2,3) for multiple basins",
            "Depth filtering: Use pressure_decibar for depth-based queries",
            "Quality filtering: Consider data_mode and qc flags",
            "Join operations: Limited to 5 JOINs maximum",
            "Complex queries: Avoid excessive nesting",
            "Performance: Use indexed columns (basin, timestamp, coordinates) in WHERE clauses"
        ],
        "performance_tips": [
            "Filter by basin first for geographic queries",
            "Use timestamp ranges for temporal filtering",
            "Apply pressure_decibar limits for depth analysis",
            "Consider data_mode = 'A' for adjusted (quality) data",
            "Use LIMIT for exploratory queries",
            "Index available on: basin, timestamp, latitude, longitude, profile_id"
        ]
    }


def print_schema_info():
    """Print formatted schema information to console."""
    schema = get_schema_info()

    print("\n" + "=" * 80)
    print("FLOATCHAT DATABASE SCHEMA")
    print("=" * 80)

    # Tables
    print("\nAVAILABLE TABLES:")
    for table_name, table_info in schema["tables"].items():
        print(f"\n{table_name.upper()}")
        print(f"   Description: {table_info['description']}")
        print(f"   Estimated rows: {table_info.get('estimated_rows', 'N/A')}")
        print(f"   Columns: {len(table_info['columns'])}")

        # Show key columns
        key_cols = [col for col in table_info['columns'][:5]]  # First 5
        for col in key_cols:
            indexed = " (indexed)" if col.get('indexed') else ""
            unit = f" ({col['unit']})" if col.get('unit') else ""
            print(f"     - {col['name']}: {col['type']}{unit}{indexed}")

        if len(table_info['columns']) > 5:
            print(f"     ... and {len(table_info['columns']) - 5} more columns")

    # Ocean basins
    print(f"\nOCEAN BASINS:")
    for basin_id, basin_info in schema["basins"].items():
        print(f"   {basin_id}: {basin_info['name']}")

    # Sample queries by category
    print(f"\nSAMPLE QUERIES:")
    for category, queries in schema["sample_queries"].items():
        category_name = category.replace('_', ' ').title()
        print(f"\n   {category_name}:")
        for i, query in enumerate(queries[:2], 1):  # Show first 2 per category
            print(f"     {i}. {query}")

    print(f"\nPERFORMANCE TIPS:")
    for tip in schema["performance_tips"][:5]:  # Show first 5 tips
        print(f"   * {tip}")

    print(f"\nQUERY GUIDELINES:")
    for guideline in schema["query_guidelines"][:8]:  # Show first 8 guidelines
        print(f"   * {guideline}")

    print("\n" + "=" * 80)


def format_query_results(result: Dict[str, Any], table_format: bool = True) -> str:
    """
    Format query results for console display.

    Args:
        result (Dict[str, Any]): Query result from query_sql_tool
        table_format (bool): Whether to use table formatting

    Returns:
        str: Formatted result string
    """
    if result["status"] != "success":
        return f"Query failed: {result['error']}"

    output = []

    # Header
    output.append("Query executed successfully")
    output.append(f"Results: {result['row_count']} rows, {result['column_count']} columns")
    output.append(f"Execution time: {result['execution_time_seconds']}s")
    output.append("")

    # Results table
    if result["rows"] and table_format:
        try:
            df = pd.DataFrame(result["rows"])
            if len(df) > 0:
                # Truncate display if too many rows
                display_df = df.head(20)
                output.append("RESULTS:")
                output.append(display_df.to_string(index=False, max_cols=10, max_colwidth=20))

                if len(df) > 20:
                    output.append(f"\n... and {len(df) - 20} more rows")
        except Exception:
            # Fallback to simple format
            output.append("RESULTS:")
            for i, row in enumerate(result["rows"][:10]):
                output.append(f"  Row {i + 1}: {row}")
            if len(result["rows"]) > 10:
                output.append(f"  ... and {len(result['rows']) - 10} more rows")

    # Statistics
    if "statistics" in result and "data_summary" in result["statistics"]:
        stats = result["statistics"]["data_summary"]

        if "numeric_columns" in stats:
            numeric_stats = stats["numeric_columns"]
            if numeric_stats:
                output.append("\nNUMERIC COLUMN STATISTICS:")
                for col, col_stats in numeric_stats.items():
                    output.append(f"  {col}: min={col_stats['min']}, max={col_stats['max']}, avg={col_stats['mean']}")

        if "unique_counts" in stats:
            unique_stats = stats["unique_counts"]
            if unique_stats:
                output.append("\nUNIQUE VALUE COUNTS:")
                for col, count in unique_stats.items():
                    output.append(f"  {col}: {count} unique values")

    return "\n".join(output)


def validate_sql_syntax(sql: str) -> Tuple[bool, str]:
    """
    Validate SQL syntax without executing the query.

    Args:
        sql (str): SQL query to validate

    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    validator = SQLValidator()
    return validator.validate_query(sql)


def test_query_sql_tool():
    """
    Test function for the query_sql tool.
    """
    print("\nTESTING QUERY_SQL TOOL")
    print("=" * 50)

    # Test cases
    test_queries = [
        # Should succeed
        ("SELECT COUNT(*) as total_profiles FROM profiles", True),
        ("SELECT basin, COUNT(*) as count FROM profiles GROUP BY basin", True),
        ("SELECT AVG(temperature_degc) as avg_temp FROM profiles WHERE pressure_decibar < 100", True),
        ("SELECT * FROM profiles WHERE basin = 1 LIMIT 5", True),

        # Should fail validation
        ("DROP TABLE profiles", False),
        ("SELECT * FROM invalid_table", False),
        ("INSERT INTO profiles VALUES (1,2,3)", False),
        ("SELECT * FROM profiles; DELETE FROM profiles", False)
    ]

    passed = 0
    failed = 0

    for sql, should_succeed in test_queries:
        print(f"\nTesting: {sql[:60]}...")

        try:
            # Test validation first
            is_valid, msg = validate_sql_syntax(sql)

            if should_succeed and not is_valid:
                print(f"Validation failed unexpectedly: {msg}")
                failed += 1
                continue
            elif not should_succeed and not is_valid:
                print(f"Correctly rejected: {msg}")
                passed += 1
                continue

            # Test actual execution for valid queries
            if should_succeed:
                result = query_sql_tool(sql=sql, limit=5)

                if result["status"] == "success":
                    print(f"Success: {result['row_count']} rows")
                    if "statistics" in result:
                        stats = result["statistics"]
                        print(f"   Type: {stats.get('query_type', 'unknown')}")
                        print(f"   Time: {stats.get('execution_time', 'N/A')}s")
                    passed += 1
                else:
                    print(f"Execution failed: {result.get('error', 'Unknown error')}")
                    failed += 1
            else:
                print(f"Should have been rejected but passed validation")
                failed += 1

        except Exception as e:
            if should_succeed:
                print(f"Exception: {str(e)}")
                failed += 1
            else:
                print(f"Correctly failed with exception")
                passed += 1

    print(f"\nTEST RESULTS: {passed} passed, {failed} failed")
    return passed, failed


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="FloatChat SQL Query Tool")

    parser.add_argument("--sql", "-s", type=str, help="SQL query to execute")
    parser.add_argument("--limit", "-l", type=int, default=1000, help="Row limit (default: 1000)")
    parser.add_argument("--configs", "-c", type=str, default="configs/intel.yaml", help="Config file path")
    parser.add_argument("--schema", action="store_true", help="Show database schema")
    parser.add_argument("--validate", "-v", type=str, help="Validate SQL query without execution")
    parser.add_argument("--test", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive SQL mode")
    parser.add_argument("--examples", action="store_true", help="Show example queries")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        if args.schema:
            print_schema_info()
            return

        if args.examples:
            show_example_queries()
            return

        if args.test:
            passed, failed = test_query_sql_tool()
            sys.exit(0 if failed == 0 else 1)

        if args.validate:
            is_valid, message = validate_sql_syntax(args.validate)
            if is_valid:
                print(f"SQL query is valid: {message}")
            else:
                print(f"SQL query is invalid: {message}")
            return

        if args.interactive:
            run_interactive_mode(args.config)
            return

        if args.sql:
            # Execute single query
            result = query_sql_tool(
                sql=args.sql,
                limit=args.limit,
                config_path=args.config
            )

            if args.format == "json":
                print(json.dumps(result, indent=2, default=str))
            else:
                print(format_query_results(result))
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def show_example_queries():
    """Show categorized example queries."""
    schema = get_schema_info()

    print("\n" + "=" * 80)
    print("FLOATCHAT SQL EXAMPLES")
    print("=" * 80)

    for category, queries in schema["sample_queries"].items():
        category_name = category.replace('_', ' ').title()
        print(f"\n{category_name.upper()}:")
        print("-" * 50)

        for i, query in enumerate(queries, 1):
            print(f"\n{i}. {query}")

    print("\n" + "=" * 80)


def run_interactive_mode(config_path: str):
    """Run interactive SQL query mode."""
    print("\n" + "=" * 80)
    print("FLOATCHAT INTERACTIVE SQL MODE")
    print("=" * 80)
    print("Type your SQL queries below. Commands:")
    print("  \\q or \\quit - Exit")
    print("  \\h or \\help - Show help")
    print("  \\s or \\schema - Show schema")
    print("  \\e or \\examples - Show examples")
    print("  \\c or \\clear - Clear screen")
    print("-" * 80)

    try:
        while True:
            try:
                query = input("\nfloatchat> ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['\\q', '\\quit']:
                    print("Goodbye!")
                    break
                elif query.lower() in ['\\h', '\\help']:
                    print_interactive_help()
                    continue
                elif query.lower() in ['\\s', '\\schema']:
                    print_schema_info()
                    continue
                elif query.lower() in ['\\e', '\\examples']:
                    show_example_queries()
                    continue
                elif query.lower() in ['\\c', '\\clear']:
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                # Execute SQL query
                print(f"\nExecuting: {query}")
                print("-" * 50)

                result = query_sql_tool(
                    sql=query,
                    limit=100,  # Smaller limit for interactive mode
                    config_path=config_path
                )

                print(format_query_results(result))

            except KeyboardInterrupt:
                print("\nUse \\q to quit")
                continue
            except EOFError:
                print("\nGoodbye!")
                break

    except Exception as e:
        print(f"Interactive mode error: {str(e)}")


def print_interactive_help():
    """Print help for interactive mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE HELP")
    print("=" * 60)
    print("\nCommands:")
    print("  \\q, \\quit     - Exit interactive mode")
    print("  \\h, \\help     - Show this help message")
    print("  \\s, \\schema   - Display database schema")
    print("  \\e, \\examples - Show example queries")
    print("  \\c, \\clear    - Clear screen")

    print("\nQuery Tips:")
    print("  * Start with simple COUNT(*) queries")
    print("  * Use LIMIT to control output size")
    print("  * Filter by basin for better performance")
    print("  * Press Ctrl+C to cancel current input")

    print("\nExample Quick Queries:")
    print("  SELECT COUNT(*) FROM profiles;")
    print("  SELECT basin, COUNT(*) FROM profiles GROUP BY basin;")
    print("  SELECT * FROM profiles LIMIT 5;")
    print("-" * 60)


if __name__ == "__main__":
    main()