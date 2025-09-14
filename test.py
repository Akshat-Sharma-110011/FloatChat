"""
Database Connection and Insertion Test Script for ARGO Data Ingestion

This script performs comprehensive testing of the PostgreSQL database connection,
table creation, data insertion, and retrieval to diagnose any issues with the
ARGO data ingestion process.

Author: FloatChat Team
Version: 1.0.0
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd

# Database URL (same as in the main script)
DATABASE_URL = "postgresql://postgres:Strong.password177013@localhost:6000/floatchat"

def setup_test_logging():
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'database_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_test_logging()

def test_basic_connection():
    """Test basic database connectivity."""
    logger.info("="*60)
    logger.info("TEST 1: Basic Database Connection")
    logger.info("="*60)

    try:
        logger.info(f"Attempting to connect to database...")
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("[SUCCESS] Successfully connected to PostgreSQL database")

        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"[SUCCESS] PostgreSQL version: {version[0]}")

        cursor.execute("SELECT current_database();")
        db_name = cursor.fetchone()
        logger.info(f"[SUCCESS] Connected to database: {db_name[0]}")

        cursor.close()
        conn.close()
        logger.info("[SUCCESS] Connection closed successfully")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Database connection failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_table_creation():
    """Test table creation and schema verification."""
    logger.info("="*60)
    logger.info("TEST 2: Table Creation and Schema")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Drop table if exists (for testing purposes)
        logger.info("Dropping existing test table if it exists...")
        cursor.execute("DROP TABLE IF EXISTS profiles_test CASCADE;")
        conn.commit()
        logger.info("[SUCCESS] Test table dropped (if existed)")

        # Create test table
        create_table_sql = """
        CREATE TABLE profiles_test (
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
        """

        logger.info("Creating test table...")
        cursor.execute(create_table_sql)
        conn.commit()
        logger.info("[SUCCESS] Test table created successfully")

        # Verify table structure
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'profiles_test' 
            ORDER BY ordinal_position;
        """)

        columns = cursor.fetchall()
        logger.info("[SUCCESS] Table schema verification:")
        for col in columns:
            logger.info(f"   - {col[0]}: {col[1]} (nullable: {col[2]})")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Table creation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_single_record_insertion():
    """Test insertion of a single record."""
    logger.info("="*60)
    logger.info("TEST 3: Single Record Insertion")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Clear test table
        cursor.execute("DELETE FROM profiles_test;")
        conn.commit()
        logger.info("[SUCCESS] Test table cleared")

        # Insert single test record
        test_record = (
            1,  # basin
            '2025-01-15T10:30:00+00:00',  # timestamp
            101,  # cycle_number
            'Primary sampling',  # vertical_sampling_scheme
            -45.123456,  # longitude
            23.654321,  # latitude
            100.50,  # pressure_decibar
            35.123456,  # salinity_psu
            15.789012  # temperature_degc
        )

        insert_sql = """
        INSERT INTO profiles_test (
            basin, timestamp, cycle_number, vertical_sampling_scheme,
            longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """

        logger.info(f"Inserting test record: {test_record}")
        cursor.execute(insert_sql, test_record)
        record_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"[SUCCESS] Single record inserted successfully with ID: {record_id}")

        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM profiles_test;")
        count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Table now contains {count} record(s)")

        # Retrieve and display the record
        cursor.execute("SELECT * FROM profiles_test WHERE id = %s;", (record_id,))
        retrieved_record = cursor.fetchone()
        logger.info(f"[SUCCESS] Retrieved record: {retrieved_record}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Single record insertion failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_batch_insertion():
    """Test batch insertion using execute_values."""
    logger.info("="*60)
    logger.info("TEST 4: Batch Record Insertion")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Clear test table
        cursor.execute("DELETE FROM profiles_test;")
        conn.commit()
        logger.info("[SUCCESS] Test table cleared")

        # Create test batch data
        test_batch = [
            (1, '2025-01-15T10:30:00+00:00', 101, 'Primary sampling', -45.123456, 23.654321, 100.50, 35.123456, 15.789012),
            (2, '2025-01-15T11:30:00+00:00', 102, 'Secondary sampling', -46.123456, 24.654321, 200.75, 35.223456, 16.789012),
            (1, '2025-01-15T12:30:00+00:00', 103, 'Tertiary sampling', -47.123456, 25.654321, 300.25, 35.323456, 17.789012),
            (2, '2025-01-15T13:30:00+00:00', 104, '', -48.123456, 26.654321, 400.00, 35.423456, 18.789012),
            (1, '2025-01-15T14:30:00+00:00', 105, 'Final sampling', -49.123456, 27.654321, 500.30, 35.523456, 19.789012)
        ]

        logger.info(f"Preparing to insert batch of {len(test_batch)} records")

        insert_sql = """
        INSERT INTO profiles_test (
            basin, timestamp, cycle_number, vertical_sampling_scheme,
            longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
        ) VALUES %s
        """

        logger.info("Executing batch insertion using execute_values...")
        execute_values(
            cursor,
            insert_sql,
            test_batch,
            template=None,
            page_size=1000
        )

        conn.commit()
        logger.info("[SUCCESS] Batch insertion completed successfully")

        # Verify batch insertion
        cursor.execute("SELECT COUNT(*) FROM profiles_test;")
        count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Table now contains {count} record(s)")

        # Display sample records
        cursor.execute("SELECT * FROM profiles_test ORDER BY id LIMIT 3;")
        sample_records = cursor.fetchall()
        logger.info("[SUCCESS] Sample records from batch:")
        for i, record in enumerate(sample_records, 1):
            logger.info(f"   Record {i}: {record}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Batch insertion failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_transaction_handling():
    """Test transaction handling and rollback scenarios."""
    logger.info("="*60)
    logger.info("TEST 5: Transaction Handling")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Clear test table
        cursor.execute("DELETE FROM profiles_test;")
        conn.commit()
        logger.info("[SUCCESS] Test table cleared")

        # Test successful transaction
        logger.info("Testing successful transaction...")
        cursor.execute("BEGIN;")
        cursor.execute("""
            INSERT INTO profiles_test (
                basin, timestamp, cycle_number, vertical_sampling_scheme,
                longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
            ) VALUES (1, '2025-01-15T10:30:00+00:00', 101, 'Test', -45.123, 23.654, 100.5, 35.123, 15.789);
        """)
        cursor.execute("COMMIT;")
        logger.info("[SUCCESS] Successful transaction completed")

        # Verify record exists
        cursor.execute("SELECT COUNT(*) FROM profiles_test;")
        count_after_commit = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Records after commit: {count_after_commit}")

        # Test rollback scenario
        logger.info("Testing rollback scenario...")
        cursor.execute("BEGIN;")
        cursor.execute("""
            INSERT INTO profiles_test (
                basin, timestamp, cycle_number, vertical_sampling_scheme,
                longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
            ) VALUES (2, '2025-01-15T11:30:00+00:00', 102, 'Test2', -46.123, 24.654, 200.5, 36.123, 16.789);
        """)
        cursor.execute("ROLLBACK;")
        logger.info("[SUCCESS] Rollback completed")

        # Verify rollback worked
        cursor.execute("SELECT COUNT(*) FROM profiles_test;")
        count_after_rollback = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Records after rollback: {count_after_rollback}")

        if count_after_rollback == count_after_commit:
            logger.info("[SUCCESS] Transaction handling working correctly")
            result = True
        else:
            logger.error("[ERROR] Transaction rollback failed")
            result = False

        cursor.close()
        conn.close()
        return result

    except Exception as e:
        logger.error(f"[ERROR] Transaction handling test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_real_table_interaction():
    """Test interaction with the actual profiles table."""
    logger.info("="*60)
    logger.info("TEST 6: Real Profiles Table Interaction")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Check if profiles table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'profiles'
            );
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            logger.warning("[WARNING] Profiles table does not exist, creating it...")

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
            """

            cursor.execute(create_profiles_table)
            conn.commit()
            logger.info("[SUCCESS] Profiles table created")
        else:
            logger.info("[SUCCESS] Profiles table already exists")

        # Get current record count
        cursor.execute("SELECT COUNT(*) FROM profiles;")
        current_count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Current profiles table record count: {current_count}")

        # Test insertion into real table
        test_record = (
            99,  # basin
            '2025-01-15T15:30:00+00:00',  # timestamp
            999,  # cycle_number
            'Test insertion',  # vertical_sampling_scheme
            -99.123456,  # longitude
            99.654321,  # latitude
            999.99,  # pressure_decibar
            99.999999,  # salinity_psu
            99.999999  # temperature_degc
        )

        logger.info("Testing insertion into real profiles table...")
        insert_sql = """
        INSERT INTO profiles (
            basin, timestamp, cycle_number, vertical_sampling_scheme,
            longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """

        cursor.execute(insert_sql, test_record)
        record_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"[SUCCESS] Test record inserted into profiles table with ID: {record_id}")

        # Verify the new count
        cursor.execute("SELECT COUNT(*) FROM profiles;")
        new_count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Profiles table now contains {new_count} records (increased by {new_count - current_count})")

        # Clean up test record
        cursor.execute("DELETE FROM profiles WHERE id = %s;", (record_id,))
        conn.commit()
        logger.info(f"[SUCCESS] Test record cleaned up from profiles table")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Real table interaction test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_data_types_and_constraints():
    """Test various data types and constraint handling."""
    logger.info("="*60)
    logger.info("TEST 7: Data Types and Constraints")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Clear test table
        cursor.execute("DELETE FROM profiles_test;")
        conn.commit()

        # Test various data scenarios
        test_cases = [
            {
                "name": "Normal data",
                "data": (1, '2025-01-15T10:30:00+00:00', 101, 'Normal', -45.123456, 23.654321, 100.50, 35.123456, 15.789012),
                "should_succeed": True
            },
            {
                "name": "NULL salinity and temperature",
                "data": (1, '2025-01-15T10:30:00+00:00', 102, 'NULL values', -45.123456, 23.654321, 100.50, None, None),
                "should_succeed": True
            },
            {
                "name": "Empty vertical_sampling_scheme",
                "data": (1, '2025-01-15T10:30:00+00:00', 103, '', -45.123456, 23.654321, 100.50, 35.123456, 15.789012),
                "should_succeed": True
            },
            {
                "name": "NULL vertical_sampling_scheme",
                "data": (1, '2025-01-15T10:30:00+00:00', 104, None, -45.123456, 23.654321, 100.50, 35.123456, 15.789012),
                "should_succeed": True
            },
            {
                "name": "Extreme longitude/latitude",
                "data": (1, '2025-01-15T10:30:00+00:00', 105, 'Extreme', -180.0, 90.0, 100.50, 35.123456, 15.789012),
                "should_succeed": True
            }
        ]

        insert_sql = """
        INSERT INTO profiles_test (
            basin, timestamp, cycle_number, vertical_sampling_scheme,
            longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        successful_inserts = 0

        for test_case in test_cases:
            try:
                logger.info(f"Testing: {test_case['name']}")
                cursor.execute(insert_sql, test_case['data'])
                conn.commit()

                if test_case['should_succeed']:
                    logger.info(f"[SUCCESS] {test_case['name']}: Insertion succeeded as expected")
                    successful_inserts += 1
                else:
                    logger.warning(f"[WARNING] {test_case['name']}: Insertion succeeded but was expected to fail")

            except Exception as e:
                conn.rollback()  # Rollback failed transaction

                if not test_case['should_succeed']:
                    logger.info(f"[SUCCESS] {test_case['name']}: Insertion failed as expected - {e}")
                else:
                    logger.error(f"[ERROR] {test_case['name']}: Insertion failed unexpectedly - {e}")

        # Verify final count
        cursor.execute("SELECT COUNT(*) FROM profiles_test;")
        final_count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Successfully inserted {successful_inserts} test records, final count: {final_count}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Data types and constraints test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_performance_and_memory():
    """Test performance with larger datasets."""
    logger.info("="*60)
    logger.info("TEST 8: Performance and Memory Usage")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Clear test table
        cursor.execute("DELETE FROM profiles_test;")
        conn.commit()
        logger.info("[SUCCESS] Test table cleared")

        # Generate larger test dataset
        batch_size = 1000
        logger.info(f"Generating batch of {batch_size} records...")

        large_batch = []
        for i in range(batch_size):
            record = (
                (i % 5) + 1,  # basin (1-5)
                f'2025-01-15T{10 + (i % 14):02d}:30:00+00:00',  # timestamp
                1000 + i,  # cycle_number
                f'Batch test record {i}',  # vertical_sampling_scheme
                -180.0 + (i * 0.001) % 360.0,  # longitude
                -90.0 + (i * 0.001) % 180.0,  # latitude
                10.0 + (i * 0.1) % 1000.0,  # pressure_decibar
                30.0 + (i * 0.01) % 10.0,  # salinity_psu
                10.0 + (i * 0.01) % 20.0   # temperature_degc
            )
            large_batch.append(record)

        logger.info(f"[SUCCESS] Generated {len(large_batch)} test records")

        # Time the insertion
        start_time = datetime.now()

        insert_sql = """
        INSERT INTO profiles_test (
            basin, timestamp, cycle_number, vertical_sampling_scheme,
            longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
        ) VALUES %s
        """

        logger.info(f"Starting batch insertion of {batch_size} records...")
        execute_values(
            cursor,
            insert_sql,
            large_batch,
            template=None,
            page_size=min(batch_size, 1000)
        )

        conn.commit()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"[SUCCESS] Batch insertion completed in {duration:.2f} seconds")
        logger.info(f"[SUCCESS] Performance: {batch_size / duration:.2f} records/second")

        # Verify count
        cursor.execute("SELECT COUNT(*) FROM profiles_test;")
        count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Final record count: {count}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Performance test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def cleanup_test_data():
    """Clean up test data and tables."""
    logger.info("="*60)
    logger.info("CLEANUP: Removing Test Data")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Drop test table
        cursor.execute("DROP TABLE IF EXISTS profiles_test CASCADE;")
        conn.commit()
        logger.info("[SUCCESS] Test table dropped successfully")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Cleanup failed: {e}")
        return False

def diagnose_ingestion_issue():
    """Specific diagnosis for the ingestion issue."""
    logger.info("="*60)
    logger.info("DIAGNOSIS: Analyzing Ingestion Issues")
    logger.info("="*60)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Check if profiles table has any constraints that might cause issues
        logger.info("Checking table constraints...")
        cursor.execute("""
            SELECT conname, contype, pg_get_constraintdef(oid) as definition
            FROM pg_constraint 
            WHERE conrelid = 'profiles'::regclass;
        """)

        constraints = cursor.fetchall()
        if constraints:
            logger.info("[SUCCESS] Table constraints found:")
            for constraint in constraints:
                logger.info(f"   - {constraint[0]} ({constraint[1]}): {constraint[2]}")
        else:
            logger.info("[INFO] No custom constraints found on profiles table")

        # Check for any triggers
        logger.info("Checking for triggers...")
        cursor.execute("""
            SELECT trigger_name, event_manipulation, action_statement
            FROM information_schema.triggers 
            WHERE event_object_table = 'profiles';
        """)

        triggers = cursor.fetchall()
        if triggers:
            logger.warning("[WARNING] Triggers found on profiles table:")
            for trigger in triggers:
                logger.warning(f"   - {trigger[0]}: {trigger[1]} -> {trigger[2]}")
        else:
            logger.info("[SUCCESS] No triggers found on profiles table")

        # Check current autocommit and transaction state
        logger.info(f"Connection autocommit: {conn.autocommit}")
        logger.info(f"Connection status: {conn.status}")

        # Test a simple insert with explicit transaction handling
        logger.info("Testing explicit transaction handling...")

        # Get current count
        cursor.execute("SELECT COUNT(*) FROM profiles;")
        before_count = cursor.fetchone()[0]
        logger.info(f"Records before test insert: {before_count}")

        # Explicit transaction
        cursor.execute("BEGIN;")
        logger.info("[SUCCESS] Transaction started")

        cursor.execute("""
            INSERT INTO profiles (
                basin, timestamp, cycle_number, vertical_sampling_scheme,
                longitude, latitude, pressure_decibar, salinity_psu, temperature_degc
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (999, '2025-01-15T15:30:00+00:00', 9999, 'Diagnosis test', -99.123, 99.123, 999.99, 99.99, 99.99))

        inserted_id = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Record inserted with ID: {inserted_id}")

        cursor.execute("COMMIT;")
        logger.info("[SUCCESS] Transaction committed")

        # Verify the insert
        cursor.execute("SELECT COUNT(*) FROM profiles;")
        after_count = cursor.fetchone()[0]
        logger.info(f"Records after test insert: {after_count}")

        if after_count > before_count:
            logger.info("[SUCCESS] Manual insertion working correctly")

            # Clean up test record
            cursor.execute("DELETE FROM profiles WHERE id = %s;", (inserted_id,))
            conn.commit()
            logger.info("[SUCCESS] Test record cleaned up")
        else:
            logger.error("[ERROR] Manual insertion not persisting!")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"[ERROR] Diagnosis failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all database tests."""
    logger.info("[START] Starting Database Connection and Insertion Tests")
    logger.info(f"Database URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'masked'}")
    logger.info(f"Test started at: {datetime.now()}")

    test_results = {}

    # Run all tests
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Table Creation", test_table_creation),
        ("Single Record Insertion", test_single_record_insertion),
        ("Batch Insertion", test_batch_insertion),
        ("Transaction Handling", test_transaction_handling),
        ("Real Table Interaction", test_real_table_interaction),
        ("Data Types and Constraints", test_data_types_and_constraints),
        ("Performance Test", test_performance_and_memory),
        ("Ingestion Issue Diagnosis", diagnose_ingestion_issue)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n[TEST] Running {test_name}...")
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
                logger.info(f"[SUCCESS] {test_name}: PASSED")
            else:
                logger.error(f"[ERROR] {test_name}: FAILED")
        except Exception as e:
            test_results[test_name] = False
            logger.error(f"[ERROR] {test_name}: CRASHED - {e}")

    # Cleanup
    logger.info(f"\n[CLEANUP] Running cleanup...")
    cleanup_test_data()

    # Summary
    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    logger.info("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "[SUCCESS] PASS" if result else "[ERROR] FAIL"
        logger.info(f"  {test_name}: {status}")

    if passed_tests == total_tests:
        logger.info("\n[SUCCESS] All tests passed! Database connectivity and operations are working correctly.")
        logger.info("If your ingestion is still not working, the issue might be in the application logic.")
    else:
        logger.warning(f"\n[WARNING] {total_tests - passed_tests} test(s) failed. Check the logs above for details.")

    logger.info(f"\nTest completed at: {datetime.now()}")

if __name__ == "__main__":
    main()