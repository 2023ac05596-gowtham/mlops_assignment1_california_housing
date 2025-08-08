#!/usr/bin/env python3
"""
Database operations for California Housing Price Prediction API
Handles SQLite database connections, table creation, and logging operations
"""

import os
import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

# Setup logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for API logging and metrics"""

    def __init__(self, db_path: str):
        """Initialize database manager with database path"""
        self.db_path = db_path
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_tables(self):
        """Initialize SQLite database tables for logging"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create predictions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    confidence REAL,
                    response_time REAL,
                    model_used TEXT,
                    endpoint TEXT,
                    error_message TEXT
                )
            """
            )

            # Create API metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER,
                    response_time REAL,
                    error_message TEXT
                )
            """
            )

            conn.commit()
            logger.info("Database tables initialized successfully")

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: float,
        confidence: float,
        response_time: float,
        model_used: str,
        endpoint: str,
        error_message: Optional[str] = None,
    ):
        """Log prediction data to SQLite database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions 
                    (timestamp, input_data, prediction, confidence, 
                    response_time,
                     model_used, endpoint, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now().isoformat(),
                        str(input_data),
                        prediction,
                        confidence,
                        response_time,
                        model_used,
                        endpoint,
                        error_message,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging prediction to database: {str(e)}")

    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        error_message: Optional[str] = None,
    ):
        """Log API request metrics to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO api_metrics
                    (
                        timestamp,
                        endpoint,
                        method,
                        status_code,
                        response_time,
                        error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now().isoformat(),
                        endpoint,
                        method,
                        status_code,
                        response_time,
                        error_message,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging API request to database: {str(e)}")

    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent successful predictions from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT timestamp, prediction, confidence, response_time
                    FROM predictions
                    WHERE error_message IS NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                return [
                    {
                        "timestamp": row[0],
                        "prediction": row[1],
                        "confidence": row[2],
                        "response_time": row[3],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting recent predictions: {str(e)}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Simple counts
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total_predictions = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM api_metrics")
                total_requests = cursor.fetchone()[0]

                return {
                    "total_predictions": total_predictions,
                    "total_requests": total_requests,
                    "database_path": self.db_path,
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}

    def health_check(self) -> bool:
        """Check if database is accessible and working"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False


# Global database manager instance (will be initialized in api.py)
db_manager: Optional[DatabaseManager] = None


def initialize_database(db_path: str) -> DatabaseManager:
    """Initialize global database manager"""
    global db_manager
    db_manager = DatabaseManager(db_path)
    db_manager.initialize_tables()
    return db_manager


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    if db_manager is None:
        raise RuntimeError(
            (
                "Database manager not initialized. "
                "Call initialize_database() first."
            )
        )
    return db_manager
