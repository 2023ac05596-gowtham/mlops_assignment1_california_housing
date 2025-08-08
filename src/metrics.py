#!/usr/bin/env python3
"""
Metrics tracking and monitoring for California Housing Price Prediction API
Handles in-memory metrics, performance calculations, and monitoring data aggregation
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

# Setup logger
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks and manages API performance metrics"""

    def __init__(self, max_response_times: int = 1000):
        """
        Initialize metrics tracker

        Args:
            max_response_times: Maximum number of response times to keep in memory
        """
        self.max_response_times = max_response_times
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics to initial state"""
        self.metrics = {
            "total_requests": 0,
            "total_predictions": 0,
            "total_errors": 0,
            "response_times": [],
            "error_types": defaultdict(int),
            "start_time": datetime.now(),
        }
        logger.info("Metrics tracker initialized/reset")

    def track_request(self):
        """Track a new incoming request"""
        self.metrics["total_requests"] += 1

    def track_prediction(self, count: int = 1):
        """Track successful predictions"""
        self.metrics["total_predictions"] += count

    def track_response_time(self, response_time: float):
        """Track response time for performance monitoring"""
        self.metrics["response_times"].append(response_time)

        # Keep only the last N response times to prevent memory issues
        if len(self.metrics["response_times"]) > self.max_response_times:
            self.metrics["response_times"] = self.metrics["response_times"][
                -self.max_response_times :
            ]

    def track_error(self, error_type: str):
        """Track API errors by type"""
        self.metrics["total_errors"] += 1
        self.metrics["error_types"][error_type] += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        response_times = self.metrics["response_times"]

        if not response_times:
            return {"average_response_time_seconds": 0, "total_requests_timed": 0}

        return {
            "average_response_time_seconds": round(
                sum(response_times) / len(response_times), 4
            ),
            "total_requests_timed": len(response_times),
        }

    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        total_requests = self.metrics["total_requests"]
        total_errors = self.metrics["total_errors"]

        success_rate = 0
        if total_requests > 0:
            success_rate = ((total_requests - total_errors) / total_requests) * 100

        return {
            "total_requests": total_requests,
            "total_predictions": self.metrics["total_predictions"],
            "total_errors": total_errors,
            "success_rate_percent": round(success_rate, 2),
            "uptime_hours": round(
                (datetime.now() - self.metrics["start_time"]).total_seconds() / 3600, 2
            ),
        }

    def get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by type"""
        return dict(self.metrics["error_types"])

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics for error scenarios"""
        return {
            "total_requests": self.metrics["total_requests"],
            "total_predictions": self.metrics["total_predictions"],
            "total_errors": self.metrics["total_errors"],
            "uptime_hours": round(
                (datetime.now() - self.metrics["start_time"]).total_seconds() / 3600, 2
            ),
        }


class ModelMetrics:
    """Simple model tracking"""

    def __init__(self):
        """Initialize model metrics tracker"""
        self.prediction_count = 0

    def track_model_prediction(
        self, model_name: str, prediction: float, confidence: float
    ):
        """Track a prediction made by a model"""
        self.prediction_count += 1

    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get basic model statistics"""
        return {
            "model_name": model_name,
            "prediction_count": self.prediction_count,
        }


# Global metrics tracker instances
metrics_tracker: Optional[MetricsTracker] = None
model_metrics: Optional[ModelMetrics] = None


def initialize_metrics() -> tuple[MetricsTracker, ModelMetrics]:
    """Initialize global metrics trackers"""
    global metrics_tracker, model_metrics
    metrics_tracker = MetricsTracker()
    model_metrics = ModelMetrics()
    logger.info("Metrics trackers initialized successfully")
    return metrics_tracker, model_metrics


def get_metrics_tracker() -> MetricsTracker:
    """Get the global metrics tracker instance"""
    if metrics_tracker is None:
        raise RuntimeError(
            "Metrics tracker not initialized. Call initialize_metrics() first."
        )
    return metrics_tracker


def get_model_metrics() -> ModelMetrics:
    """Get the global model metrics instance"""
    if model_metrics is None:
        raise RuntimeError(
            "Model metrics not initialized. Call initialize_metrics() first."
        )
    return model_metrics


class RequestTimer:
    """Context manager for timing requests and automatically tracking metrics"""

    def __init__(self, endpoint: str = "unknown"):
        self.endpoint = endpoint
        self.start_time = None
        self.response_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.response_time = time.time() - self.start_time
            if metrics_tracker:
                metrics_tracker.track_response_time(self.response_time)

    def get_response_time(self) -> float:
        """Get the measured response time"""
        return self.response_time or 0.0
