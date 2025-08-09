#!/usr/bin/env python3
"""
Prometheus metrics collection for California Housing API
Custom metrics for monitoring model performance and API usage
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from fastapi import FastAPI
import time
from datetime import datetime
from typing import Dict, Any

# Essential Custom metrics for bonus requirements
prediction_requests_total = Counter(
    "housing_prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)

prediction_duration = Histogram(
    "housing_prediction_duration_seconds",
    "Duration of prediction requests",
    ["endpoint"],
)

model_predictions_total = Counter(
    "housing_model_predictions_total", "Total predictions made by model"
)

model_load_status = Gauge(
    "housing_model_loaded",
    "Whether the ML model is successfully loaded (1 = loaded, 0 = not loaded)",
)

api_errors_total = Counter(
    "housing_api_errors_total",
    "Total number of API errors",
    ["endpoint", "status_code"],
)

# Retraining metrics
retraining_trigger_total = Counter(
    "housing_retraining_triggered_total",
    "Number of times model retraining was triggered",
)

new_data_points = Counter(
    "housing_new_data_points_total", "Total number of new training data points received"
)


class PrometheusMetricsCollector:
    """Simplified metrics collection for Prometheus"""

    def __init__(self):
        self.start_time = time.time()

    def track_prediction_request(self, endpoint: str, status: str = "success"):
        """Track prediction requests"""
        prediction_requests_total.labels(endpoint=endpoint, status=status).inc()

    def track_prediction_duration(self, endpoint: str, duration: float):
        """Track prediction duration"""
        prediction_duration.labels(endpoint=endpoint).observe(duration)

    def track_model_prediction(self):
        """Track model predictions"""
        model_predictions_total.inc()

    def track_api_error(self, endpoint: str, status_code: int):
        """Track API errors"""
        api_errors_total.labels(endpoint=endpoint, status_code=str(status_code)).inc()

    def set_model_status(self, loaded: bool):
        """Set model load status"""
        model_load_status.set(1 if loaded else 0)

    def track_retraining_trigger(self):
        """Track retraining triggers"""
        retraining_trigger_total.inc()

    def track_new_data_points(self, count: int = 1):
        """Track new training data points"""
        new_data_points.inc(count)


# Global metrics collector instance
prometheus_collector = PrometheusMetricsCollector()


def setup_prometheus_monitoring(app: FastAPI) -> Instrumentator:
    """
    Set up Prometheus monitoring for FastAPI application
    """
    # Create instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/docs", "/openapi.json", "/redoc"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="housing_api_requests_inprogress",
        inprogress_labels=True,
    )

    # Add default metrics
    instrumentator.add(
        metrics.request_size_and_response_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="housing_api",
            metric_subsystem="requests",
        )
    )

    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="housing_api",
            metric_subsystem="requests",
        )
    )

    instrumentator.add(
        metrics.requests(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="housing_api",
            metric_subsystem="http",
        )
    )

    # Instrument the app
    instrumentator.instrument(app)

    return instrumentator


def get_prometheus_collector() -> PrometheusMetricsCollector:
    """Get the global Prometheus metrics collector"""
    return prometheus_collector
