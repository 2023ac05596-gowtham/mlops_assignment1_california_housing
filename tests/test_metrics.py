#!/usr/bin/env python3
"""
Unit tests for metrics module
"""

from pathlib import Path
import sys
import time
from datetime import datetime
from unittest.mock import patch
from metrics import MetricsTracker, RequestTimer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMetricsTracker:
    """Test cases for MetricsTracker class"""

    def test_initialization(self):
        """Test metrics tracker initialization"""
        tracker = MetricsTracker()

        assert tracker.metrics["total_requests"] == 0
        assert tracker.metrics["total_predictions"] == 0
        assert tracker.metrics["total_errors"] == 0
        assert tracker.metrics["response_times"] == []
        assert isinstance(tracker.metrics["start_time"], datetime)

    def test_track_request(self):
        """Test request tracking"""
        tracker = MetricsTracker()

        tracker.track_request()
        assert tracker.metrics["total_requests"] == 1

        tracker.track_request()
        assert tracker.metrics["total_requests"] == 2

    def test_track_prediction(self):
        """Test prediction tracking"""
        tracker = MetricsTracker()

        tracker.track_prediction()
        assert tracker.metrics["total_predictions"] == 1

        tracker.track_prediction(count=5)
        assert tracker.metrics["total_predictions"] == 6

    def test_track_response_time(self):
        """Test response time tracking"""
        tracker = MetricsTracker()

        tracker.track_response_time(0.5)
        tracker.track_response_time(1.0)

        assert len(tracker.metrics["response_times"]) == 2
        assert 0.5 in tracker.metrics["response_times"]
        assert 1.0 in tracker.metrics["response_times"]

    def test_track_response_time_max_limit(self):
        """Test response time tracking with maximum limit"""
        tracker = MetricsTracker(max_response_times=3)

        # Add more response times than limit
        for i in range(5):
            tracker.track_response_time(float(i))

        # Should only keep the last 3
        assert len(tracker.metrics["response_times"]) == 3
        assert tracker.metrics["response_times"] == [2.0, 3.0, 4.0]

    def test_track_error(self):
        """Test error tracking"""
        tracker = MetricsTracker()

        tracker.track_error("ValidationError")
        tracker.track_error("TimeoutError")
        tracker.track_error("ValidationError")

        assert tracker.metrics["total_errors"] == 3
        assert tracker.metrics["error_types"]["ValidationError"] == 2
        assert tracker.metrics["error_types"]["TimeoutError"] == 1

    def test_get_performance_metrics_empty(self):
        """Test performance metrics with no data"""
        tracker = MetricsTracker()

        metrics = tracker.get_performance_metrics()

        assert metrics["average_response_time_seconds"] == 0
        assert metrics["total_requests_timed"] == 0

    def test_get_performance_metrics_with_data(self):
        """Test performance metrics with data"""
        tracker = MetricsTracker()

        tracker.track_response_time(0.5)
        tracker.track_response_time(1.0)
        tracker.track_response_time(1.5)

        metrics = tracker.get_performance_metrics()

        assert metrics["average_response_time_seconds"] == 1.0
        assert metrics["total_requests_timed"] == 3

    def test_get_api_statistics(self):
        """Test API statistics calculation"""
        tracker = MetricsTracker()

        # Track some requests and errors
        tracker.track_request()
        tracker.track_request()
        tracker.track_request()
        tracker.track_error("TestError")
        tracker.track_prediction(2)

        stats = tracker.get_api_statistics()

        assert stats["total_requests"] == 3
        assert stats["total_predictions"] == 2
        assert stats["total_errors"] == 1
        # Success rate: (3-1)/3 * 100 = 66.67%
        assert abs(stats["success_rate_percent"] - 66.67) < 0.01
        assert "uptime_hours" in stats

    def test_get_api_statistics_no_requests(self):
        """Test API statistics with no requests"""
        tracker = MetricsTracker()

        stats = tracker.get_api_statistics()

        assert stats["total_requests"] == 0
        assert stats["success_rate_percent"] == 0

    def test_reset_metrics(self):
        """Test metrics reset functionality"""
        tracker = MetricsTracker()

        # Add some data
        tracker.track_request()
        tracker.track_error("TestError")
        tracker.track_response_time(0.5)

        # Reset
        tracker.reset_metrics()

        # Verify reset
        assert tracker.metrics["total_requests"] == 0
        assert tracker.metrics["total_errors"] == 0
        assert tracker.metrics["response_times"] == []
        assert len(tracker.metrics["error_types"]) == 0


class TestRequestTimer:
    """Test cases for RequestTimer context manager"""

    def test_request_timer_context_manager(self):
        """Test RequestTimer as context manager"""
        tracker = MetricsTracker()

        with RequestTimer(tracker):
            time.sleep(0.1)  # Simulate some work

        # Should have tracked the response time
        assert len(tracker.metrics["response_times"]) == 1
        assert tracker.metrics["response_times"][0] >= 0.1

    def test_request_timer_manual_usage(self):
        """Test RequestTimer manual start/stop"""
        tracker = MetricsTracker()
        timer = RequestTimer(tracker)

        timer.start()
        time.sleep(0.05)
        timer.stop()

        assert len(tracker.metrics["response_times"]) == 1
        assert tracker.metrics["response_times"][0] >= 0.05

    def test_request_timer_without_tracker(self):
        """Test RequestTimer without tracker (should not fail)"""
        timer = RequestTimer(None)

        timer.start()
        time.sleep(0.01)
        timer.stop()

        # Should not raise any errors

    @patch('metrics.time.time')
    def test_request_timer_time_calculation(self, mock_time):
        """Test RequestTimer time calculation"""
        tracker = MetricsTracker()

        # Mock time progression
        mock_time.side_effect = [1.0, 1.5]  # 0.5 second difference

        timer = RequestTimer(tracker)
        timer.start()
        timer.stop()

        assert len(tracker.metrics["response_times"]) == 1
        assert tracker.metrics["response_times"][0] == 0.5
