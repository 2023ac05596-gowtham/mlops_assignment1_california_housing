# Monitoring Guide: Prometheus & Grafana

Real-time monitoring for the California Housing ML API using Prometheus for metrics collection and Grafana for visualization.

## Overview

- **Prometheus**: Scrapes metrics from `/prometheus` endpoint every 15s
- **Grafana**: Displays interactive dashboards at http://localhost:3000
- **ML Metrics**: Model performance, prediction rates, retraining events

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Prometheus    │    │     Grafana     │
│   (Port 8000)   │    │   (Port 9090)   │    │   (Port 3000)   │
│                 │    │                 │    │                 │
│ /prometheus ────┼───►│ Scrapes metrics │    │ Queries metrics │
│ /predict        │    │ every 15s       │    │ & renders       │
│ /predict/batch  │    │                 │    │ dashboards      │
│ /training/*     │    │ Stores time-    │◄───┤                 │
│                 │    │ series data     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- California Housing API project setup complete
- Ports 3000, 8000, 9090 available

### 1. Start Services
```bash
# Start Prometheus, Grafana, and the ML API
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access URLs
- **Grafana**: http://localhost:3000 (credentials: admin/admin123)
- **Prometheus**: http://localhost:9090
- **API**: http://localhost:8000

### 3. Test with Sample Request
```bash
# Generate some predictions to see metrics
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"MedInc": 8.33, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}'

# Generate batch predictions
curl -X POST "http://localhost:8000/predict/batch" \
-H "Content-Type: application/json" \
-d '{"features": [{"MedInc": 8.33, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}]}'
```

## Key Prometheus Queries

```promql
# Total prediction requests
sum(housing_prediction_requests_total)

# Average response time (filtered for successful requests)
rate(housing_api_requests_http_request_duration_seconds_sum{handler=~"/predict.*",status="200"}[5m]) / rate(housing_api_requests_http_request_duration_seconds_count{handler=~"/predict.*",status="200"}[5m]) * 1000

# Model status (1 = loaded)
housing_model_loaded

# Total model predictions
housing_model_predictions_total

# Model retraining events
housing_retraining_triggered_total

# Error rate (4xx/5xx errors per second)
sum(rate(housing_api_http_http_requests_total{handler=~"/predict.*",status=~"4.*|5.*"}[5m]))

# Success rate percentage
(sum(rate(housing_api_http_http_requests_total{handler=~"/predict.*",status="200"}[5m])) / sum(rate(housing_api_http_http_requests_total{handler=~"/predict.*"}[5m]))) * 100
```

## Grafana Dashboard

Pre-configured dashboard with 7 panels:
1. **Total Prediction Requests** - Cumulative API requests to prediction endpoints
2. **Average Response Time** - API latency in milliseconds (successful requests only)
3. **Model Status** - Model loaded status (1=loaded, 0=not loaded)
4. **Total Model Predictions** - Total predictions made by the ML model
5. **Model Retraining Events** - Number of model retrain triggers
6. **Error Rate** - Failed requests per second (4xx/5xx errors)
7. **Success Rate** - Percentage of successful API calls

## Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `housing_prediction_requests_total` | Counter | Total API requests to prediction endpoints |
| `housing_model_predictions_total` | Counter | Total predictions made by ML model |
| `housing_model_loaded` | Gauge | Model load status (1=loaded, 0=not loaded) |
| `housing_retraining_triggered_total` | Counter | Model retraining events triggered |
| `housing_api_requests_http_request_duration_seconds` | Histogram | HTTP request duration for all endpoints |
| `housing_api_http_http_requests_total` | Counter | Total HTTP requests by endpoint and status |

## Troubleshooting

### **Grafana Shows "No Data"**
```bash
# Check metrics are exposed
curl http://localhost:8000/prometheus

# Test with a prediction
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"MedInc": 8.0, "HouseAge": 30.0, "AveRooms": 7.0, "AveBedrms": 1.1, "Population": 4500.0, "AveOccup": 3.8, "Latitude": 38.0, "Longitude": -122.0}'
```

### Services Not Running
```bash
# Check all services status
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs

# View individual service logs
docker-compose -f docker-compose.monitoring.yml logs -f grafana
docker-compose -f docker-compose.monitoring.yml logs -f prometheus  
docker-compose -f docker-compose.monitoring.yml logs -f housing-api

# Test individual components
curl http://localhost:9090/-/healthy    # Prometheus health
curl http://localhost:3000/api/health   # Grafana health  
curl http://localhost:8000/health       # API health

# Restart
docker-compose -f docker-compose.monitoring.yml restart
```