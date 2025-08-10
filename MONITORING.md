# 📊 Monitoring Guide: Prometheus & Grafana

This guide provides comprehensive instructions for setting up and using **Prometheus** and **Grafana** monitoring for the California Housing ML API project.

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture) 
- [🚀 Quick Start](#-quick-start)
- [📊 Prometheus Setup](#-prometheus-setup)
- [📈 Grafana Setup](#-grafana-setup)
- [📏 Available Metrics](#-available-metrics)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

The monitoring stack provides real-time observability for our ML API with:

- **📊 Prometheus**: Time-series metrics collection and storage
- **📈 Grafana**: Interactive dashboards and visualizations  
- **🤖 ML-Focused Metrics**: Model performance, prediction rates, batch processing

---

## 🏗️ Architecture

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

**Data Flow:**
1. **ML API** exposes metrics via `/prometheus` endpoint
2. **Prometheus** scrapes and stores metrics every 15 seconds  
3. **Grafana** queries Prometheus and displays dashboards
4. **Users** monitor ML model performance in real-time

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- California Housing API project setup complete
- Ports 3000, 8000, 9090 available

### 1. Start the Monitoring Stack

```bash
# Start Prometheus, Grafana, and the ML API
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services are running
docker-compose -f docker-compose.monitoring.yml ps
```

**Expected Output:**
```
        Name                      Command               State           Ports         
-------------------------------------------------------------------------------------
housing-api          python -m uvicorn src.api: ...   Up      0.0.0.0:8000->8000/tcp
grafana              /run.sh                          Up      0.0.0.0:3000->3000/tcp
prometheus           /bin/prometheus --config.f ...   Up      0.0.0.0:9090->9090/tcp
```

### 2. Access the Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **🏠 ML API** | http://localhost:8000 | N/A |
| **📊 Prometheus** | http://localhost:9090 | N/A |
| **📈 Grafana** | http://localhost:3000 | admin / admin123 |

### 3. Generate Sample Data

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

### 4. View Your First Dashboard

1. Open **Grafana**: http://localhost:3000
2. Login with `admin` / `admin123`
3. Navigate to **Dashboards → California Housing API**

---

## 📊 Prometheus Setup

### Sample Prometheus Queries

```promql
# Total prediction requests
sum(housing_prediction_requests_total)

# Request rate (per second)
rate(housing_prediction_requests_total[5m])

# 95th percentile response time  
histogram_quantile(0.95, housing_prediction_duration_seconds_bucket)

# Error rate
rate(housing_api_errors_total[5m])

# Current batch processing volume
housing_batch_size_total
```

---

## 📈 Grafana Setup

### Initial Configuration

**Grafana** is pre-configured with:
- **Datasource**: Prometheus at http://localhost:9090
- **Dashboard**: California Housing API monitoring dashboard
- **Admin User**: admin / admin123

### Accessing Grafana

1. **Open Browser**: http://localhost:3000
2. **Login**: admin / admin123  
3. **Navigate**: Dashboards → California Housing API

### Dashboard Panels

The pre-built dashboard includes 5 key panels:

#### 1. 📊 **Total Prediction Requests**
- **Type**: Stat panel  
- **Metric**: `sum(housing_prediction_requests_total)`
- **Shows**: Total predictions since startup

#### 2. ⏱️ **Response Time (95th Percentile)**
- **Type**: Time series graph
- **Metric**: `histogram_quantile(0.95, housing_prediction_duration_seconds_bucket)`  
- **Shows**: API performance trends

#### 3. 🤖 **Model Status**
- **Type**: Stat panel with thresholds
- **Metric**: `housing_model_loaded`
- **Shows**: Whether ML model is loaded (Green=1, Red=0)

#### 4. 📈 **New Training Data**
- **Type**: Time series graph
- **Metric**: `rate(housing_new_data_points_total[5m])`
- **Shows**: Training data submission rate

#### 5. 🔄 **Retraining Events** 
- **Type**: Time series graph
- **Metric**: `housing_retraining_triggered_total`
- **Shows**: Model retraining history

---

## 📏 Available Metrics

### 🎯 Core ML Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `housing_prediction_requests_total` | Counter | Total prediction requests | endpoint, status |
| `housing_model_predictions_total` | Counter | Total model predictions made | - |
| `housing_prediction_duration_seconds` | Histogram | Prediction response time | endpoint |
| `housing_batch_size_total` | Histogram | Batch request sizes | - |
| `housing_model_loaded` | Gauge | Model load status (1=loaded) | - |

### 🔄 ML Pipeline Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `housing_new_data_points_total` | Counter | New training data submitted | - |
| `housing_retraining_triggered_total` | Counter | Model retraining events | - |

### 🚨 Error & Health Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `housing_api_errors_total` | Counter | API errors by endpoint | endpoint, status_code |

### 🔍 FastAPI Built-in Metrics

These are automatically available from the `/metrics` endpoint:

- `housing_api_requests_duration_seconds` - Request processing time
- `housing_api_requests_size_bytes` - Request payload sizes  
- `housing_api_responses_size_bytes` - Response payload sizes
- `housing_api_requests_inprogress` - Concurrent requests
- `housing_api_http_requests_total` - HTTP request counts

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 🚫 **Grafana Shows "No Data"**

**Symptoms**: Dashboard panels show "No data" or "N/A"

**Solutions**:
```bash
# 1. Check if Prometheus is collecting metrics
curl http://localhost:9090/api/v1/query?query=housing_prediction_requests_total

# 2. Verify API is exposing metrics
curl http://localhost:8000/prometheus

# 3. Check Grafana datasource
# Go to Configuration → Data Sources → Prometheus
# Test connection should show "Data source is working"

# 4. Generate test data
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"MedInc": 5.0, "HouseAge": 10.0, "AveRooms": 6.0, "AveBedrms": 1.0, "Population": 3000.0, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0}'
```

#### 🔌 **Prometheus Not Scraping**

**Symptoms**: Targets show as "DOWN" in Prometheus

**Solutions**:
```bash
# 1. Check Docker network connectivity
docker network ls
docker network inspect monitoring_monitoring

# 2. Verify API health
curl http://localhost:8000/health

# 3. Check Prometheus logs
docker-compose -f docker-compose.monitoring.yml logs prometheus

# 4. Restart monitoring stack
docker-compose -f docker-compose.monitoring.yml restart
```

#### 📊 **Missing Metrics**

**Symptoms**: Some expected metrics don't appear

**Solutions**:
```bash
# 1. Check which metrics are actually exposed
curl -s http://localhost:8000/prometheus | grep housing_

# 2. Verify metric names in Grafana queries
# Common typo: housing_prediction_request_total vs housing_prediction_requests_total

# 3. Check if API endpoints have been called
# Metrics only appear after the corresponding API usage
```

#### 🔄 **Dashboard Not Updating**

**Symptoms**: Data appears stale or frozen

**Solutions**:
1. **Check refresh interval**: Set to 30s or 1m
2. **Verify time range**: Ensure it covers recent data  
3. **Clear browser cache**: Hard refresh (Ctrl+F5)
4. **Check Prometheus retention**: Default 30 days

#### 🚨 **Memory Issues**

**Symptoms**: Containers using too much memory

**Solutions**:
```bash
# 1. Check container memory usage
docker stats

# 2. Adjust Prometheus retention
# Edit monitoring/prometheus.yml:
# --storage.tsdb.retention.time=7d  # Reduce from 30d

# 3. Limit Docker container memory  
# Add to docker-compose.monitoring.yml:
# mem_limit: 512m
```

### Debug Commands

```bash
# Check all services status
docker-compose -f docker-compose.monitoring.yml ps

# View service logs
docker-compose -f docker-compose.monitoring.yml logs -f grafana
docker-compose -f docker-compose.monitoring.yml logs -f prometheus  
docker-compose -f docker-compose.monitoring.yml logs -f housing-api

# Test individual components
curl http://localhost:9090/-/healthy    # Prometheus health
curl http://localhost:3000/api/health   # Grafana health  
curl http://localhost:8000/health       # API health

# Check Docker networking
docker exec -it monitoring_prometheus_1 ping housing-api
```

---

## 📚 Additional Resources

### Documentation Links
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **PromQL**: https://prometheus.io/docs/prometheus/latest/querying/
- **FastAPI Monitoring**: https://prometheus.io/docs/guides/instrumenting/

---

*For additional help, refer to the main [README.md](README.md) or check the troubleshooting section above.*