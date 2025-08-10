# Model Retraining

Automated model retraining system that collects new training data via API and retrains the model when ≥50 samples are available.

## API Endpoints

### Submit Single Training Data
```bash
curl -X POST http://localhost:8000/training/submit \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "MedInc": 8.33,
      "HouseAge": 41.0,
      "AveRooms": 6.98,
      "AveBedrms": 1.02,
      "Population": 322.0,
      "AveOccup": 2.56,
      "Latitude": 37.88,
      "Longitude": -122.23
    },
    "actual_price": 452600.0
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Training data submitted successfully. Total samples: 51",
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

### Submit Batch Training Data
```bash
curl -X POST http://localhost:8000/training/submit/batch \
  -H "Content-Type: application/json" \
  -d '{
    "training_data": [
      {
        "features": {
          "MedInc": 8.33,
          "HouseAge": 41.0,
          "AveRooms": 6.98,
          "AveBedrms": 1.02,
          "Population": 322.0,
          "AveOccup": 2.56,
          "Latitude": 37.88,
          "Longitude": -122.23
        },
        "actual_price": 452600.0
      },
      {
        "features": {
          "MedInc": 7.25,
          "HouseAge": 35.0,
          "AveRooms": 5.85,
          "AveBedrms": 0.98,
          "Population": 285.0,
          "AveOccup": 2.12,
          "Latitude": 36.75,
          "Longitude": -121.48
        },
        "actual_price": 389000.0
      }
    ]
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Batch processed: 2 samples added",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "samples_added": 2,
  "total_samples": 52,
  "failed_samples": 0
}
```

> ✅ **Efficient Batch Submission**: Submit up to 100 samples in a single API call for faster data collection.

### Check Retraining Status
```bash
curl -X GET http://localhost:8000/training/status
```

**Response:**
```json
{
  "new_data_samples": 75,
  "model_age_days": 5,
  "should_retrain": true,
  "retrain_reason": "Sufficient new data available (75 samples >= 50 threshold)",
  "daily_retrain_attempts": 1,
  "max_daily_attempts": 2,
  "thresholds": {
    "min_samples": 50,
    "max_model_age_days": 7
  },
  "recent_retrains": []
}
```

### Manual Retraining
```bash
curl -X POST http://localhost:8000/training/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "manual_trigger",
    "force": false
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Retraining triggered successfully",
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

## Process

1. **Data Collection**: Submit samples via `/training/submit` (single) or `/training/submit/batch` (up to 100) → stored in `data/new_training_data.csv`
2. **Auto Trigger**: Retraining starts when ≥50 samples collected
3. **Model Training**: DecisionTree model trained with new + existing data
4. **Validation & Replacement**: New model validated before replacing current model
5. **Rate Limiting**: Max 2 retraining attempts per day

## Safety Features

- **Data Validation**: Feature validation and reasonable price ranges
- **Model Validation**: Performance checks before deployment
- **Backup**: Previous models backed up before replacement
- **Rate Limiting**: Max 2 retraining attempts per day

## Configuration & Files

- **Sample Threshold**: 50 samples, **Daily Limit**: 2 attempts
- **Training Data**: `data/new_training_data.csv`
- **Logs**: `data/retrain_log.json`
- **Model Backups**: `models/backup/`
- **Implementation**: `src/retraining.py`

## Troubleshooting

**Check Status & Logs:**
```bash
# Check retraining status
curl http://localhost:8000/training/status

# Check logs for errors
cat data/retrain_log.json

# Verify training data format
head data/new_training_data.csv
```

**Common Issues:**
- **Rate limit exceeded**: Wait for daily limit reset
- **Data validation errors**: Ensure feature values are within expected ranges
- **Batch submission failures**: Check response for failed samples and retry with corrected data
- **Retraining failures**: Check logs for specific error messages