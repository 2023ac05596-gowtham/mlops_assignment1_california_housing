#!/usr/bin/env python3
"""
FastAPI application for California Housing Price Prediction
Enhanced modular version with separated concerns
"""

import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request

# Import custom modules
try:
    # Try relative imports first (when run as module)
    from .database import initialize_database, get_db_manager
    from .metrics import (
        initialize_metrics,
        get_metrics_tracker,
        get_model_metrics,
        RequestTimer,
    )
    from .schemas import (
        HousingFeatures,
        PredictionResponse,
        BatchPredictionRequest,
        BatchPredictionResponse,
    )
    from .models import (
        load_model,
        get_model,
        get_model_name,
        is_model_loaded,
        make_prediction,
        prepare_feature_array,
        validate_business_logic,
        calculate_confidence_score,
        calculate_prediction_range,
    )
    from .config import Config, logger
except ImportError:
    # Fall back to direct imports (when run directly)
    from database import initialize_database, get_db_manager
    from metrics import (
        initialize_metrics,
        get_metrics_tracker,
        get_model_metrics,
        RequestTimer,
    )
    from schemas import (
        HousingFeatures,
        PredictionResponse,
        BatchPredictionRequest,
        BatchPredictionResponse,
    )
    from models import (
        load_model,
        get_model,
        get_model_name,
        is_model_loaded,
        make_prediction,
        prepare_feature_array,
        validate_business_logic,
        calculate_confidence_score,
        calculate_prediction_range,
    )
    from config import Config, logger

# Initialize FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
)


# Middleware for request/response tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware to track all incoming requests and responses"""
    start_time = time.time()

    # Get metrics tracker
    metrics_tracker = get_metrics_tracker()
    db_manager = get_db_manager()

    # Track request
    metrics_tracker.track_request()

    response = await call_next(request)

    # Calculate response time
    response_time = time.time() - start_time
    metrics_tracker.track_response_time(response_time)

    # Log to database
    error_message = (
        None if response.status_code < 400 else f"HTTP {response.status_code}"
    )
    db_manager.log_api_request(
        endpoint=str(request.url.path),
        method=request.method,
        status_code=response.status_code,
        response_time=response_time,
        error_message=error_message,
    )

    # Track errors
    if response.status_code >= 400:
        error_key = f"HTTP_{response.status_code}"
        metrics_tracker.track_error(error_key)

    return response


@app.on_event("startup")
async def startup_event():
    """Initialize the model, database, and metrics on startup"""
    load_model()
    initialize_database(str(Config.get_db_path()))
    initialize_metrics()
    logger.info("FastAPI application started successfully")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "model": get_model_name(),
        "endpoints": {
            "/predict": "Single prediction",
            "/predict/batch": "Batch predictions",
            "/health": "Health check",
            "/metrics": "API monitoring metrics",
            "/docs": "API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = is_model_loaded()
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_name": get_model_name(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics")
async def get_metrics():
    """
    Monitoring metrics endpoint for observability
    Returns API usage statistics, performance metrics, and error rates
    """
    try:
        # Get trackers
        metrics_tracker = get_metrics_tracker()
        model_metrics_tracker = get_model_metrics()
        db_manager = get_db_manager()

        # Get metrics from trackers
        api_stats = metrics_tracker.get_api_statistics()
        performance_metrics = metrics_tracker.get_performance_metrics()
        error_breakdown = metrics_tracker.get_error_breakdown()

        # Get database statistics
        db_stats = db_manager.get_database_stats()
        recent_predictions = db_manager.get_recent_predictions(10)

        # Get model statistics
        model_stats = model_metrics_tracker.get_model_stats(get_model_name())

        # Simple combined stats
        combined_api_stats = api_stats

        return {
            "timestamp": datetime.now().isoformat(),
            "api_statistics": combined_api_stats,
            "performance_metrics": performance_metrics,
            "error_breakdown": error_breakdown,
            "model_info": {
                "model_name": get_model_name(),
                "model_loaded": is_model_loaded(),
                "model_stats": model_stats,
            },
            "recent_predictions": recent_predictions,
            "database_health": {
                "connected": db_manager.health_check(),
                "stats": db_stats,
                "logs_directory": str(Config.get_logs_dir()),
                "database_path": str(Config.get_db_path()),
            },
        }

    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        try:
            metrics_tracker = get_metrics_tracker()
            basic_stats = metrics_tracker.get_basic_stats()
        except Exception:
            basic_stats = {"error": "Metrics tracker not available"}

        return {
            "timestamp": datetime.now().isoformat(),
            "error": f"Unable to generate metrics: {str(e)}",
            "basic_stats": basic_stats,
            "model_loaded": is_model_loaded(),
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: HousingFeatures):
    """
    Make a single prediction for housing price with confidence scores
    """
    with RequestTimer("/predict") as timer:
        input_data = features.dict()
        error_message = None
        prediction = 0.0
        confidence = 0.0

        # Get trackers
        metrics_tracker = get_metrics_tracker()
        model_metrics_tracker = get_model_metrics()
        db_manager = get_db_manager()

        try:
            # Log the prediction request
            logger.info(f"Prediction request received: {input_data}")

            if not is_model_loaded():
                error_message = "Model not loaded"
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Model not loaded",
                        "suggestion": (
                            (
                                "Please wait for the model to load or contact "
                                "support"
                            )
                        ),
                    },
                )

            # Validate business logic
            validation_errors = validate_business_logic(features)
            if validation_errors:
                raise HTTPException(
                    status_code=422,
                    detail=validation_errors[0],
                )

            # Prepare input data
            feature_array = prepare_feature_array(features)

            # Make prediction
            prediction = make_prediction(feature_array)[0]

            # Calculate confidence score
            confidence = calculate_confidence_score(get_model(), feature_array)

            # Track metrics
            metrics_tracker.track_prediction(1)
            model_metrics_tracker.track_model_prediction(
                get_model_name(), float(prediction), confidence
            )

            # Calculate prediction range
            pred_range = calculate_prediction_range(prediction, confidence)

            # Log the prediction result
            logger.info(
                f"Prediction made: ${prediction:.2f} "
                f"(confidence: {confidence:.3f})"
            )

            # Log to database
            response_time = timer.get_response_time()
            db_manager.log_prediction(
                input_data=input_data,
                prediction=float(prediction),
                confidence=confidence,
                response_time=response_time,
                model_used=get_model_name(),
                endpoint="/predict",
            )

            response = PredictionResponse(
                predicted_price=round(float(prediction), 2),
                confidence_score=round(confidence, 2),
                prediction_range={
                    "min": round(pred_range["min"], 2),
                    "max": round(pred_range["max"], 2),
                    "range_percent": pred_range["range_percent"],
                },
                model_used=get_model_name(),
                timestamp=datetime.now().isoformat(),
            )

            return response

        except HTTPException as he:
            # Log error to database
            response_time = timer.get_response_time()
            error_message = str(he.detail)
            db_manager.log_prediction(
                input_data=input_data,
                prediction=prediction,
                confidence=confidence,
                response_time=response_time,
                model_used=get_model_name(),
                endpoint="/predict",
                error_message=error_message,
            )
            # Re-raise HTTP exceptions (validation errors)
            raise
        except Exception as e:
            # Log error to database
            response_time = timer.get_response_time()
            error_message = str(e)
            db_manager.log_prediction(
                input_data=input_data,
                prediction=prediction,
                confidence=confidence,
                response_time=response_time,
                model_used=get_model_name(),
                endpoint="/predict",
                error_message=error_message,
            )
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"Prediction error: {str(e)}",
                    "suggestion": (
                        "Please check your input values and try again. "
                        "Contact support if the problem persists."
                    ),
                },
            )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple housing features with confidence scores
    """
    with RequestTimer("/predict/batch") as timer:
        input_data = {
            "batch_size": len(request.features),
            "features": [f.dict() for f in request.features],
        }
        error_message = None

        # Get trackers
        metrics_tracker = get_metrics_tracker()
        model_metrics_tracker = get_model_metrics()
        db_manager = get_db_manager()

        try:
            logger.info(
                (
                    f"Batch prediction request received for "
                    f"{len(request.features)} samples"
                )
            )

            if not is_model_loaded():
                error_message = "Model not loaded"
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Model not loaded",
                        "suggestion": (
                            (
                                "Please wait for the model to load or contact "
                                "support"
                            )
                        ),
                    },
                )

            if len(request.features) > 1000:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (
                            (
                                f"Batch size too large: "
                                f"{len(request.features)} samples"
                            )
                        ),
                        "suggestion": (
                            "Maximum 1000 predictions allowed. "
                            "Please split your request into smaller batches."
                        ),
                    },
                )

            if len(request.features) == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Empty batch request",
                        "suggestion": (
                            "Please provide at least one set of features for "
                            "prediction"
                        ),
                    },
                )

            # Validate each input and prepare data
            feature_arrays = []
            for i, features in enumerate(request.features):
                # Business logic validation
                validation_errors = validate_business_logic(features)
                if validation_errors:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": (
                                f"Invalid input at index {i}: "
                                f"{validation_errors[0]['error']}"
                            ),
                            "suggestion": validation_errors[0]["suggestion"],
                        },
                    )

                feature_array = prepare_feature_array(features)
                feature_arrays.append(feature_array[0])  # Extract the 1D array

            feature_matrix = np.array(feature_arrays)

            # Make predictions
            predictions = make_prediction(feature_matrix)

            # Track metrics
            metrics_tracker.track_prediction(len(predictions))

            # Calculate confidence scores and build detailed predictions
            detailed_predictions = []
            confidence_scores = []

            for i, (pred, feature_row) in enumerate(
                zip(predictions, feature_arrays)
            ):
                feature_array_single = np.array([feature_row])
                confidence = calculate_confidence_score(
                    get_model(), feature_array_single
                )
                confidence_scores.append(confidence)

                # Track individual prediction for model metrics
                model_metrics_tracker.track_model_prediction(
                    get_model_name(), float(pred), confidence
                )

                pred_range = calculate_prediction_range(pred, confidence)

                detailed_predictions.append(
                    {
                        "predicted_price": round(float(pred), 2),
                        "confidence_score": round(confidence, 2),
                        "prediction_range": {
                            "min": round(pred_range["min"], 2),
                            "max": round(pred_range["max"], 2),
                            "range_percent": pred_range["range_percent"],
                        },
                    }
                )

            logger.info(
                f"Batch prediction completed for {len(predictions)} samples"
            )

            # Log batch prediction to database
            # (using average values for the batch)
            response_time = timer.get_response_time()
            avg_prediction = float(np.mean(predictions))
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

            db_manager.log_prediction(
                input_data=input_data,
                prediction=avg_prediction,
                confidence=avg_confidence,
                response_time=response_time,
                model_used=get_model_name(),
                endpoint="/predict/batch",
            )

            response = BatchPredictionResponse(
                predictions=detailed_predictions,
                model_used=get_model_name(),
                timestamp=datetime.now().isoformat(),
                count=len(predictions),
            )

            return response

        except HTTPException as he:
            # Log error to database
            response_time = timer.get_response_time()
            error_message = str(he.detail)
            db_manager.log_prediction(
                input_data=input_data,
                prediction=0.0,
                confidence=0.0,
                response_time=response_time,
                model_used=get_model_name(),
                endpoint="/predict/batch",
                error_message=error_message,
            )
            # Re-raise HTTP exceptions (validation errors)
            raise
        except Exception as e:
            # Log error to database
            response_time = timer.get_response_time()
            error_message = str(e)
            db_manager.log_prediction(
                input_data=input_data,
                prediction=0.0,
                confidence=0.0,
                response_time=response_time,
                model_used=get_model_name(),
                endpoint="/predict/batch",
                error_message=error_message,
            )
            logger.error(f"Error making batch predictions: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"Batch prediction error: {str(e)}",
                    "suggestion": (
                        "Please check your input data format and try again. "
                        "Contact support if the problem persists."
                    ),
                },
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD,
        log_level=Config.LOG_LEVEL,
    )
