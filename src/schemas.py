#!/usr/bin/env python3
"""
Pydantic schemas for California Housing Price Prediction API
Request and response models for API endpoints
"""

from typing import List
from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    """
    Input schema for housing features
    """

    MedInc: float = Field(
        ...,
        description="Median income in block group",
        ge=0.5,
        le=15.0,
        title="Median Income",
    )
    HouseAge: float = Field(
        ...,
        description="Median house age in block group",
        ge=1,
        le=52,
        title="House Age",
    )
    AveRooms: float = Field(
        ...,
        description="Average number of rooms per household",
        ge=2.0,
        le=15.0,
        title="Average Rooms",
    )
    AveBedrms: float = Field(
        ...,
        description="Average number of bedrooms per household",
        ge=0.1,
        le=5.0,
        title="Average Bedrooms",
    )
    Population: float = Field(
        ..., description="Block group population", ge=3, le=40000, title="Population"
    )
    AveOccup: float = Field(
        ...,
        description="Average number of household members",
        ge=1.0,
        le=50.0,
        title="Average Occupancy",
    )
    Latitude: float = Field(
        ...,
        description="Latitude (California bounds)",
        ge=32.5,
        le=41.95,
        title="Latitude",
    )
    Longitude: float = Field(
        ...,
        description="Longitude (California bounds)",
        ge=-124.35,
        le=-114.13,
        title="Longitude",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.33,
                "HouseAge": 41.0,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322.0,
                "AveOccup": 2.56,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for predictions
    """

    predicted_price: float = Field(
        ..., description="Predicted median house value in USD (United States Dollars)"
    )
    confidence_score: float = Field(
        ..., description="Prediction confidence score (0-1, higher is better)"
    )
    prediction_range: dict = Field(
        ..., description="Estimated price range in USD (min/max bounds)"
    )
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """
    Schema for batch predictions
    """

    features: List[HousingFeatures] = Field(
        ..., description="List of housing features for batch prediction"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions
    """

    predictions: List[dict] = Field(
        ...,
        description="List of prediction results with confidence scores (prices in USD)",
    )
    model_used: str = Field(..., description="Model used for predictions")
    timestamp: str = Field(..., description="Prediction timestamp")
    count: int = Field(..., description="Number of predictions made")
