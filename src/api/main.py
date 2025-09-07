from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
import joblib
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="Transportation Demand Forecasting API",
    description="API for predicting transportation demand using LSTM and ARIMA models",
    version="1.0.0"
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    features: List[List[float]]  # Sequence of features
    external_features: Optional[Dict[str, float]] = None
    
class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    sequences: List[List[List[float]]]  # Multiple sequences
    external_features: Optional[List[Dict[str, float]]] = None

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float]
    confidence_intervals: Optional[List[List[float]]] = None
    model_info: Dict[str, Any]
    timestamp: str

# Global variables for models
lstm_model = None
arima_model = None
preprocessor = None
model_metadata = {}

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global lstm_model, arima_model, preprocessor, model_metadata
    
    try:
        # Load LSTM model
        lstm_model = torch.load('models/lstm_model.pth', map_location='cpu')
        lstm_model.eval()
        
        # Load ARIMA model
        arima_model = joblib.load('models/arima_model.pkl')
        
        # Load preprocessor
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        # Load metadata
        model_metadata = joblib.load('models/metadata.pkl')
        
        logging.info("Models loaded successfully")
        
    except Exception as e:
        logging.error(f"Error loading models: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Transportation Demand Forecasting API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "lstm": lstm_model is not None,
            "arima": arima_model is not None,
            "preprocessor": preprocessor is not None
        }
    }

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if not model_metadata:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "metadata": model_metadata,
        "lstm_parameters": sum(p.numel() for p in lstm_model.parameters()) if lstm_model else 0,
        "feature_names": model_metadata.get('feature_names', []),
        "model_performance": model_metadata.get('performance_metrics', {})
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """Make single prediction"""
    if lstm_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to tensor
        input_tensor = torch.FloatTensor(request.features).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction, attention_weights = lstm_model(input_tensor)
            pred_value = prediction.item()
        
        # Calculate confidence interval (simplified)
        uncertainty = model_metadata.get('prediction_std', 0.1)
        confidence_interval = [pred_value - 1.96 * uncertainty, pred_value + 1.96 * uncertainty]
        
        return PredictionResponse(
            predictions=[pred_value],
            confidence_intervals=[confidence_interval],
            model_info={
                "model_type": "LSTM",
                "attention_weights_shape": attention_weights.shape,
                "uncertainty": uncertainty
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=PredictionResponse)
async def batch_predict_demand(request: BatchPredictionRequest):
    """Make batch predictions"""
    if lstm_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = []
        confidence_intervals = []
        
        for sequence in request.sequences:
            # Convert to tensor
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction, _ = lstm_model(input_tensor)
                pred_value = prediction.item()
                predictions.append(pred_value)
                
                # Calculate confidence interval
                uncertainty = model_metadata.get('prediction_std', 0.1)
                confidence_interval = [pred_value - 1.96 * uncertainty, pred_value + 1.96 * uncertainty]
                confidence_intervals.append(confidence_interval)
        
        return PredictionResponse(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_info={
                "model_type": "LSTM",
                "batch_size": len(request.sequences),
                "uncertainty": model_metadata.get('prediction_std', 0.1)
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict_arima")
async def predict_arima(steps: int = 24, external_features: Optional[Dict] = None):
    """Make ARIMA predictions"""
    if arima_model is None:
        raise HTTPException(status_code=503, detail="ARIMA model not loaded")
    
    try:
        # Make ARIMA prediction
        forecast = arima_model.forecast(steps=steps)
        
        return {
            "predictions": forecast.tolist(),
            "model_info": {
                "model_type": "ARIMA",
                "steps": steps,
                "aic": getattr(arima_model, 'aic', None),
                "bic": getattr(arima_model, 'bic', None)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ARIMA prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
