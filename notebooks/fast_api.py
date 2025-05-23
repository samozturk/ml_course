# This code is intended to be run as a Python script using Uvicorn.
# Save this as `main_api.py` (or similar) and run from terminal:
# uvicorn main_api:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging # Re-using logging setup from previous module for context
import os
# (You might want to move Pydantic models and logging config to separate files in a real app)

# --- Pydantic Models for API ---
# Input features model (re-using from previous section for consistency)
class ModelInputFeatures(BaseModel):
    feature1_num: float
    feature2_num: float
    feature3_cat: str # In a real API, might use Enum: Literal['A', 'B', 'C']
    feature4_ord: str # Enum: Literal['Low', 'Medium', 'High']
    feature5_nan_num: Optional[float] = None

class PredictionPayload(BaseModel):
    instances: List[ModelInputFeatures]

class PredictionResponse(BaseModel):
    predictions: List[float] # Example: list of predicted scores or class labels
    model_version: str = "v0.1.0" # Example metadata

# --- Basic Logging Setup (can be more sophisticated) ---
LOG_DIR = "logs" # Ensure this directory exists or create it
LOG_FILE = os.path.join(LOG_DIR, "fastapi_app.log")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- FastAPI Application ---
app = FastAPI(
    title="ML Model Prediction API",
    description="An API to get predictions from a dummy ML model.",
    version="1.0.0",
)

# --- Dummy Model Placeholder ---
# In a real application, you would load your trained model here (e.g., using joblib)
# For now, let's define a placeholder function.
def dummy_ml_model_predict(input_features_list: List[ModelInputFeatures]) -> List[float]:
    """
    Placeholder for an ML model.
    This dummy model "predicts" by summing feature1_num and feature2_num,
    and adds a small value based on category.
    """
    predictions = []
    for features in input_features_list:
        logger.info(f"Predicting for features: {features.model_dump()}")
        # Dummy logic
        pred_value = features.feature1_num + features.feature2_num
        if features.feature3_cat == 'A':
            pred_value += 1.0
        elif features.feature3_cat == 'B':
            pred_value += 2.0
        else: # 'C' or others
            pred_value += 0.5

        # Consider feature5_nan_num if not None
        if features.feature5_nan_num is not None:
            pred_value += features.feature5_nan_num * 0.1

        predictions.append(round(pred_value, 2))
    return predictions

# --- API Endpoints ---
@app.get("/")
async def read_root():
    """A simple root endpoint to check if the API is running."""
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the ML Model Prediction API!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check '/health' accessed.")
    return {"status": "ok", "message": "API is healthy."}

@app.post("/predict", response_model=PredictionResponse)
async def create_prediction(payload: PredictionPayload):
    """"
    Args:
        payload (PredictionPayload): The input data for prediction.
        Returns:
            PredictionResponse: The prediction results."""
    logger.info(f"Received prediction request with {len(payload.instances)} instances.")
    if not payload.instances:
        logger.warning("Prediction request received with no instances.")
        raise HTTPException(status_code=400, detail="No instances provided for prediction.")

    try:
        # In a real app, you'd pass this to your actual model's prediction function
        # The model would likely expect data in a specific format (e.g., Pandas DataFrame or NumPy array)
        # For now, our dummy model takes the Pydantic models directly
        predictions = dummy_ml_model_predict(payload.instances)
        logger.info(f"Generated predictions: {predictions}")
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# To run this application:
# 1. Save the code as `main_api.py`.
# 2. Make sure you have FastAPI and Uvicorn installed:
#    pip install fastapi uvicorn[standard] pydantic
# 3. Run from your terminal in the directory where you saved the file:
#    uvicorn main_api:app --reload
# or the new way: fastapi dev fast_api.py
#
# You can then access the API documentation at http://127.0.0.1:8000/docs
# And make POST requests to http://127.0.0.1:8000/predict
# Example POST request body (e.g., using curl or Postman):
"""
{
  "instances": [
    {
      "feature1_num": 10.5,
      "feature2_num": 5.2,
      "feature3_cat": "A",
      "feature4_ord": "Medium",
      "feature5_nan_num": 1.0
    },
    {
      "feature1_num": 2.0,
      "feature2_num": 1.1,
      "feature3_cat": "C",
      "feature4_ord": "Low"
    }
  ]
}
"""
# Expected response:
"""
{
  "predictions": [
    16.8,  // (10.5 + 5.2 + 1.0 (for A) + 1.0*0.1)
    3.6    // (2.0 + 1.1 + 0.5 (for C) + 0 (feature5 is None))
  ],
  "model_version": "v0.1.0"
}
"""
if __name__ == "__main__":
    # This block is for direct execution (python main_api.py) but uvicorn is preferred for serving.
    # For development, you'd typically use: uvicorn main_api:app --reload
    print("To run this FastAPI application, use Uvicorn:")
    print("uvicorn main_api:app --reload")
    print("Then open your browser to http://127.0.0.1:8000/docs")

    # You could add a simple test call here if not using uvicorn for a quick check
    # test_payload_dict = {
    #   "instances": [
    #     { "feature1_num": 10.5, "feature2_num": 5.2, "feature3_cat": "A", "feature4_ord": "Medium", "feature5_nan_num": 1.0 },
    #     { "feature1_num": 2.0, "feature2_num": 1.1, "feature3_cat": "C", "feature4_ord": "Low" }
    #   ]
    # }
    # test_payload_obj = PredictionPayload(**test_payload_dict)
    # preds = dummy_ml_model_predict(test_payload_obj.instances)
    # print(f"Test predictions (direct call): {preds}")