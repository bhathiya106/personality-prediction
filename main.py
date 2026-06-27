from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ===============================
# Load the trained model and encoder
# ===============================
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

# ===============================
# Create FastAPI application
# ===============================
app = FastAPI(
    title="Personality Prediction API",
    description="Predict whether a person is an Introvert or Extrovert",
    version="1.0"
)

# ===============================
# Input Schema
# ===============================
class PersonalityInput(BaseModel):
    Time_spent_Alone: int
    Stage_fear: str
    Social_event_attendance: int
    Going_outside: int
    Drained_after_socializing: str
    Friends_circle_size: int
    Post_frequency: int

# ===============================
# Home Route
# ===============================
@app.get("/")
def home():
    return {
        "message": "Welcome to the Personality Prediction API!"
    }

# ===============================
# Prediction Route
# ===============================
@app.post("/predict")
def predict(data: PersonalityInput):

    # Convert Yes/No to numeric values
    stage_fear = 1 if data.Stage_fear.lower() == "yes" else 0
    drained = 1 if data.Drained_after_socializing.lower() == "yes" else 0

    # Create DataFrame with the same column order used during training
    input_df = pd.DataFrame({
        "Time_spent_Alone": [data.Time_spent_Alone],
        "Stage_fear": [stage_fear],
        "Social_event_attendance": [data.Social_event_attendance],
        "Going_outside": [data.Going_outside],
        "Drained_after_socializing": [drained],
        "Friends_circle_size": [data.Friends_circle_size],
        "Post_frequency": [data.Post_frequency]
    })

    # Make prediction
    prediction = model.predict(input_df)

    # Convert numeric prediction back to label
    personality = encoder.inverse_transform(prediction)

    # Return prediction
    return {
        "prediction": personality[0]
    }
