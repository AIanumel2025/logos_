from fastapi import FastAPI
from pydantic import BaseModel

from app.preprocessing import clean_text
from app.inference import predict_sentiment

# initialise the app

app = FastAPI(
    title="AI Social Sentiment API",
    description="Transformer-based sentiment analysis service",
    version="1.0"
)

# request the schema

class TextRequest(BaseModel):
    text: str


# health check endpoint

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


# prediction endpoint

@app.post("/predict")
def predict(request: TextRequest):

    # Step 1: clean text
    cleaned = clean_text(request.text)

    # Step 2: model inference
    result = predict_sentiment(cleaned)

    # Step 3: return structured output
    return {
        "input_text": request.text,
        "cleaned_text": cleaned,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"]
    }
