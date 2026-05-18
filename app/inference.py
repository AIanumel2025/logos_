from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# MODEL CONFIGURATION

rob_model = "cardiffnlp/twitter-roberta-base-sentiment"

# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(rob_model)
model = AutoModelForSequenceClassification.from_pretrained(rob_model)

# Sentiment labels

LABELS = {
0: "Negative",
1: "Neutral",
2: "Positive"
}


# SENTIMENT PREDICTION FUNCTION

def predict_sentiment(text):

# Tokenize input text
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probabilities = F.softmax(outputs.logits, dim=1)

# Extract prediction
predicted_class = torch.argmax(probabilities).item()

# Confidence score
confidence = probabilities[0][predicted_class].item()

return {
    "text": text,
    "sentiment": LABELS[predicted_class],
    "confidence": round(confidence * 100, 2)
}

