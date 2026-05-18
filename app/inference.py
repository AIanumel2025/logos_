from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Model configuration

rob_model = "cardiffnlp/twitter-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(rob_model) # applying model's unique tokenizer
model = AutoModelForSequenceClassification.from_pretrained(rob_model)

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# Sentiment prediction

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

    # Get predicted class
    predicted_class = torch.argmax(probabilities).item()

    # Confidence score
    confidence = probabilities[0][predicted_class].item()

    return {
        "text": text,
        "sentiment": LABELS[predicted_class],
        "confidence": round(confidence * 100, 2)
    }
