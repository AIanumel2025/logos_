import streamlit as st

from app.preprocessing import clean_text
from app.inference import predict_sentiment

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Social Sentiment Engine",
    page_icon="🧠",
    layout="centered"
)


# TITLE & DESCRIPTION

st.title("🧠 AI Social Sentiment Engine")
st.markdown("""
A transformer-based NLP system for classifying analysing twitter/ X sentiments in social media text.

Built using:
- Twitter-RoBERTa (Hugging Face)
- PyTorch inference pipeline
- Custom preprocessing layer
""")


# USER INPUT

user_input = st.text_area(
    "Enter text for sentiment analysis:",
    placeholder="e.g. I love how AI is changing the world!",
    height=150
)

# INFERENCE TRIGGER

if st.button("Analyse Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Step 1: preprocessing
        cleaned_text = clean_text(user_input)

        # Step 2: inference
        result = predict_sentiment(cleaned_text)

        
        # OUTPUT DISPLAY
        
        st.subheader("Result")

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        elif sentiment == "Negative":
            st.error(f"Sentiment: {sentiment}")
        else:
            st.info(f"Sentiment: {sentiment}")

        st.metric(label="Confidence Score", value=f"{confidence}%")

        
        # DEBUG VIEW
        
        with st.expander("View processed input"):
            st.write("Original Text:", user_input)
            st.write("Cleaned Text:", cleaned_text)
