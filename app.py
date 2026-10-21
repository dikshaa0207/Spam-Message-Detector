import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the model. Check spam_model.pkl.")
    st.stop()

st.title("📩 Spam Message Detector")
st.write("Enter an SMS message to check whether it is likely spam.")

message = st.text_area(
    "Enter your message",
    placeholder="Type your SMS here...",
    height=150
)

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("🚨 This message is classified as SPAM.")
        else:
            st.success("✅ This message is classified as HAM (not spam).")

st.caption("Powered by TF-IDF and Multinomial Naive Bayes.")
