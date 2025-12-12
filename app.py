import streamlit as st
import requests

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="centered"
)

# API endpoint (your HF Space URL)
API_URL = "https://anishxagrawal-sentiment-analysis-api.hf.space"

# --- Initialize session state ---
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "text_key" not in st.session_state:
    st.session_state.text_key = 0

# Title and description
st.title("🎭 Sentiment Analysis App")
st.markdown("Analyze the sentiment of any text using AI!")

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Enter your text:")
    
with col2:
    st.markdown("### Stats:")

# --- Quick Examples ---
st.markdown("**Try these examples:**")
col_ex1, col_ex2, col_ex3 = st.columns(3)

if col_ex1.button("😊 Positive Example"):
    st.session_state.user_input = "I absolutely love this product! It's amazing and exceeded my expectations!"
    st.session_state.text_key += 1

if col_ex2.button("😞 Negative Example"):
    st.session_state.user_input = "This is terrible and very disappointing. Would not recommend."
    st.session_state.text_key += 1

if col_ex3.button("😐 Mixed Example"):
    st.session_state.user_input = "The product is okay, nothing special but not bad either."
    st.session_state.text_key += 1

# --- Text Area (auto refresh using text_key) ---
user_text = st.text_area(
    "Type or paste your text here:",
    value=st.session_state.user_input,
    height=150,
    key=f"text_area_{st.session_state.text_key}"
)

# Sync text area to session state
st.session_state.user_input = user_text

# --- Analyze Button ---
if st.button("🔍 Analyze Sentiment", type="primary"):
    if st.session_state.user_input.strip():
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": st.session_state.user_input},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()

                    # Check if it's an error from your API validation
                    if "error" in result:
                        st.warning(f"⚠️ {result['error']}")
                    else:
                        label = result["sentiment"]
                        confidence = result["confidence"] * 100

                        if label.lower() == "positive":
                            st.success(f"😊 Sentiment: **Positive**\n\nConfidence: **{confidence:.2f}%**")
                        elif label.lower() == "negative":
                            st.error(f"😞 Sentiment: **Negative**\n\nConfidence: **{confidence:.2f}%**")
                        else:
                            st.info(f"😐 Sentiment: **Neutral**\n\nConfidence: **{confidence:.2f}%**")

                else:
                    st.error(f"API Error: {response.status_code}")
                    
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The API might be starting up (takes ~30 seconds on first use).")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Could not connect to API: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text first!")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a DistilBERT model to analyze sentiment in text. "
    "The model is deployed as a FastAPI service."
)

st.sidebar.header("API Info")

try:
    stats_response = requests.get(f"{API_URL}/stats", timeout=5)
    if stats_response.status_code == 200:
        stats_data = stats_response.json()
        st.sidebar.metric(
            label="Total Predictions",
            value=stats_data["total_predictions"]
        )
    else:
        st.sidebar.write("Stats unavailable")
except:
    st.sidebar.write("Could not fetch stats")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Anish Agrawal")
st.sidebar.markdown(f"[View API Docs]({API_URL}/docs)")
