import streamlit as st
import requests
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

# API endpoint
API_URL = "https://anishxagrawal-sentiment-analysis-api.hf.space"

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "text_key" not in st.session_state:
    st.session_state.text_key = 0

# Title
st.title("🎭 Sentiment Analysis App")
st.markdown("Analyze the sentiment of any text using AI!")

# Create tabs
tab1, tab2 = st.tabs(["📝 Single Text", "📊 Batch Analysis"])

# ==================== TAB 1: Single Text ====================
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Enter your text:")
    
    with col2:
        st.markdown("### Stats:")
    
    # Quick examples
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
    
    # Text area
    user_text = st.text_area(
        "Type or paste your text here:",
        value=st.session_state.user_input,
        height=150,
        key=f"text_area_{st.session_state.text_key}"
    )
    
    st.session_state.user_input = user_text
    
    # Analyze button
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

# ==================== TAB 2: Batch Analysis ====================
with tab2:
    st.markdown("### 📊 Analyze Multiple Texts")
    st.markdown("Upload a CSV file or enter multiple texts (one per line)")
    
    # Choose input method
    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload CSV", "✍️ Enter Multiple Texts"],
        horizontal=True
    )
    
    texts_to_analyze = []
    
    if input_method == "📄 Upload CSV":
        st.markdown("**CSV Format:** File should have a column named 'text' or 'review' or 'comment'")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Find text column
                text_column = None
                for col in ['text', 'review', 'comment', 'content', 'message']:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column is None:
                    text_column = st.selectbox("Select the text column:", df.columns)
                
                st.markdown(f"**Using column:** `{text_column}`")
                
                # Preview
                with st.expander("📋 Preview Data"):
                    st.dataframe(df.head(10))
                
                texts_to_analyze = df[text_column].astype(str).tolist()
                
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    else:  # Manual text entry
        batch_text_input = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="I love this product!\nThis is terrible\nNot bad at all\n..."
        )
        
        if batch_text_input:
            texts_to_analyze = [line.strip() for line in batch_text_input.split('\n') if line.strip()]
            st.info(f"📝 {len(texts_to_analyze)} texts ready to analyze")
    
    # Analyze button
    if st.button("🚀 Analyze All", type="primary", disabled=len(texts_to_analyze) == 0):
        if texts_to_analyze:
            try:
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    response = requests.post(
                        f"{API_URL}/batch-predict",
                        json={"texts": texts_to_analyze},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display summary
                        st.markdown("### 📊 Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Processed", result["summary"]["successfully_processed"])
                        with col2:
                            st.metric("😊 Positive", result["summary"]["positive_count"])
                        with col3:
                            st.metric("😞 Negative", result["summary"]["negative_count"])
                        with col4:
                            st.metric("Avg Confidence", f"{result['summary']['average_confidence']*100:.1f}%")
                        
                        # Most common sentiment
                        most_common = result["summary"]["most_common_sentiment"]
                        if most_common == "POSITIVE":
                            st.success(f"🎉 Overall sentiment: **Positive** ({result['summary']['positive_count']} out of {result['total_processed']})")
                        else:
                            st.error(f"😔 Overall sentiment: **Negative** ({result['summary']['negative_count']} out of {result['total_processed']})")
                        
                        # Display results table
                        st.markdown("### 📋 Detailed Results")
                        
                        results_df = pd.DataFrame(result["results"])
                        results_df["confidence"] = results_df["confidence"].apply(lambda x: f"{x*100:.2f}%")
                        
                        # Add emoji column
                        results_df["emoji"] = results_df["sentiment"].apply(
                            lambda x: "😊" if x == "POSITIVE" else "😞"
                        )
                        
                        # Reorder columns
                        results_df = results_df[["emoji", "text", "sentiment", "confidence"]]
                        
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try with fewer texts or wait for API to warm up.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Could not connect to API: {str(e)}")

# ==================== Sidebar ====================
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