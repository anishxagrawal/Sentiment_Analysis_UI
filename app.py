import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
                            confidence = result["calibrated_confidence"] * 100
    
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
                        st.markdown("### 📊 Summary Statistics")
                        
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
                        
                        # Create DataFrame for visualizations
                        results_df = pd.DataFrame(result["results"])
                        
                        # ==================== VISUALIZATIONS ====================
                        st.markdown("### 📈 Visual Insights")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Pie Chart - Sentiment Distribution
                            sentiment_counts = results_df['sentiment'].value_counts()
                            
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=sentiment_counts.index,
                                values=sentiment_counts.values,
                                hole=0.4,
                                marker=dict(
                                    colors=['#4CAF50', '#F44336'],  # Softer green and red
                                    line=dict(color='#2d2d2d', width=2)
                                ),
                                textinfo='label+percent',
                                textfont=dict(size=15, color='white')
                            )])
                            
                            fig_pie.update_layout(
                                title={
                                    'text': "Sentiment Distribution",
                                    'font': {'size': 18, 'color': '#E0E0E0'}
                                },
                                showlegend=True,
                                height=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E0E0E0'),
                                legend=dict(
                                    font=dict(size=12, color='#E0E0E0'),
                                    bgcolor='rgba(50,50,50,0.5)'
                                )
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)

                        results_df["confidence"] = pd.to_numeric(
                            results_df["confidence"], errors="coerce"
                        )

                        
                        with viz_col2:
                            # Histogram - Confidence Distribution
                            fig_hist = go.Figure(
                                data=[
                                    go.Histogram(
                                    x=results_df["confidence"],
                                    nbinsx=20,
                                    marker=dict(color="#64B5F6")
                                    )
                                ]
                            )

                            fig_hist.update_layout(
                                title=dict(
                                    text="Confidence Score Distribution",
                                    font=dict(size=18)
                                ),
                                xaxis_title="Confidence Score",
                                yaxis_title="Number of Texts",
                                height=400,
                                showlegend=False
                            )

                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Bar Chart - Sentiment Timeline
                        st.markdown("#### Sentiment Timeline")
                        
                        results_df['index'] = range(1, len(results_df) + 1)
                        
                        fig_timeline = go.Figure()
                        
                        for sentiment, color in [('POSITIVE', '#66BB6A'), ('NEGATIVE', '#EF5350')]:
                            df_sentiment = results_df[results_df['sentiment'] == sentiment]
                            
                            fig_timeline.add_trace(go.Bar(
                                x=df_sentiment['index'],
                                y=df_sentiment['confidence'],
                                name=sentiment,
                                marker=dict(
                                    color=color,
                                    line=dict(color='#2d2d2d', width=1)
                                ),
                                hovertemplate='<b>Text %{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
                            ))
                        
                        fig_timeline.update_layout(
                            title=dict(
                                text="Sentiment Confidence by Text Order",
                                font=dict(size=18)
                            ),
                            xaxis=dict(
                                title=dict(
                                    text="Text Number",
                                    font=dict(size=14)
                                ),
                                tickfont=dict(size=12),
                                gridcolor="#3d3d3d"
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="Confidence Score",
                                    font=dict(size=14)
                                ),
                                tickfont=dict(size=12),
                                gridcolor="#3d3d3d",
                                range=[0, 1]   # confidence is 0–1, makes plot stable
                            ),
                            height=400,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30,30,30,0.3)",
                            barmode="group",        # ✅ safer than overlay
                            showlegend=True
                        )

                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Display results table
                        st.markdown("### 📋 Detailed Results")
                        
                        # Format for display
                        display_df = results_df.copy()
                        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.2f}%")
                        
                        # Add emoji column
                        display_df["emoji"] = display_df["sentiment"].apply(
                            lambda x: "😊" if x == "POSITIVE" else "😞"
                        )
                        
                        # Reorder columns
                        display_df = display_df[["emoji", "text", "sentiment", "confidence"]]
                        
                        # Color-code the dataframe
                        def highlight_sentiment(row):
                            if row["sentiment"] == "POSITIVE":
                                return [
                                    "background-color: #1E3D34; color: #E8F5E9"
                                ] * len(row)
                            else:
                                return [
                                    "background-color: #3D1E1E; color: #FDECEA"
                                ] * len(row)

                        
                        st.dataframe(
                            display_df.style.apply(highlight_sentiment, axis=1),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download button
                        csv = results_df[['text', 'sentiment', 'confidence']].to_csv(index=False)
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