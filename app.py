import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

API_URL = "https://anishxagrawal-sentiment-analysis-api.hf.space"

# ---------------- Session State ----------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "text_key" not in st.session_state:
    st.session_state.text_key = 0

# ---------------- Title ----------------
st.title("🎭 Sentiment Analysis App")
st.markdown("Analyze the sentiment of any text using AI.")

tab1, tab2 = st.tabs(["📝 Single Text", "📊 Batch Analysis"])

# =====================================================
# TAB 1 — SINGLE TEXT
# =====================================================
with tab1:
    st.markdown("### Enter your text")

    col_ex1, col_ex2, col_ex3 = st.columns(3)

    if col_ex1.button("😊 Positive Example"):
        st.session_state.user_input = "I absolutely love this product! It's amazing."
        st.session_state.text_key += 1

    if col_ex2.button("😞 Negative Example"):
        st.session_state.user_input = "This is terrible and disappointing."
        st.session_state.text_key += 1

    if col_ex3.button("😐 Mixed Example"):
        st.session_state.user_input = "The product is okay, nothing special but not bad."
        st.session_state.text_key += 1

    user_text = st.text_area(
        "Text",
        value=st.session_state.user_input,
        height=150,
        key=f"text_{st.session_state.text_key}"
    )

    st.session_state.user_input = user_text

    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_text.strip():
            try:
                with st.spinner("Analyzing..."):
                    res = requests.post(
                        f"{API_URL}/predict",
                        json={"text": user_text},
                        timeout=30
                    )

                if res.status_code == 200:
                    result = res.json()

                    label = result["sentiment"]
                    confidence = result["calibrated_confidence"] * 100

                    if label == "POSITIVE":
                        st.success(f"😊 **Positive** — {confidence:.2f}%")
                    elif label == "NEGATIVE":
                        st.error(f"😞 **Negative** — {confidence:.2f}%")
                    else:
                        st.info(f"😐 **Neutral** — {confidence:.2f}%")

                else:
                    st.error(f"API Error: {res.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Please enter some text.")

# =====================================================
# TAB 2 — BATCH ANALYSIS
# =====================================================
with tab2:
    st.markdown("### 📊 Batch Sentiment Analysis")

    input_method = st.radio(
        "Input method",
        ["📄 Upload CSV", "✍️ Enter Texts"],
        horizontal=True
    )

    texts = []

    if input_method == "📄 Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            col = next((c for c in df.columns if c.lower() in ["text", "review", "comment"]), None)
            if not col:
                col = st.selectbox("Select text column", df.columns)
            texts = df[col].astype(str).tolist()
            st.success(f"Loaded {len(texts)} texts")

    else:
        raw = st.text_area("One text per line", height=200)
        if raw:
            texts = [t.strip() for t in raw.split("\n") if t.strip()]
            st.info(f"{len(texts)} texts ready")

    if st.button("🚀 Analyze All", type="primary", disabled=len(texts) == 0):
        try:
            with st.spinner("Processing batch..."):
                res = requests.post(
                    f"{API_URL}/batch-predict",
                    json={"texts": texts},
                    timeout=60
                )

            if res.status_code == 200:
                data = res.json()
                results_df = pd.DataFrame(data["results"])

                # ---------------- Summary ----------------
                st.markdown("### 📊 Summary")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Processed", data["summary"]["successfully_processed"])
                c2.metric("😊 Positive", data["summary"]["positive_count"])
                c3.metric("😞 Negative", data["summary"]["negative_count"])
                c4.metric("😐 Neutral", data["summary"]["neutral_count"])

                # ---------------- Pie Chart ----------------
                st.markdown("### Sentiment Distribution")

                sentiment_counts = results_df["sentiment"].value_counts()

                fig_pie = go.Figure(
                    data=[go.Pie(
                        labels=sentiment_counts.index,
                        values=sentiment_counts.values,
                        hole=0.4
                    )]
                )

                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

                # ---------------- Histogram ----------------
                st.markdown("### Confidence Distribution")

                fig_hist = go.Figure(
                    data=[go.Histogram(x=results_df["calibrated_confidence"], nbinsx=20)]
                )
                fig_hist.update_layout(
                    xaxis_title="Calibrated Confidence",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # ---------------- Timeline ----------------
                st.markdown("### Confidence Timeline")

                results_df["index"] = range(1, len(results_df) + 1)

                fig_timeline = go.Figure()
                colors = {
                    "POSITIVE": "#66BB6A",
                    "NEGATIVE": "#EF5350",
                    "NEUTRAL": "#FFCA28"
                }

                for s in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    subset = results_df[results_df["sentiment"] == s]
                    fig_timeline.add_trace(go.Bar(
                        x=subset["index"],
                        y=subset["calibrated_confidence"],
                        name=s,
                        marker_color=colors[s]
                    ))

                fig_timeline.update_layout(
                    yaxis=dict(range=[0, 1]),
                    barmode="group",
                    height=400
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

                # ---------------- Table ----------------
                st.markdown("### Detailed Results")

                display_df = results_df.copy()
                display_df["confidence"] = display_df["calibrated_confidence"].apply(
                    lambda x: f"{x*100:.2f}%"
                )
                display_df["emoji"] = display_df["sentiment"].map({
                    "POSITIVE": "😊",
                    "NEGATIVE": "😞",
                    "NEUTRAL": "😐"
                })

                display_df = display_df[["emoji", "text", "sentiment", "confidence"]]

                st.dataframe(display_df, use_container_width=True)

                # ---------------- Download ----------------
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "sentiment_results.csv",
                    "text/csv"
                )

            else:
                st.error(f"API Error: {res.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("About")
st.sidebar.info(
    "Sentiment analysis using DistilBERT with confidence calibration "
    "and neutral inference logic."
)

st.sidebar.header("API Stats")
try:
    r = requests.get(f"{API_URL}/stats", timeout=5)
    if r.status_code == 200:
        st.sidebar.metric("Total Predictions", r.json()["total_predictions"])
except:
    st.sidebar.write("Stats unavailable")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Anish Agrawal")
st.sidebar.markdown(f"[API Docs]({API_URL}/docs)")
