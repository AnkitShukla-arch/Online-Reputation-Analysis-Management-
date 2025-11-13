import streamlit as st
import pandas as pd
import subprocess
import threading
import time
from pathlib import Path
import requests

st.set_page_config(page_title="AI Reputation Dashboard", layout="wide")

st.markdown("""
    <style>
    body { background: linear-gradient(120deg, #0a0f24, #1b2845); color: white; }
    .stButton>button {
        background: #0ff;
        color: black;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 AI-Powered Reputation Analysis Dashboard")
st.caption("Live Social + News Sentiment • Powered by Dual AI Agents")

brand = st.text_input("Enter Brand Name:", "Tesla")
days = st.slider("Days to Monitor:", 1, 30, 7)
platforms = st.multiselect("Platforms", ["news", "twitter", "reddit"], default=["news", "twitter"])
keywords = st.text_area("Keywords", "autopilot, battery, software update").split(",")

st.markdown("---")

if st.button("🔥 Start Analysis"):
    st.info("Starting monitoring agent...")
    def run_monitoring():
        subprocess.run(["python", "monitoring_agent_v3.py"], check=False)
    threading.Thread(target=run_monitoring).start()
    time.sleep(8)

    st.success("✅ Monitoring complete. Fetching latest data...")

    # Find latest CSV
    files = sorted(Path("outputs").glob("monitoring_*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        st.error("No monitoring results found.")
    else:
        df = pd.read_csv(files[0])
        st.dataframe(df.head(20), use_container_width=True)

        # Sentiment stats
        pos = (df["sentiment"] == "positive").sum()
        neg = (df["sentiment"] == "negative").sum()
        neu = (df["sentiment"] == "neutral").sum()

        st.metric("Positive Mentions", pos)
        st.metric("Negative Mentions", neg)
        st.metric("Neutral Mentions", neu)

        # Send to Sentiment Agent
        try:
            mentions = df[["platform", "author", "text", "date"]].to_dict(orient="records")
            resp = requests.post("http://127.0.0.1:8001/process_batch", json={
                "brand": brand,
                "mentions": mentions,
                "historical_negative_ratio": 0.1,
                "historical_window_size": 200
            })
            if resp.status_code == 200:
                data = resp.json()["data"]
                st.subheader("📊 Reputation Summary")
                st.json(data["summary"])
                st.metric("Reputation Score", data["reputation_score"])
            else:
                st.warning("Sentiment Agent not responding properly.")
        except Exception as e:
            st.error(f"⚠️ Error contacting sentiment agent: {e}")
