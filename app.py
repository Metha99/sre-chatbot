import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config FIRST
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# 🌟 Move glowing circle below the title
st.markdown("""
<style>
body {
    background-color: #0f0f0f;
    color: #f0f0f0;
    font-family: 'Segoe UI', sans-serif;
}

.pulse-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 40px 0 20px 0;
}

.pulse-circle {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: radial-gradient(circle, #00f5d4, #00c4a7);
    box-shadow: 0 0 60px #00f5d4, 0 0 80px #00c4a7;
    animation: pulse 1.8s infinite ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 60px #00f5d4, 0 0 80px #00c4a7; }
    50% { transform: scale(1.15); box-shadow: 0 0 90px #00f5d4, 0 0 120px #00c4a7; }
    100% { transform: scale(1); box-shadow: 0 0 60px #00f5d4, 0 0 80px #00c4a7; }
}

input {
    background-color: #1e1e1e !important;
    color: #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #00f5d4;'>Ask Niel 🤖</h1>", unsafe_allow_html=True)

# 🌟 Glowing Circle Below the Title (new positioning)
st.markdown("""
<div class="pulse-wrapper">
    <div class="pulse-circle"></div>
</div>
""", unsafe_allow_html=True)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Input box for user query
query = st.text_input("Ask your error here:")

# Placeholder for typing animation
typing_placeholder = st.empty()

if query:
    # Typing animation (optional – looks like typing dots)
    typing_placeholder.markdown("""
    <style>
    .typing {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .typing span {
        width: 8px;
        height: 8px;
        margin: 0 4px;
        background-color: #00f5d4;
        border-radius: 50%;
        animation: blink 1.4s infinite both;
    }
    .typing span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0; }
        40% { opacity: 1; }
    }
    </style>
    <div class="typing">
        <span></span><span></span><span></span>
    </div>
    """, unsafe_allow_html=True)

    # Process input
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Remove animation after results are fetched
    typing_placeholder.empty()

    # Show results
    st.markdown("### 🔍 Best Match Found")
    st.write(f"**Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.write(f"**Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.write(f"**Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.write(f"**Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
