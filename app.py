import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config FIRST
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# 🌟 Move glowing circle below the title
st.markdown("""
<style>
body {
    background-color: #1a1a1a;
    color: #e8e8e8;
    font-family: 'Segoe UI', sans-serif;
}

.pulse-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 40px 0 20px 0;
}

.pulse-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle, #9c00ff, #00f5d4);
    box-shadow: 0 0 80px #9c00ff, 0 0 120px #00f5d4, 0 0 140px #9c00ff;
    animation: pulse 1.5s infinite ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 80px #9c00ff, 0 0 120px #00f5d4; }
    50% { transform: scale(1.2); box-shadow: 0 0 150px #9c00ff, 0 0 200px #00f5d4; }
    100% { transform: scale(1); box-shadow: 0 0 80px #9c00ff, 0 0 120px #00f5d4; }
}

input {
    background-color: #333333 !important;
    color: #ffffff !important;
    border: 1px solid #00f5d4 !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
}

input:focus {
    border-color: #9c00ff !important;
    box-shadow: 0 0 8px 4px rgba(156, 0, 255, 0.3) !important;
}

h1 {
    text-align: center;
    color: #00f5d4;
    font-size: 3rem;
    font-weight: 600;
}

h2, h3 {
    color: #9c00ff;
}

button {
    background-color: #00f5d4;
    color: #1a1a1a;
    border-radius: 6px;
    border: none;
    padding: 12px 24px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
}

button:hover {
    background-color: #9c00ff;
    color: white;
}

button:active {
    transform: scale(0.95);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>Ask Niel 🤖</h1>", unsafe_allow_html=True)

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
        background-color: #9c00ff;
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
