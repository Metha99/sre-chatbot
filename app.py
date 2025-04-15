import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config FIRST
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# 🌟 Adjusting layout for a clean, minimal look
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #2b2b2b, #121212); 
    color: #e8e8e8;
    font-family: 'Segoe UI', sans-serif;
    margin: 0;
    padding: 0;
}

.pulse-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
}

.pulse-circle {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: radial-gradient(circle, #00ff99, #0099cc);
    box-shadow: 0 0 80px #00ff99, 0 0 120px #0099cc, 0 0 140px #00ff99;
    animation: pulse 1.5s infinite ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 80px #00ff99, 0 0 120px #0099cc; }
    50% { transform: scale(1.2); box-shadow: 0 0 150px #00ff99, 0 0 200px #0099cc; }
    100% { transform: scale(1); box-shadow: 0 0 80px #00ff99, 0 0 120px #0099cc; }
}

input {
    background-color: #333333 !important;
    color: #ffffff !important;
    border: 1px solid #00ff99 !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    transition: 0.3s ease;
    width: 100%;
}

input:focus {
    border-color: #00ffcc !important;
    box-shadow: 0 0 8px 4px rgba(0, 255, 204, 0.3) !important;
}

h1 {
    text-align: center;
    color: #00ff99;
    font-size: 2.5rem;
    font-weight: 600;
    margin-top: 30px;
}

h2, h3 {
    color: #0099cc;
}

button {
    background-color: #00ff99;
    color: #121212;
    border-radius: 8px;
    border: none;
    padding: 12px 24px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 30px;
}

button:hover {
    background-color: #0099cc;
    color: white;
}

button:active {
    transform: scale(0.95);
}

.result-card {
    background-color: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    font-size: 1.1rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.result-card h3 {
    color: #00ff99;
    font-size: 1.3rem;
    margin-bottom: 10px;
}

.result-card p {
    color: #e8e8e8;
    line-height: 1.6;
    margin-bottom: 8px;
}

.typing-animation {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

.typing-animation span {
    width: 10px;
    height: 10px;
    margin: 0 5px;
    background-color: #00ff99;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
}

.typing-animation span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-animation span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes blink {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
}

#chat-avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background-image: url('https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3pxZmc4YTF4NmRubWt0ajIyNG04aHdiamV5cWx3YWFxdXFlcTZnbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YaiYxMzWNgtg8iNaAm/giphy.gif');
    background-size: cover;
    background-position: center;
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
    # Show typing animation until results are fetched
    typing_placeholder.markdown("""
    <div class="typing-animation">
        <span></span><span></span><span></span>
    </div>
    """, unsafe_allow_html=True)

    # Process input
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Remove animation after results are fetched
    typing_placeholder.empty()

    # Show results in clean layout without extra box
    st.markdown("### 🔍 Best Match Found")
    st.markdown(f"**Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.markdown(f"**Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.markdown(f"**Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.markdown(f"**Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
