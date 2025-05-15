import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config FIRST
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# Elegant minimal style
st.markdown("""
    <style>
        html, body {
            background-color: #0e1117;
            color: #f1f1f1;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 600;
            text-align: center;
            color: #00f5d4;
            margin-top: 2rem;
            text-shadow: 0px 0px 6px rgba(0, 245, 212, 0.4);
        }
        .glass-box {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            margin-top: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .dots {
            font-size: 18px;
            letter-spacing: 3px;
            animation: blink 1.2s infinite steps(1, end);
        }
        @keyframes blink {
            0%   { opacity: 0.2; }
            50%  { opacity: 1; }
            100% { opacity: 0.2; }
        }
        input, textarea {
            background-color: #1a1d23 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Glowing Title
st.markdown('<div class="title">🤖 Ask Niel</div>', unsafe_allow_html=True)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Input box
query = st.text_input("What error are you seeing?", placeholder="Type an error message...")

# Typing placeholder
typing_placeholder = st.empty()

if query:
    # Siri-style dots animation
    typing_placeholder.markdown('<div class="dots">🤖 Thinking...</div>', unsafe_allow_html=True)

    # Perform embedding similarity
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Clear typing animation
    typing_placeholder.empty()

    # Display response in elegant box
    with st.container():
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Best Match Found")
        st.markdown(f"**🧾 Error Code:** {df.iloc[best_idx]['Error Code']}")
        st.markdown(f"**📣 Message:** {df.iloc[best_idx]['Error Message']}")
        st.markdown(f"**📌 Cause:** {df.iloc[best_idx]['Cause']}")
        st.markdown(f"**🛠️ Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
        st.markdown('</div>', unsafe_allow_html=True)
