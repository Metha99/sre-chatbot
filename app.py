import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Must be first command!
st.set_page_config(page_title="Ask Niel - SRE Helper", page_icon="🤖", layout="centered")

# Custom CSS for animations and styling
st.markdown("""
    <style>
        body {
            background-color: #0f1117;
            color: white;
        }
        .glow {
            font-size: 36px;
            color: #0ff;
            text-align: center;
            text-shadow: 0 0 5px #0ff, 0 0 10px #0ff, 0 0 20px #0ff;
        }
        .typing-dots {
            display: inline-block;
            animation: blink 1s infinite;
            font-size: 24px;
            margin-top: 10px;
        }
        @keyframes blink {
            0% {opacity: 0;}
            50% {opacity: 1;}
            100% {opacity: 0;}
        }
        .glass-box {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        input {
            background-color: #1c1e26 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# App title with glow effect
st.markdown('<div class="glow">🤖 Ask Niel - Your AI SRE Assistant</div>', unsafe_allow_html=True)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Search input
query = st.text_input("🔎 Describe the error you're seeing")

# Placeholder for typing animation
typing_placeholder = st.empty()

# Show results
if query:
    # Show typing animation
    typing_placeholder.markdown('<div class="typing-dots">🤖 Thinking...</div>', unsafe_allow_html=True)

    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Clear the typing animation
    typing_placeholder.empty()

    # Display result in styled glass box
    with st.container():
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown(f"### 🔍 Best Match Found")
        st.markdown(f"**🧾 Error Code:** {df.iloc[best_idx]['Error Code']}")
        st.markdown(f"**📣 Message:** {df.iloc[best_idx]['Error Message']}")
        st.markdown(f"**📌 Cause:** {df.iloc[best_idx]['Cause']}")
        st.markdown(f"**🛠️ Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
        st.markdown('</div>', unsafe_allow_html=True)
