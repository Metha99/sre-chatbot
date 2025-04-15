import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Page setup - must be first Streamlit command
st.set_page_config(page_title="Ask Niel – AI SRE Helper", page_icon="🤖", layout="centered")

# 🔧 Modern futuristic UI styling
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #0f0f3f, #1a1a60, #111);
            color: #f1f5f9;
            animation: bgFade 12s ease-in-out infinite alternate;
        }

        @keyframes bgFade {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }

        h1 {
            font-size: 2.8rem;
            text-align: center;
            font-weight: bold;
            margin-top: 2rem;
            background: linear-gradient(to right, #00f0ff, #38bdf8, #4ade80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stTextInput>div>div>input {
            width: 100% !important;
            font-size: 1.3rem;
            padding: 1rem;
            border-radius: 14px;
            border: none;
            background: rgba(255, 255, 255, 0.07);
            color: #e2e8f0;
            box-shadow: 0 0 12px #0ff;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 8px #38bdf8; }
            50% { box-shadow: 0 0 18px #38bdf8; }
            100% { box-shadow: 0 0 8px #38bdf8; }
        }

        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 8px 30px rgba(0, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .result-card p {
            font-size: 1.1rem;
            color: #e0f2fe;
            margin-bottom: 1rem;
            position: relative;
        }

        .result-card p::after {
            content: "";
            display: block;
            width: 100%;
            height: 2px;
            background: linear-gradient(to right, #38bdf8, transparent);
            margin-top: 0.5rem;
        }

        .stButton>button {
            background-color: #4ade80;
            color: black;
            font-weight: 600;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #22c55e;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and embed error data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# App title
st.title("🤖 Ask Niel")

# Search bar
query = st.text_input("What error are you facing?")

# Handle query
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Display best match result
    st.markdown(f"""
    <div class="result-card">
        <h2>🧠 Best Match Found</h2>
        <p><strong>🔑 Error Code:</strong> {df.iloc[best_idx]['Error Code']}</p>
        <p><strong>💬 Message:</strong> {df.iloc[best_idx]['Error Message']}</p>
        <p><strong>⚠️ Cause:</strong> {df.iloc[best_idx]['Cause']}</p>
        <p><strong>🛠 Resolution:</strong> {df.iloc[best_idx]['Resolution Steps']}</p>
    </div>
    """, unsafe_allow_html=True)
