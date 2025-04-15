import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Page setup - must be first Streamlit command
st.set_page_config(page_title="Ask Niel – AI SRE Helper", page_icon="🤖", layout="centered")

# 🔧 Modern futuristic UI styling
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: radial-gradient(circle at 20% 20%, #111827, #0f172a);
            color: #f8fafc;
        }

        .stTextInput>div>div>input {
            width: 100% !important;
            font-size: 1.2rem;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #4ade80;
            background: rgba(255, 255, 255, 0.05);
            color: #f8fafc;
        }

        .stTextInput>div>div>input:focus {
            border: 1px solid #38bdf8;
            box-shadow: 0 0 10px #38bdf8;
        }

        h1 {
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #38bdf8, #4ade80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 2rem;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(8px);
            animation: fadeIn 0.6s ease-in-out;
        }

        .result-card h2 {
            color: #38bdf8;
        }

        .result-card p {
            margin-bottom: 1rem;
            color: #e2e8f0;
        }

        @keyframes fadeIn {
            0% {opacity: 0; transform: translateY(10px);}
            100% {opacity: 1; transform: translateY(0);}
        }

        .stButton>button {
            background-color: #4ade80;
            color: black;
            font-weight: 600;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
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
st.title("🤖 Ask Niel – AI SRE Assistant")

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
