import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ------------------ STYLING: DO NOT CHANGE ANYTHING BELOW ------------------ #
st.set_page_config(page_title="Ask Niel – AI SRE Assistant", layout="centered")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #05060f, #0a0c2a);
            color: #e0f2fe;
        }

        h1 {
            font-size: 2.8rem;
            text-align: center;
            font-weight: bold;
            margin-top: 2rem;
            background: linear-gradient(to right, #38bdf8, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Siri-like pulse ring */
        .siri-circle {
            margin: 40px auto 20px;
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: radial-gradient(circle at center, #4fc3f7 30%, transparent 70%);
            animation: pulseRing 2s infinite ease-in-out;
        }

        @keyframes pulseRing {
            0% {
                transform: scale(1);
                opacity: 0.9;
            }
            50% {
                transform: scale(1.3);
                opacity: 0.5;
            }
            100% {
                transform: scale(1);
                opacity: 0.9;
            }
        }

        .stTextInput>div>div>input {
            width: 100%;
            font-size: 1.2rem;
            padding: 1rem;
            border-radius: 12px;
            border: none;
            background: rgba(255, 255, 255, 0.07);
            color: #e2e8f0;
            box-shadow: 0 0 15px #38bdf8;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            backdrop-filter: blur(10px);
            animation: fadeInUp 0.6s ease-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .typing {
            width: 3rem;
            display: flex;
            justify-content: space-between;
            margin: 20px auto;
        }

        .typing span {
            width: 8px;
            height: 8px;
            background: #38bdf8;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }

        .typing span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.9); opacity: 0.4; }
            40% { transform: scale(1.4); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ ACTUAL APP ------------------ #
st.title("🤖 Ask Niel")
st.markdown('<div class="siri-circle"></div>', unsafe_allow_html=True)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Search UI
query = st.text_input("Ask me about an error you're facing:")

if query:
    st.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"**🆔 Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.markdown(f"**💬 Message:** {df.iloc[best_idx]['Error Message']}")
    st.markdown(f"**🧠 Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.markdown(f"**🛠 Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
    st.markdown('</div>', unsafe_allow_html=True)
