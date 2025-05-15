import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config FIRST!
st.set_page_config(page_title="Ask Niel", page_icon="🔧", layout="centered")

# 🔵 Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main {
        background-color: #0d1117;
    }
    .stTextInput > div > div > input {
        background-color: #161b22;
        color: #58a6ff;
        border: 1px solid #30363d;
        padding: 10px;
        font-size: 18px;
        border-radius: 10px;
    }
    .stTextInput > label {
        font-weight: bold;
        color: #58a6ff;
    }
    .stMarkdown h1, .stMarkdown h2 {
        color: #58a6ff;
    }
    .pulse-circle {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #00f5d4;
        box-shadow: 0 0 0 rgba(0, 245, 212, 0.7);
        animation: pulse-animation 1.6s infinite;
    }
    @keyframes pulse-animation {
        0% {
            box-shadow: 0 0 0 0 rgba(0, 245, 212, 0.7);
        }
        70% {
            box-shadow: 0 0 0 20px rgba(0, 245, 212, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(0, 245, 212, 0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# 🤖 Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🧠 Load CSV and compute embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# 🌟 App title
st.title("🤖 Ask Niel")

# 🔍 User input
query = st.text_input("Enter the error you're seeing:")

# 🔄 Typing placeholder with animation
typing_placeholder = st.empty()

if query:
    # Show glowing pulse while thinking
    typing_placeholder.markdown("""
        <div style='display: flex; justify-content: center; margin-top: 30px; margin-bottom: 20px;'>
            <div class="pulse-circle"></div>
        </div>
    """, unsafe_allow_html=True)

    # Compute similarity
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Clear animation
    typing_placeholder.empty()

    # Show result
    st.subheader("✅ Best Match Found")
    st.markdown(f"**🔢 Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.markdown(f"**📄 Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.markdown(f"**🧠 Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.markdown(f"**🛠 Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
