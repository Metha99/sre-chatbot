import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ✅ Set page configuration first
st.set_page_config(page_title="Ask Niel", page_icon="🔧", layout="centered")

# 🎨 Custom CSS for a futuristic glowing pulse and minimal theme
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

    /* 🌐 Glowing Siri-like pulse */
    .pulse-container {
        display: flex;
        justify-content: center;
        margin: 40px 0 20px 0;
    }
    .pulse-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: radial-gradient(circle, #00f5d4, #00c4a7);
        box-shadow: 0 0 20px #00f5d4;
        animation: pulse 1.6s infinite ease-in-out;
    }
    @keyframes pulse {
        0% {
            transform: scale(1);
            box-shadow: 0 0 20px #00f5d4;
        }
        50% {
            transform: scale(1.2);
            box-shadow: 0 0 30px #00f5d4;
        }
        100% {
            transform: scale(1);
            box-shadow: 0 0 20px #00f5d4;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 🤖 Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🧠 Load and process the CSV
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# 🧠 App title
st.title("🤖 Ask Niel")

# 🔍 Input field
query = st.text_input("Enter the error you're seeing:")

# ✨ Typing animation while searching
typing_placeholder = st.empty()

if query:
    typing_placeholder.markdown("""
        <div class="pulse-container">
            <div class="pulse-circle"></div>
        </div>
    """, unsafe_allow_html=True)

    # 🔍 Find the closest match
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    typing_placeholder.empty()

    # 🧾 Display results
    st.subheader("✅ Best Match Found")
    st.markdown(f"**🔢 Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.markdown(f"**📄 Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.markdown(f"**🧠 Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.markdown(f"**🛠 Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
