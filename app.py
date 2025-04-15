import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Custom CSS for modern design and animations
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #F4F6F9;
        }
        .stTextInput input {
            font-size: 18px;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #ccc;
            width: 100%;
            margin-top: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stTextInput input:focus {
            border-color: #4a90e2;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #357ab7;
        }
        .result-card {
            background-color: #fff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        h1, h2 {
            color: #333;
            font-family: 'Poppins', sans-serif;
        }
        .stTextInput {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="SRE Error Helper", page_icon="🔧", layout="centered")

# Title
st.title("🔧 SRE Error Helper")

# Input box
query = st.text_input("Enter the error you're seeing:")

# When the user enters a query
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))
    
    # Display result in a card style
    st.markdown(f"""
    <div class="result-card">
        <h2><b>Best Match Found</b></h2>
        <p><b>🔑 Error Code:</b> {df.iloc[best_idx]['Error Code']}</p>
        <p><b>💬 Error Message:</b> {df.iloc[best_idx]['Error Message']}</p>
        <p><b>⚠️ Likely Cause:</b> {df.iloc[best_idx]['Cause']}</p>
        <p><b>🔧 Suggested Fix:</b> {df.iloc[best_idx]['Resolution Steps']}</p>
    </div>
    """, unsafe_allow_html=True)
