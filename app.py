import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="SRE Error Helper", page_icon="🔧", layout="centered")

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
            background: linear-gradient(135deg, #28313b, #485461); /* Gradient background */
            color: #fff; /* Light text color */
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .stTextInput input {
            font-size: 20px;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #4a90e2;
            width: 80%; /* Wider search bar */
            margin-top: 40px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            background: rgba(255, 255, 255, 0.1); /* Semi-transparent background */
            color: white; /* White text */
        }
        .stTextInput input:focus {
            border-color: #50C878; /* Emerald green focus */
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            padding: 12px 25px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #357ab7;
            transform: scale(1.05); /* Slight grow on hover */
        }
        .result-card {
            background-color: rgba(255, 255, 255, 0.15); /* Slight transparency */
            border-radius: 10px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .result-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        h1, h2 {
            font-family: 'Poppins', sans-serif;
            color: #fff;
        }
        .result-card h3 {
            color: #50C878;
        }
        .result-card p {
            color: #ccc; /* Lighter text for better contrast */
        }
        .stTextInput {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🔧 SRE Error Helper")

# Input box
query = st.text_input("Enter the error you're seeing:")

# When the user enters a query
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))
    
    # Display result in a modern card style
    st.markdown(f"""
    <div class="result-card">
        <h2><b>Best Match Found</b></h2>
        <h3><b>🔑 Error Code:</b> {df.iloc[best_idx]['Error Code']}</h3>
        <p><b>💬 Error Message:</b> {df.iloc[best_idx]['Error Message']}</p>
        <p><b>⚠️ Likely Cause:</b> {df.iloc[best_idx]['Cause']}</p>
        <p><b>🔧 Suggested Fix:</b> {df.iloc[best_idx]['Resolution Steps']}</p>
    </div>
    """, unsafe_allow_html=True)
