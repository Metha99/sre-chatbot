import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the CSV
@st.cache_data
def load_data():
    df = pd.read_csv('errors.csv')
    df['embedding'] = df['error'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# UI
st.title("SRE Chatbot 🤖")
user_input = st.text_input("Describe your error or issue:")

if user_input:
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = df['embedding'].apply(lambda x: util.cos_sim(user_embedding, x).item())
    best_match = df.iloc[scores.idxmax()]
    st.subheader("Best Match 🔍")
    st.write(f"**Error:** {best_match['error']}")
    st.write(f"**Solution:** {best_match['solution']}")
