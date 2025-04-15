import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Set config FIRST
st.set_page_config(page_title="Ask Niel", layout="centered")

# Load model and data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df, model

df, model = load_data()

# Load the chat component
with open(os.path.join("chat_component", "index.html"), "r") as f:
    chat_html = f.read()

# Receive input using streamlit_component
query = components.html(chat_html, height=450)

# Optional: Show result when query is received
# We'll simulate query processing next
