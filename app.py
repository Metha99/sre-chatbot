import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Make sure this is the very first Streamlit command!
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# Load model and data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df, model

df, model = load_data()

# App title
st.title("🤖 Ask Niel")

# Embed the HTML chat component from index.html in the same folder
with open("index.html", "r") as f:
    html_string = f.read()
components.html(html_string, height=500)

# Input field from the Streamlit interface (fallback if needed)
query = st.text_input("Or type here if the chat is unresponsive:", "")

# Show best match if query is entered
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    st.subheader("🔍 Best Match Found")
    st.write(f"**Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.write(f"**Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.write(f"**Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.write(f"**Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
