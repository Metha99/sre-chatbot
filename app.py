import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Set config (must be first)
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# Load model and data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df, model

df, model = load_data()

# Show the floating chat UI
with open("index.html", "r") as f:
    components.html(f.read(), height=500)

# Get query from URL
query = st.query_params.get("q", "")

# Only respond if query exists
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Format response (used inside the chat)
    st.markdown(f"""
    <script>
        const responseText = `
        🤖 <strong>Ask Niel:</strong><br><br>
        <strong>Error Code:</strong> {df.iloc[best_idx]['Error Code']}<br>
        <strong>Error Message:</strong> {df.iloc[best_idx]['Error Message']}<br>
        <strong>Cause:</strong> {df.iloc[best_idx]['Cause']}<br>
        <strong>Fix:</strong> {df.iloc[best_idx]['Resolution Steps']}
        `;

        window.parent.postMessage({{
            type: "niel_response",
            message: responseText
        }}, "*");
    </script>
    """, unsafe_allow_html=True)
