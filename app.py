import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Set page config (must be FIRST)
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# Load model and data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df, model

df, model = load_data()

# Inject the chat widget (index.html)
with open("index.html", "r") as f:
    components.html(f.read(), height=450)

# Read query from query string
query = st.query_params.get("q", "")

# Run only if query is provided (from the chat input)
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))

    # Return JSON for JavaScript to pick up later (optional)
    st.markdown("### 🔍 Best Match Found")
    st.markdown(f"**You asked:** {query}")
    st.markdown(f"**Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.markdown(f"**Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.markdown(f"**Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.markdown(f"**Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
