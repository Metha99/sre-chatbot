import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config
st.set_page_config(page_title="Ask Niel", page_icon="🤖", layout="centered")

# Page styling (your original CSS)
st.markdown("""<style> 
body {
    background: linear-gradient(135deg, #2b2b2b, #121212); 
    color: #e8e8e8;
    font-family: 'Segoe UI', sans-serif;
    margin: 0;
    padding: 0;
}

.pulse-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
}

.pulse-circle {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: radial-gradient(circle, #9c27b0, #673ab7);  /* Purplish gradient */
    box-shadow: 0 0 80px #9c27b0, 0 0 120px #673ab7, 0 0 140px #9c27b0;
    animation: pulse 1.5s infinite ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 80px #9c27b0, 0 0 120px #673ab7; }
    50% { transform: scale(1.2); box-shadow: 0 0 150px #9c27b0, 0 0 200px #673ab7; }
    100% { transform: scale(1); box-shadow: 0 0 80px #9c27b0, 0 0 120px #673ab7; }
}

input {
    background-color: #333333 !important;
    color: #ffffff !important;
    border: 1px solid #9c27b0 !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    transition: 0.3s ease;
    width: 100%;
}

input:focus {
    border-color: #9c27b0 !important;
    box-shadow: 0 0 8px 4px rgba(156, 39, 176, 0.3) !important;
}

h1 {
    text-align: center;
    color: #9c27b0;
    font-size: 2.5rem;
    font-weight: 600;
    margin-top: 30px;
}

h2, h3 {
    color: #673ab7;
}

button {
    background-color: #9c27b0;
    color: #121212;
    border-radius: 8px;
    border: none;
    padding: 12px 24px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 30px;
}

button:hover {
    background-color: #673ab7;
    color: white;
}

button:active {
    transform: scale(0.95);
}

.result-card {
    background-color: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 0 20px rgba(156, 39, 176, 0.2);
    font-size: 1.1rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.result-card h3 {
    color: #9c27b0;
    font-size: 1.3rem;
    margin-bottom: 10px;
}

.result-card p {
    color: #e8e8e8;
    line-height: 1.6;
    margin-bottom: 8px;
}

.typing-animation {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

.typing-animation span {
    width: 10px;
    height: 10px;
    margin: 0 5px;
    background-color: #9c27b0;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
}

.typing-animation span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-animation span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes blink {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
}
</style>""", unsafe_allow_html=True)

# Title and animation
st.markdown("<h1>Ask Niel 🤖</h1>", unsafe_allow_html=True)
st.markdown("""<div class="pulse-wrapper"><div class="pulse-circle"></div></div>""", unsafe_allow_html=True)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load all datasets from GitHub
@st.cache_data
def load_data():
    # Replace with your GitHub file URLs
    df_errors = pd.read_csv("https://raw.githubusercontent.com/Metha99/sre-chatbot/main/errors.csv")
    df_azure = pd.read_csv("https://raw.githubusercontent.com/Metha99/sre-chatbot/main/azure_logs.csv")
    df_gitlab = pd.read_csv("https://raw.githubusercontent.com/Metha99/sre-chatbot/main/gitlab_jobs.csv")
    df_snow = pd.read_csv("https://raw.githubusercontent.com/Metha99/sre-chatbot/main/servicenow_tickets.csv")

    # Generate embeddings for all datasets
    df_errors["embedding"] = df_errors["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    df_azure["embedding"] = df_azure["Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    df_gitlab["embedding"] = df_gitlab["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    df_snow["embedding"] = df_snow["Summary"].apply(lambda x: model.encode(x, convert_to_tensor=True))

    return df_errors, df_azure, df_gitlab, df_snow

df_errors, df_azure, df_gitlab, df_snow = load_data()

# Input from user
query = st.text_input("Ask your error here:")
typing_placeholder = st.empty()

if query:
    typing_placeholder.markdown("""
    <div class="typing-animation">
        <span></span><span></span><span></span>
    </div>
    """, unsafe_allow_html=True)

    # Create query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Find best match from each dataset
    def find_best_match(df, col="embedding"):
        scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df[col]]
        best_idx = scores.index(max(scores))
        return df.iloc[best_idx]

    best_error = find_best_match(df_errors)
    best_azure = find_best_match(df_azure)
    best_gitlab = find_best_match(df_gitlab)
    best_snow = find_best_match(df_snow)

    typing_placeholder.empty()

    # Display results
    st.markdown("### 🔍 Best Match from Error Database")
    st.markdown(f"**Error Code:** {best_error['Error Code']}")
    st.markdown(f"**Error Message:** {best_error['Error Message']}")
    st.markdown(f"**Likely Cause:** {best_error['Cause']}")
    st.markdown(f"**Suggested Fix:** {best_error['Resolution Steps']}")

    st.markdown("---")
    st.markdown("### ☁️ Related Azure Log")
    st.markdown(f"**Timestamp:** {best_azure['Timestamp']}")
    st.markdown(f"**Resource:** {best_azure['Resource']}")
    st.markdown(f"**Error Code:** {best_azure['Error Code']}")
    st.markdown(f"**Message:** {best_azure['Message']}")

    st.markdown("---")
    st.markdown("### 🚀 Related GitLab Job")
    st.markdown(f"**Job ID:** {best_gitlab['Job ID']}")
    st.markdown(f"**Status:** {best_gitlab['Status']}")
    st.markdown(f"**Pipeline ID:** {best_gitlab['Pipeline ID']}")
    st.markdown(f"**Error Message:** {best_gitlab['Error Message']}")

    st.markdown("---")
    st.markdown("### 🎫 Related ServiceNow Ticket")
    st.markdown(f"**Ticket ID:** {best_snow['Ticket ID']}")
    st.markdown(f"**Summary:** {best_snow['Summary']}")
    st.markdown(f"**Status:** {best_snow['Status']}")
    st.markdown(f"**Resolution:** {best_snow['Resolution']}")
